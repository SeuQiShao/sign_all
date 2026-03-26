import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from model import utils
from model.modules import *
import warnings
from torch_geometric.nn import MessagePassing
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint


##################GSI_Cell######################################
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GSICell(MessagePassing):
    """
    GSICell 支持按维度独立生成基函数（mask 选择），并一次性输出 D 维。

    要求：
      - x: [N, D]
      - batchs.f_mask: [Mf, D]
      - batchs.c_mask: [Mc, D]
      - batchs.edge_index: standard edge index
      - 可选 batchs.k 等其他字段由原结构保持

    设计要点：
      - 基函数生成函数 utils.fun_lib / utils.coupled_fun_lib **不** 要求能一次性处理多维输入。
        本实现会对每个输出维度 d 单独调用这些函数，函数会在 mask 的作用下只返回被选择的基函数列，
        从而避免在内存中生成完整的大基函数库。
      - wf_2 / wc_2 的形状为 (2*Mf, D) 和 (2*Mc, D)，保持原有 ± 拼接策略（基函数和其取负）。
      - message & forward 都按维度循环，但对于每个维度只处理被 mask 选中的基函数列，从而节约显存。
    """

    def __init__(self, args):
        super(GSICell, self).__init__(aggr=args.agg)
        self.args = args

        # 基本设置
        self.poly_p = args.poly_p
        self.poly_n = args.poly_n
        self.activate = args.activate
        self.device = args.device
        self.num_nodes = args.num_atoms
        self.D = args.dims

        # 估算基函数数目（仅用 names=True 获取长度）
        f_basis = utils.fun_lib(torch.empty(0, args.dims), self.poly_p, self.poly_n, activate=self.activate, device="cpu", names=True)
        c_basis = utils.coupled_fun_lib(None, None, self.poly_p, self.poly_n, activate=self.activate, device="cpu", names=True)
        self.Mf = len(f_basis)
        self.Mc = len(c_basis)

        # 可学习参数：保留 ± 拼接 -> (2*M, D)
        self.wf_2 = nn.Parameter(
            torch.cat((0.99 * torch.ones(self.Mf, self.D),
                       0.01 * torch.ones(self.Mf, self.D)), dim=0),
            requires_grad=True
        )
        self.wc_2 = nn.Parameter(
            torch.cat((0.99 * torch.ones(self.Mc, self.D),
                       0.01 * torch.ones(self.Mc, self.D)), dim=0),
            requires_grad=True
        )

        # 可选边属性矩阵
        self.UseEdgeAttr = getattr(args, "UseEdgeAttr", False)
        if self.UseEdgeAttr:
            self.edge_attr_all = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes))
            nn.init.xavier_uniform_(self.edge_attr_all)

        # 内存优化选项
        self.memory_efficient = getattr(args, "memory_efficient", True)
        # 若需要可以设置 chunk_size > 1 将维度分块处理
        self.chunk_size = getattr(args, "chunk_size", None)

    def forward(self, t, x, batchs):
        """
        x: [N, D]
        batchs.f_mask: [Mf, D]
        batchs.c_mask: [Mc, D]
        返回: [N, D]
        """
        device = x.device
        edge_index = batchs.edge_index
        k = getattr(batchs, "k", None)

        # 边属性
        if self.UseEdgeAttr:
            edge_attr = self.edge_attr_all[edge_index[0] % self.num_nodes, edge_index[1] % self.num_nodes].view(-1, 1)
        else:
            edge_attr = 1.0

        # C 部分通过 propagate 聚合，message 会处理每个维度的基函数
        c_out = self.propagate(edge_index, x=x, edge_attr=edge_attr, wc_1=batchs.c_mask, k=k)

        # F 部分：对每个维度单独调用 utils.fun_lib
        f_mask = batchs.f_mask.to(device)  # [Mf, D]

        N = x.shape[0]
        D = self.D
        f_out = torch.zeros(N, D, device=device)

        # 预计算每列的 active index 与 mask values，避免在循环内重复转换
        f_active_idx = []  # list of tensors of selected indices for each d
        f_mask_vals = []   # list of tensors of mask values for selected indices per d
        for d in range(D):
            col = f_mask[:, d]
            sel = torch.nonzero(col.abs() > 0, as_tuple=True)[0]
            if sel.numel() == 0:
                f_active_idx.append(None)
                f_mask_vals.append(None)
            else:
                f_active_idx.append(sel)
                f_mask_vals.append(col[sel])

        # 对每个维度单独计算基函数并乘权
        # 这里使用 no_grad 计算基函数，遵循你原先设计（基函数本身不学）
        for d in range(D):
            sel = f_active_idx[d]
            if sel is None:
                continue

            # 选出的基础基序（长度 r）对应 utils.fun_lib 的 mask
            mask_full = torch.zeros(self.Mf, device=device)
            mask_full[sel] = 1.0

            with torch.no_grad():
                # 注意：传入的 x_slice 形状 [N,1]（列向量），由于 fun_lib 只返回被 mask 选中的列，节省显存
                #x_slice = x[:, d:d+1]
                F_basis = utils.fun_lib(x, self.poly_p, self.poly_n, self.device, self.activate, mask=mask_full)
                # F_basis: [N, r]
                if F_basis.numel() == 0:
                    continue
                F_basis_pm = torch.cat([F_basis, -F_basis], dim=1)  # [N, 2*r]

            # 构造权重：对应 sel 的行在 wf_2 中的位置 + sel+Mf（正负拼接）
            # wf_2 是 [2*Mf, D]，我们要选出 2*r 个行并取第 d 列
            pos = sel
            neg = sel + self.Mf
            posneg = torch.cat([pos, neg], dim=0)

            # mask values对应 sel (r,) -> 复制为 ± (2*r,)
            mask_vals = f_mask_vals[d]
            mask_pm = torch.cat([mask_vals, mask_vals], dim=0).to(device)  # [2*r]

            wf_cols = self.wf_2[posneg, d].to(device)  # [2*r]
            weights = mask_pm * wf_cols  # [2*r]

            # 计算 F_basis_pm [N,2*r] @ weights [2*r] -> [N]
            f_out[:, d] = F_basis_pm @ weights

        return c_out + f_out

    def message(self, x_i, x_j, edge_attr, wc_1, k):
        """
        x_i, x_j: [E, D]
        wc_1: [Mc, D]
        返回: [E, D]
        """
        device = x_i.device
        E = x_i.shape[0]
        D = self.D

        # 预处理 wc mask per-dim
        wc = wc_1.to(device)
        wc_active_idx = []
        wc_mask_vals = []
        for d in range(D):
            col = wc[:, d]
            sel = torch.nonzero(col.abs() > 0, as_tuple=True)[0]
            if sel.numel() == 0:
                wc_active_idx.append(None)
                wc_mask_vals.append(None)
            else:
                wc_active_idx.append(sel)
                wc_mask_vals.append(col[sel])

        out = torch.zeros(E, D, device=device)

        # 对每个维度单独计算耦合基函数
        for d in range(D):
            sel = wc_active_idx[d]
            if sel is None:
                continue

            # 构造完整 mask 传入 coupled_fun_lib
            mask_full = torch.zeros(self.Mc, device=device)
            mask_full[sel] = 1.0

            with torch.no_grad():
                # x_i[:, d:d+1], x_j[:, d:d+1] -> 返回 [E, r]
                C_basis = utils.coupled_fun_lib(x_i[:, d:d+1], x_j[:, d:d+1], self.poly_p, self.poly_n, self.device, self.activate, mask=mask_full)
                if C_basis.numel() == 0:
                    continue
                C_basis_pm = torch.cat([C_basis, -C_basis], dim=1)  # [E, 2*r]

            # 构造权重（同样 pos/neg）
            pos = sel
            neg = sel + self.Mc
            posneg = torch.cat([pos, neg], dim=0)

            mask_vals = wc_mask_vals[d]
            mask_pm = torch.cat([mask_vals, mask_vals], dim=0).to(device)  # [2*r]
            wc_cols = self.wc_2[posneg, d].to(device)  # [2*r]
            weights = mask_pm * wc_cols  # [2*r]

            # 计算 C_basis_pm [E,2*r] @ weights [2*r] -> [E]
            c_d = C_basis_pm @ weights

            # 乘以边属性
            if not torch.is_tensor(edge_attr):
                ea = torch.ones(E, device=device) * float(edge_attr)
            else:
                ea = edge_attr.view(-1).to(device)

            out[:, d] = ea * c_d

        return out

    def update(self, aggr_out):
        return aggr_out


# class GSICell(MessagePassing):
#     def __init__(self, args):
#         # 确定聚合方式（mean, add, max等）
#         super(GSICell, self).__init__(aggr=args.agg)
#         self.args = args
        
#         ########### 公共参数 #############
#         self.poly_p = args.poly_p
#         self.poly_n = args.poly_n
#         self.activate = args.activate
#         self.device = args.device
#         self.num_nodes = args.num_atoms
#         f_basis = utils.fun_lib(torch.empty(0,args.dims), args.poly_p, args.poly_n, activate=args.activate, device="cpu", names=True)
#         c_basis = utils.coupled_fun_lib(None, None, args.poly_p, args.poly_n, device="cpu", activate=args.activate,  names=True)
#         self.num_func_lib = len(f_basis)
#         self.num_coupled_fun_lib = len(c_basis)
        
#         ########### F部分参数 #############
#         # self.num_func_lib = 1 * (args.poly_n + 3 + args.poly_p)
#         # if self.activate:
#         #     self.num_func_lib += 3
#         # F部分系数矩阵
#         self.wf_2 = nn.Parameter(
#             torch.cat((0.99*torch.ones(self.num_func_lib,1), 
#                       0.01*torch.ones(self.num_func_lib,1)), dim=0), 
#             requires_grad=True
#         )
        
#         ########### C部分参数 #############
#         # self.num_coupled_fun_lib = 1*(3*(args.poly_p-1)+4*(args.poly_n-1)+8+4)
#         # if self.activate:
#         #     self.num_coupled_fun_lib += 12
#         # C部分系数矩阵
#         self.wc_2 = nn.Parameter(
#             torch.cat((0.99*torch.ones(self.num_coupled_fun_lib,1),
#                      0.01*torch.ones(self.num_coupled_fun_lib,1)), dim=0),
#             requires_grad=True
#         )
#         # 边属性处理
#         self.UseEdgeAttr = args.UseEdgeAttr
#         if self.UseEdgeAttr:
#             self.edge_attr_all = nn.Parameter(
#                 torch.randn(self.num_nodes, self.num_nodes)
#             )
#             nn.init.xavier_uniform_(self.edge_attr_all)

#     def forward(self, t, x, batchs):
#         #========= 耦合部分计算 =========
#         edge_index = batchs.edge_index
        
#         # 准备C部分参数
#         wc_1 = batchs.c_mask
#         k = batchs.k
#         # mask_index = (wc_1.abs() > 0)
#         # wc_1 = wc_1[mask_index]
#         # if wc_1.dim() > 0:  # 非标量时拼接
#         #     wc_1 = torch.cat([wc_1, wc_1])
            
#         # 边属性处理
#         if self.UseEdgeAttr:
#             edge_attr = self.edge_attr_all[edge_index[0]%self.num_nodes, 
#                                         edge_index[1]%self.num_nodes].view(-1,1)
#         else:
#             edge_attr = 1.0
            
#         # 消息传播
#         c_out = self.propagate(
#             edge_index, 
#             x=x, 
#             edge_attr=edge_attr,
#             wc_1=wc_1,
#             k = k
#         )
        
#         #========= 节点自身动力学 =========

#         wf_1 = batchs.f_mask

#         # 计算F部分
#         with torch.no_grad():
#             F_msg = utils.fun_lib(batchs.neighbor, self.poly_p, self.poly_n, 
#                                     self.device, self.activate, mask = wf_1.detach().squeeze())
#             #F_msg = F_msg[:, f_mask_index.squeeze()]
#             F_msg = torch.cat([F_msg, -F_msg], dim=1)
#         f_mask_index = (wf_1.abs() > 0)
#         wf_1 = wf_1[f_mask_index]
#         if wf_1.dim() > 0:  # 非标量时拼接
#             wf_1 = torch.cat([wf_1, wf_1])
#         extended_mask = torch.cat([f_mask_index, f_mask_index], dim=0)
#         F_weights = wf_1 * self.wf_2[extended_mask]
#         f_out = torch.mm(F_msg, F_weights.unsqueeze(1))

#         return c_out + f_out


#     def message(self, x_i, x_j, edge_attr, wc_1, k):
#         """耦合消息生成"""
#         # 生成耦合项基函数
#         with torch.no_grad():
#             C_msg = utils.coupled_fun_lib(x_i, x_j, self.poly_p, self.poly_n,
#                                          self.device, self.activate, mask = wc_1.detach().squeeze())
#             #C_msg = C_msg[:, mask_index.squeeze()]
#             C_msg = torch.cat([C_msg, -C_msg], dim=1)
            
#         # 权重处理
#         mask_index = (wc_1.abs() > 0)
#         wc_1 = wc_1[mask_index]
#         if wc_1.dim() > 0:  # 非标量时拼接
#             wc_1 = torch.cat([wc_1, wc_1])
#         extended_mask = torch.cat([mask_index, mask_index], dim=0)
#         C_weights = wc_1 * self.wc_2[extended_mask]
        
#         # 计算带权消息
#         return edge_attr * torch.mm(C_msg, C_weights.unsqueeze(1))

#     def update(self, aggr_out):
#         """直接返回聚合结果"""
#         return aggr_out



##################GSI##########################

class DGSIDecoder(nn.Module):
    def __init__(self, args):
        super(DGSIDecoder, self).__init__()
        self.teacher = args.teacher
        self.time_stamp = args.time_stamp
        self.activate = args.activate
        self.device = args.device
        self.dims = args.dims
        # self.k = args.k
        ############SI Cell################
        self.GSICell = GSICell(args)
        self.num_func_lib = self.GSICell.Mf
        self.num_coupled_fun_lib = self.GSICell.Mc



    def single_step_forward(self, t, batchs, step_x):

        x_dot = self.GSICell(t, step_x, batchs)

        return x_dot * t + step_x
    
    
    def forward(self, t, batchs, c_mask =None, f_mask =None):
        out = []
        if c_mask is not None:
            batchs.c_mask = c_mask.to(self.device)
            batchs.f_mask = f_mask.to(self.device)
        else:
            batchs.c_mask = torch.ones(self.num_coupled_fun_lib,self.dims).to(self.device)
            batchs.f_mask = torch.ones(self.num_func_lib,self.dims).to(self.device)
        total_steps = batchs.x.shape[1]
        start_step = 1
        if batchs.train:
            start_step = torch.randint(0, total_steps - 200, (1,)).item()
            for i in range(start_step, start_step + 200):
                if i == start_step:
                    step_x = batchs.x[:,i,:]
                else:
                    step_x = out[-1]
                out.append(self.single_step_forward(torch.diff(t)[0], batchs, step_x))
        else:
            for i in range(total_steps - 1):
                if i%self.teacher == 0:            
                    step_x = batchs.x[:,i,:]
                else:
                    step_x = out[-1]
                out.append(self.single_step_forward(torch.diff(t)[0], batchs, step_x))

        out = torch.stack(out,1)

        wc_2 = self.GSICell.wc_2.squeeze()
        wf_2 = self.GSICell.wf_2.squeeze()
        wc = -wc_2.reshape(2,-1, out.shape[-1]).T.diff().squeeze().T * batchs.c_mask.squeeze()
        wf = -wf_2.reshape(2,-1,out.shape[-1]).T.diff().squeeze().T * batchs.f_mask.squeeze()
     
        return out, wc, wf, start_step

############ODEs#################


######################## 参数化ODE包装器 ########################
class ParametricODE(nn.Module):
    """带参数持久化的ODE函数包装器"""
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc  # 原始GSICell实例
        self.current_batchs = None  # 参数存储

    def forward(self, t, x):
        # 此时t可能为标量或张量，需统一处理
        if isinstance(t, torch.Tensor) and t.dim() > 0:
            # 处理时间序列输入
            return self.odefunc(t[0], x, self.current_batchs)  # 使用当前batch参数
        else:
            return self.odefunc(t, x, self.current_batchs)

    def set_batchs(self, batchs):
        """注入当前批次的参数"""
        self.current_batchs = batchs

######################## 核心修改2：ODE求解模块 ########################
class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-3, atol=1e-4, method='dopri5', adjoint=False):
        super().__init__()
        self.odefunc = odefunc  # 接收ParametricODE实例
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint

    def forward(self, vt, x, batchs):
        # 注入当前批次参数
        self.odefunc.set_batchs(batchs)
        
        # 转换时间类型
        integration_time = vt.type_as(x)
        
        # 执行ODE求解
        if self.adjoint:
            solution = odeint_adjoint(
                self.odefunc, 
                x, 
                integration_time,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method
            )
        else:
            solution = odeint(
                self.odefunc, 
                x, 
                integration_time,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method
            )
        
        # 返回完整时间序列结果
        return solution

######################## 核心修改3：完整解码器结构 ########################
class CGSIDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 共享参数的核心组件
        self.gsicell = GSICell(args)
        self.parametric_ode = ParametricODE(self.gsicell)
        
        # ODE求解配置
        self.neural_dynamic_layer = ODEBlock(
            self.parametric_ode,
            # rtol=args.rtol,
            # atol=args.atol,
            # method=args.method,
            # adjoint=args.adjoint
        )
        
        # 其他必要参数
        self.teacher = args.teacher
        self.time_stamp = args.time_stamp
        self.device = args.device
        
        # 初始化掩码参数
        self.register_buffer('c_mask', torch.ones(self.gsicell.num_coupled_fun_lib, 1))
        self.register_buffer('f_mask', torch.ones(self.gsicell.num_func_lib, 1))

    def forward(self, t, batchs, c_mask=None, f_mask=None):
        """完整前向传播流程"""
        # 1. 掩码处理
        if c_mask is not None:
            batchs.c_mask = c_mask.to(self.device)
            batchs.f_mask = f_mask.to(self.device)
        else:
            batchs.c_mask = self.c_mask
            batchs.f_mask = self.f_mask

        # 2. 时间窗口分割
        all_preds = []
        total_steps = batchs.x.shape[1] # 总时间步数
        teacher_interval = self.teacher
        
        # 计算需要的时间窗口数量
        num_windows = (total_steps + teacher_interval - 1) // teacher_interval
        
        for window_idx in range(num_windows):
            # 安全计算时间切片范围
            start_step = window_idx * teacher_interval
            end_step = min((window_idx + 1) * teacher_interval, total_steps)
            
            # 获取当前时间窗口
            window_times = t[start_step:end_step]
            
            # 至少需要两个时间点才能进行积分
            if len(window_times) < 2:
                continue
            
            # 3. 获取初始状态
            if window_idx == 0:
                # 首窗口使用真实初始值
                pred_times = 0
                x0 = batchs.x[:, start_step, :]
            else:
                # 后续窗口使用前窗口的末状态
                x0 = batchs.x[:, pred_times, :].detach()  # 截断梯度流
            
            # 4. 执行ODE求解
            window_pred = self.neural_dynamic_layer(
                vt=window_times,
                x=x0,
                batchs=batchs
            )  # 返回形状 [T, B, D]
            
            # 5. 拼接结果
            if window_idx == 0:
                all_preds.append(window_pred)  # 包含初始状态
            else:
                all_preds.append(window_pred)  # 去重连接点
            
            # 保存末状态供下次使用
            pred_times = pred_times + len(window_times)

        
        # 6. 结果聚合
        full_pred = torch.cat(all_preds, dim=0)  # 形状 [Total_T, B, D]
        
        # 转置为 [B, Total_T, D]
        output = full_pred.permute(1, 0, 2)
        
        # 7. 系数计算（保持原逻辑）
        wc_2 = self.gsicell.wc_2.squeeze()
        wf_2 = self.gsicell.wf_2.squeeze()
        wc = -wc_2.reshape(2, -1).T.diff().squeeze() * batchs.c_mask.squeeze()
        wf = -wf_2.reshape(2, -1).T.diff().squeeze() * batchs.f_mask.squeeze()
        
        return output, wc, wf  # 输出与输入时间对齐


if __name__ == '__main__':
    pass



