import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F
import numpy as np
import networkx as nx
from model import utils
from model.modules import *
import warnings
import torch_geometric as pyg
import torch_geometric.nn as g_nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torchdiffeq._impl as ode
import copy
warnings.filterwarnings("ignore")



##################GSI_Cell######################################

class GSI_F(MessagePassing):
    def __init__(self, args):
        super(GSI_F, self).__init__()
        self.aggr = args.agg
        ###########SI Part#############
        self.ploy_p = torch.arange(1, args.poly_p)  # 5
        self.ploy_n = -torch.arange(1, args.poly_n)  # 3
        self.activate = args.activate # 3
        self.time_stamp = args.time_stamp
        self.node_feature = args.dims
        self.num_nodes = args.num_atoms
        self.batch_size = args.batch_size
        self.device = args.device
        #self.num_func_lib = 2 * (args.poly_n + args.poly_p + args.act_gamma + 4)
        self.num_func_lib = 1 * (args.poly_n + 3 + args.poly_p)
        if self.activate:
            self.num_func_lib += 3
        ##########SI BASE COFFS##########

        #self.wf_2 = nn.Parameter(torch.cat((0.99 * torch.ones(self.num_func_lib, 1), 0.01 * torch.ones(self.num_func_lib, 1)),0), requires_grad=True)
        self.wf_2 = nn.Parameter(torch.cat((0.99 * torch.ones(self.num_func_lib, 1), 0.01 * torch.ones(self.num_func_lib, 1)),0), requires_grad=True)

    def forward(self, t, x, batchs):
        #edge_index = batchs.edge_index
        wf_1 =batchs.f_mask
        if not isinstance(wf_1, int):
            wf_1 = torch.cat((wf_1,wf_1))
        wf_2 = self.wf_2
        out = self.update(x=x, wf_1 = wf_1, wf_2 = wf_2)
        return out




    def update(self, x, wf_1, wf_2):
        with torch.no_grad(): 
            F_msg = utils.fun_lib(x, self.ploy_p, self.ploy_n, self.device, activate = self.activate)
            F_msg = torch.cat((F_msg, -F_msg), 1)
        F_w = wf_1 * wf_2
        out = torch.mm(F_msg, F_w)
        return out

class GSI_C(MessagePassing):
    def __init__(self, args):
        super(GSI_C, self).__init__()
        self.aggr = args.agg
        ###########SI Part#############
        self.ploy_p = torch.arange(1, args.poly_p)  # 5
        self.ploy_n = -torch.arange(1, args.poly_n)  # 3
        self.activate = args.activate # 3
        self.time_stamp = args.time_stamp
        self.node_feature = args.dims
        self.num_nodes = args.num_atoms
        self.batch_size = args.batch_size
        self.device = args.device
        #self.num_coupled_fun_lib = 2 * (3 * (args.poly_p - 1) + 4 * (args.poly_n - 1) + 8 + 4 + 4 * (args.act_gamma + 1))
        self.num_coupled_fun_lib = 1 * (3 * (args.poly_p - 1) + 4 * (args.poly_n - 1) + 8 + 4)
        if self.activate:
            self.num_coupled_fun_lib += 12
        self.UseEdgeAttr = args.UseEdgeAttr
        ##########SI BASE COFFS##########
        # self.wc_2 = nn.Parameter(torch.cat((0.99 * torch.ones(self.num_coupled_fun_lib, 1), 0.01 * torch.ones(self.num_coupled_fun_lib, 1)),0),
        #                              requires_grad=True) 
        self.wc_2 = nn.Parameter(torch.cat((0.99 * torch.ones(self.num_coupled_fun_lib, 1), 0.01 * torch.ones(self.num_coupled_fun_lib, 1)),0),
                                     requires_grad=True) 
        ###########Edge_Attr####################
        if self.UseEdgeAttr:
            self.edge_attr_all = nn.Parameter(
                torch.randn(self.num_nodes, self.num_nodes, requires_grad=True)
            )
            torch.nn.init.xavier_uniform_(self.edge_attr_all)



    def forward(self, t, x, batchs):
        edge_index = batchs.edge_index
        if self.UseEdgeAttr:
            edge_attr = self.edge_attr_all[edge_index[0]%self.num_nodes, edge_index[1]%self.num_nodes].reshape(-1,1)
        else:
            edge_attr = 1
        wc_1 = batchs.c_mask
        if not isinstance(wc_1, int):
            wc_1 = torch.cat((wc_1,wc_1))

        wc_2 = self.wc_2
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, wc_1 = wc_1, wc_2 = wc_2)
        return out
        #return x

    def message(self, x_i, x_j, edge_attr, wc_1, wc_2):
        # dxi/dt = \sum kij * G(xi, xj) * Aij + F(xi)
        with torch.no_grad(): 
            C_msg = utils.coupled_fun_lib(x_i, x_j, self.ploy_p, self.ploy_n, self.device, activate= self.activate)
            C_msg = torch.cat((C_msg, -C_msg),1)
        C_w = wc_1 * wc_2
        out = edge_attr * torch.mm(C_msg, C_w)
        return out


    def update(self, aggr_out):
        return aggr_out

class GSICell(nn.Module):
    def __init__(self, args):
        super(GSICell, self).__init__()
        self.GSI_C = GSI_C(args)
        self.GSI_F = GSI_F(args)
        self.F_coef = args.F_coef
        self.UseF = args.UseF
    def forward(self,t,x, batchs):
        if self.UseF:
            C = self.GSI_C(t, x, batchs)
            F = self.GSI_F(t, x, batchs)
            return C + self.F_coef * F
        else:
            return self.GSI_C(t, x, batchs)


##################GSI##########################

class DGSIDecoder(nn.Module):
    def __init__(self, args):
        super(DGSIDecoder, self).__init__()
        self.teacher = args.teacher
        self.time_stamp = args.time_stamp
        self.activate = args.activate
        self.device = args.device
        self.num_func_lib = 1 * (args.poly_n + args.poly_p + 3)
        self.num_coupled_fun_lib = 1 * (3 * (args.poly_p - 1) + 4 * (args.poly_n - 1) + 8 + 4)
        if self.activate:
            self.num_func_lib += 3
            self.num_coupled_fun_lib += 12
        self.k = args.k
        ############SI Cell################
        self.GSICell = GSICell(args)



    def single_step_forward(self, t, batchs, step_x):

        x_dot = self.GSICell(t, step_x, batchs)

        return x_dot * t + step_x
    
    
    def forward(self, t, batchs, c_mask =None, f_mask =None):
        out = []
        batchs.k = self.k
        if c_mask is not None:
            batchs.c_mask = c_mask.to(self.device)
            batchs.f_mask = f_mask.to(self.device)
        else:
            batchs.c_mask = torch.ones(self.num_coupled_fun_lib,1).to(self.device)
            batchs.f_mask = torch.ones(self.num_func_lib,1).to(self.device)
        for i in range(self.time_stamp - 1):
            if i%self.teacher == 0:            
                step_x = batchs.x[:,i,:]
            else:
                step_x = out[-1]
            out.append(self.single_step_forward(torch.diff(t)[0], batchs, step_x))

        out = torch.stack(out,1)

        wc_2 = self.GSICell.GSI_C.wc_2.squeeze()
        wf_2 = self.GSICell.GSI_F.wf_2.squeeze()
        wc = -wc_2.reshape(2,-1).T.diff().squeeze() * batchs.c_mask.squeeze()
        wf = -wf_2.reshape(2,-1).T.diff().squeeze() * batchs.f_mask.squeeze()
     
        return out, wc, wf

############ODEs#################
class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=True): 
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x, para):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector, para= para,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector, para = para,
                             rtol=self.rtol, atol=self.atol, method=self.method)
     
        return out[-1] if self.terminal else out  




class CGSIDecoder(nn.Module):
    '''
    TO DO, CGSI need the paras updated together.
    '''
    def __init__(self, args, rtol=.01, 
                atol=.001, 
                method='dopri5'):
            super(CGSIDecoder, self).__init__()
            self.teacher = args.teacher
            self.time_stamp = args.time_stamp
            self.device = args.device
            self.num_func_lib = 1 * (args.poly_n + args.poly_p + 3)
            self.num_coupled_fun_lib = 1 * (3 * (args.poly_p - 1) + 4 * (args.poly_n - 1) + 8 + 4)
            self.k = args.k
            ############SI Cell################
            self.GSICell = GSICell(args)            
            ###########ODEs#########################
            self.rtol = rtol
            self.atol = atol
            self.method = method
            self.adjoint = args.adjoint
            self.ODEFunction = GSICell(args).to(args.device)  # OM
            self.neural_dynamic_layer = ODEBlock(
                    self.ODEFunction,
                    rtol= self.rtol, atol= self.atol, method= self.method, adjoint=self.adjoint).to(args.device) 


    def single_step_forward(self, t, batchs, step_x):

        return self.neural_dynamic_layer(t, step_x, batchs)
    
    def forward(self, t, batchs, c_mask =None, f_mask =None):
        out = []
        batchs.k = self.k
        if c_mask is not None:
            batchs.c_mask = c_mask.to(self.device)
            batchs.f_mask = f_mask.to(self.device)
        else:
            batchs.c_mask = torch.ones(self.num_coupled_fun_lib,1).to(self.device)
            batchs.f_mask = torch.ones(self.num_func_lib,1).to(self.device)
        #step odes
        if self.time_stamp% self.teacher == 0:
            epochs = self.time_stamp//self.teacher
        else:
            epochs = self.time_stamp//self.teacher + 1

        for i in range(epochs):
            step_x = batchs.x[:,i*self.teacher,:]
            if i == self.time_stamp//self.teacher:
                vt = t[i * self.teacher:]
            else:
                vt = t[i * self.teacher: (i + 1) * self.teacher]

            if len(vt) == 1:
                continue
            pred_ = self.single_step_forward(vt, batchs, step_x)
            if i == 0:
                pred = pred_
            else:
                pred = torch.cat((pred, pred_), 0)
        
        output = pred[1:, :, :].permute(1,0,2)
        # for i in range(len(t) - 1):
        #     batchs.neighbor = batchs.x[:,i,:]
        #     if i%self.teacher == 0:            
        #         step_x = batchs.x[:,i,[batchs.k]]
        #     else:
        #         step_x = out[-1]
        #     vt = t[i: i + 2]
        #     out.append(self.single_step_forward(vt, batchs, step_x)[-1])

        
        # output = torch.stack(out,1)


        wc_2 = self.ODEFunction.GSI_C.wc_2.squeeze()
        wf_2 = self.ODEFunction.GSI_F.wf_2.squeeze()
        wc = -wc_2.reshape(2,-1).T.diff().squeeze() * batchs.c_mask.squeeze()
        wf = -wf_2.reshape(2,-1).T.diff().squeeze() * batchs.f_mask.squeeze()
        return output, wc, wf




if __name__ == '__main__':
    pass



