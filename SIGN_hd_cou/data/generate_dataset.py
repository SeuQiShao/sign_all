import torch
import torchdiffeq
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, from_networkx
from dynamics import *
import argparse
from model.utils import get_edgeindex_4_edges, R_syn
from mpl_toolkits.mplot3d import Axes3D  # 引入 3D 绘图工具

class DatasetGenerator:
    def __init__(self, args):
        self.args = args
        


    def generate_timestamp(self, mode='uniform'):
        num_points = self.args.num_points
        T = self.args.sample_interval * num_points
        if mode == 'uniform':
            return torch.linspace(0, T, num_points, device=self.args.device)
        elif mode == 'random':
            return torch.sort(torch.rand(num_points, device=self.args.device) * T)[0]
        else:
            raise ValueError("Invalid mode. Choose 'uniform' or 'random'.")

    def generate_relations(self):
        n = self.args.node_num
        seed = np.random.randint(0, 9999)
        network_type = self.args.network

        if network_type == 'random':
            G = nx.erdos_renyi_graph(n, 0.6, seed=seed)
        elif network_type == 'power_law':
            G = nx.barabasi_albert_graph(n, 3, seed=seed)
        elif network_type == 'small_world':
            G = nx.newman_watts_strogatz_graph(n, 3, 0.3, seed=seed)
        elif network_type == 'from_file':
            if self.args.file_path is None:
                raise ValueError("For 'from_file' network type, a valid file_path must be provided.")
            try:
                edge_index, n = get_edgeindex_4_edges(self.args.file_path)
                if 'bn-human' in self.args.file_path:
                    sub_edge_indices, sub_num_nodes = partition_graph_pyg(edge_index, n, num_parts=2)
                    edge_index = sub_edge_indices[0]
                    n = sub_num_nodes[0]
                    print('\n Graph partition finished! Edgeindex shape:', edge_index.shape, ' Node_num:', n)
            except Exception as e:
                raise ValueError(f"Error loading graph from file {self.args.file_path}: {e}")
        else:
            raise ValueError("Invalid network type. Choose 'random', 'power_law', or 'small_world'.")

        self.node_num = n
        if network_type != 'from_file':
            # Get edge_index from NetworkX graph
            if self.args.undirected:
                edge_index = to_undirected(from_networkx(G).edge_index).to(self.args.device)
            else:
                edge_index = from_networkx(G).edge_index.to(device=self.args.device)


        return edge_index

    def initialize_model(self, edge_index):
        # 根据 model_name 初始化动力学模型
        if self.args.model_name == 'FHN':
            self.model = FitzHughDynamics(edge_index, stre=self.args.stre).to(self.args.device)
            self.para_a = self.model.a
            self.para_b = self.model.b
            self.para_c = self.model.c
            self.para_e = self.model.e
            self.para_v = self.model.v
            self.para = [self.para_a, self.para_b, self.para_c, self.para_e, self.para_v]
        
        elif self.args.model_name == 'HR':
            self.model = HRDynamics(edge_index).to(self.args.device)
            self.para_a = self.model.a
            self.para_b = self.model.b
            self.para_c = self.model.c
            self.para_e = self.model.e
            self.para_s = self.model.s
            self.para_I = self.model.I
            self.para_mu = self.model.mu
            self.para = [self.para_a, self.para_b, self.para_c, self.para_e, self.para_s, self.para_mu, self.para_I]

        elif self.args.model_name == 'Rossler':
            self.model = RosslerDynamics(edge_index).to(self.args.device)
            self.para_a = self.model.a
            self.para_b = self.model.b
            self.para_c = self.model.c
            self.para_e = self.model.e
            self.para_w = self.model.w
            self.para = [self.para_a, self.para_b, self.para_c, self.para_e, self.para_w]
                         
        elif self.args.model_name == 'TestModel':
            #self.model = HyperEdgeDynamics(edge_index, B).to(self.args.device)
            pass
        else:
            raise ValueError(f"Model '{self.args.model_name}' is not recognized.")

    def generate_single_sample(self, x0, timestamps):
        x0 = x0.to(self.args.device)
        timestamps = timestamps.to(self.args.device)
        with torch.no_grad():
            solution = torchdiffeq.odeint(self.model, x0, timestamps, method=self.args.method).to(self.args.device)
        return solution


    def plot_single_sample(self, solutions, variable_names=None):
        solution = solutions.x.permute(1, 0, 2)
        timestamps = solutions.t
        num_nodes, num_dims = solution.shape[1], solution.shape[2]

        # 节点选择逻辑（统一处理所有维度情况）
        if num_nodes > self.args.plot_node_num:
            selected_nodes = torch.randperm(num_nodes)[:self.args.plot_node_num].tolist()
        else:
            selected_nodes = list(range(num_nodes))

        # 图形布局配置
        plot_phase_space = num_dims >= 2
        num_main_plots = num_dims + (1 if plot_phase_space else 0)
        fig = plt.figure(figsize=(5 * num_main_plots, 4)) if num_main_plots > 0 else plt.figure()
        fig.suptitle(f"{self.args.model_name} - Dynamics Visualization", fontsize=12)

        # 子图创建（自动处理3D投影）
        axs = []
        if num_main_plots > 0:
            try:  # 尝试创建常规子图布局
                fig, temp_axs = plt.subplots(1, num_main_plots, figsize=(5*num_main_plots, 4), squeeze=False)
                axs = temp_axs[0].tolist()
            except TypeError:  # 处理单子图情况
                axs = [fig.add_subplot(111)]

            # 3D投影特殊处理
            if plot_phase_space and num_dims >= 3:
                axs[-1].remove()
                axs[-1] = fig.add_subplot(1, num_main_plots, num_main_plots, projection='3d')

        # 时间序列绘制（统一处理所有维度）
        for dim in range(num_dims):
            ax = axs[dim]
            for node in selected_nodes:
                ax.plot(timestamps.cpu().numpy(),
                        solution[:, node, dim].cpu().numpy(),
                        label=f"Node {variable_names[node] if variable_names else node+1}")
            ax.set(xlabel="Time", ylabel=f"X{dim+1}", title=f"X{dim+1} Time Series")
            ax.legend(loc='best')

        # 相空间轨迹绘制（智能维度处理）
        if plot_phase_space:
            ax = axs[-1]
            data = {i: solution[:, i].cpu().numpy() for i in selected_nodes}
            
            if num_dims >= 3:
                for node, traj in data.items():
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                            label=f"Node {variable_names[node] if variable_names else node+1}")
                ax.set(xlabel="X1", ylabel="X2", zlabel="X3", title="3D Phase Space")
            else:
                for node, traj in data.items():
                    ax.plot(traj[:, 0], traj[:, 1],
                            label=f"Node {variable_names[node] if variable_names else node+1}")
                ax.set(xlabel="X1", ylabel="X2", title="Phase Portrait")

            ax.legend(loc='best')
            if num_dims >= 3:  # 优化3D视角
                ax.view_init(elev=15, azim=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    def generate_batch_sample(self):
        data_list = []
        for _ in tqdm(range(self.args.sample_num)):           
            # 生成时间戳
            timestamps = self.generate_timestamp()

            # 生成图结构和高阶交互张量
            edge_index = self.generate_relations()

            # 初值 x0 的维度为 (node_num, init_dim)，并乘以 init_scale 调整大小
            x0 = torch.rand((self.node_num, self.args.init_dim), device=self.args.device) * torch.tensor(self.args.init_scale, device=self.args.device)

            # 初始化动力学模型
            self.initialize_model(edge_index)
            
            # 生成该初值和时间戳对应的时序数据
            solution = self.generate_single_sample(x0, timestamps)
            syn = R_syn(solution.permute(1,0,2))
            print('syn of {}: {}'.format(self.args.stre, syn))
            # 将数据存储为 PyG Data 格式
            data = Data(
                x=solution.permute(1,0,2),
                edge_index=edge_index,
                t=timestamps,
                para = self.para,
                ayn = syn
            )
            data_list.append(data)



        # 保存数据
        self.suffix = '{}_{}_{}_{}_{}_{}'.format(self.args.model_name, self.node_num, self.args.network, self.args.sample_interval, self.args.num_points, self.args.stre)
        # 创建模型名称的子目录
        model_name_dir = os.path.join(self.args.save_path, self.suffix, 'raw')
        os.makedirs(model_name_dir, exist_ok=True)  # 创建目录（如果不存在）

        # 创建可视化图片的保存目录
        visualization_path = os.path.join(self.args.save_path, 'visualizations', self.suffix)
        os.makedirs(visualization_path, exist_ok=True)  # 创建目录（如果不存在）

        torch.save(data_list, os.path.join(model_name_dir, f'data_{self.args.sample_num}.pt'))

        # 可视化数据并保存图片（最多可视化前五个样本）
        for i, solution in enumerate(data_list[:10]):  # 仅处理前五个样本
            self.plot_single_sample(solution, variable_names=[f'Node {j}' for j in range(self.node_num)])
            plt.savefig(os.path.join(visualization_path, f'visualization_{i}.png'))
            plt.close()  # 关闭当前图以释放内存
        
        # 生成数据情况
        print('Data Shape:', solution.x.shape)


def get_args():
    parser = argparse.ArgumentParser(description="Dataset Generator for Hypergraph Dynamics Model")

    # 动态模型参数
    parser.add_argument('--sample_interval', type=float, default=0.01, help='Total time for ODE integration')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of time points in the time series')
    parser.add_argument('--init_dim', type=int, default=3, help='Dimension of initial state for each node')

    # 数据生成参数
    parser.add_argument('--sample_num', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--save_path', type=str, default='/data/scratch/acw802/SIGN_hd_cou/SIGN_data', help='Path to save the generated data')
    parser.add_argument('--stre', type=float, default=0.1, help='coupling strength')

    # ODE求解器参数
    parser.add_argument('--method', type=str, choices=["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams", "fixed_adams", "scipy_solver"],
                        default='rk4') 

    # 图结构参数
    parser.add_argument('--node_num', type=int, default=1000, help='Number of nodes in the graph')
    parser.add_argument('--network', type=str, choices=['random', 'power_law', 'small_world', 'from_file'], default='small_world', help='Type of graph topology to use')
    #network files: github/musae_git_edges.csv, bn-fly/bn-fly-drosophila_medulla_1.edges, bn-human-BNU_1_0025890_session_178k/bn-human-BNU_1_0025890_session_1.edges
    # catster/soc-catster.edges, fb-indiana/socfb-Indiana.mtx, human-gene/bio-human-gene2.edges, mammalia-voles-bhp-trapping-2k/mammalia-voles-bhp-trapping.edges
    # bio-mouse-gene/bio-mouse-gene.edges
    parser.add_argument('--file_path', type=str, default='D:/QMcode/HiSIGN/SIGN_original/networks/bn-human-BNU_1_0025890_session_178k/bn-human-BNU_1_0025890_session_1.edges', help='file path')

    parser.add_argument('--undirected', type=bool, default=False, help='generate undirected graph')    

    # 时间序列采样模式
    parser.add_argument('--timestamp_mode', type=str, choices=['uniform', 'random'], default='uniform', help='Mode for generating timestamps')

    # 绘图
    parser.add_argument('--plot_node_num', type=int, default=10, help='Number of nodes to plot.')

    # 设备选择
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to run the model on')

    # 动力学
    parser.add_argument('--model_name', type=str, choices= ['FHN', 'HR', 'Rossler'], default='HR', help='Name of the model for saving data')
    # SIS: 0.5, Kuramoto: 10, Gene: 1
    parser.add_argument('--init_scale', type=float, default=[0.5], help='Scale factor for initial state values')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)
    dataset_generator = DatasetGenerator(args)
    dataset_generator.generate_batch_sample()