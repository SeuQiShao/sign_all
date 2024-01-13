from cProfile import label
import os
import argparse
from re import L
import time
from turtle import color
#from matplotlib.lines import _LineStyle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import sys
import datetime
sys.path.append("..")
sys.path.append(".")
import torchdiffeq._impl as ode

from dynamics import *
import networkx as nx
from model.utils import *
import tqdm
import torch_geometric as pyg
from torch.utils import data
from torch_geometric.data import Data, TemporalData
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, add_random_edge, dropout_edge, remove_self_loops
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch




class generate_dataset():
    def __init__(self, args):
        #self.seed = args.seed
        self.args = args
        self.cos = args.cos
        self.method = args.method
        self.sample_num = args.sample_num
        self.sampled_time = args.sampled_time
        self.rate = args.rate
        self.node_num = args.n
        self.network = args.network
        self.T = args.T
        self.time_tick = args.time_tick
        self.init_range = args.init_range 
        self.do_interval = args.do_interval
        self.do_num = args.do_num
        self.do_operator_num = args.do_operator_num
        self.ode_model = args.ode_model
        self.tick_rate = args.sample_tick_rate
        self.layout = args.layout
        #self.network = args.network
        self.save_path = os.path.join(args.save_path, self.ode_model + '_' + str(self.node_num) + '_' + self.network +'/raw')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.suffix = '_' + self.ode_model + '_' + self.network + '_' + self.layout + '_' + self.sampled_time + '_' + str(self.time_tick) + '_' + str(self.do_num) + '_' + str(self.node_num)
        ######plot######
        self.plot = args.plot
        self.plot_path = args.plot_path
        self.plot_node_num = args.plot_node_num
        self.plot_line_style = args.plot_line_style
        self.plot_line_color = args.plot_line_color
        if self.plot:
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)


    def generate_ralations(self):
        n = self.node_num
        seed = np.random.randint(0,9999)
        if self.network == 'random':
            #print("Choose graph: " + self.network)
            G = nx.erdos_renyi_graph(n, 0.6, seed=seed)
            self.G = to_undirected(from_networkx(G).edge_index)
        elif self.network == 'power_law':
            #print("Choose graph: " + self.network)
            G = nx.barabasi_albert_graph(n, 3, seed=seed)
            self.G =  to_undirected(from_networkx(G).edge_index)
        elif self.network == 'small_world':
            #print("Choose graph: " + self.network)
            G = nx.newman_watts_strogatz_graph(n, 3, 0.3, seed=seed)
            self.G =  to_undirected(from_networkx(G).edge_index)



    def generate_timestamp(self):
        if self.sampled_time == 'equal':
            #print('Build Equally-sampled -time dynamics')
            t = torch.linspace(0., self.T, self.time_tick)  # args.time_tick) # 100 vector
            # train_deli = 80
            id_train = list(range(int(self.time_tick * self.tick_rate))) # first 80 % for train
            t = t[id_train]
            self.t = t
        elif self.sampled_time == 'irregular':
            #print('Build irregularly-sampled -time dynamics')
            # irregular time sequence
            sparse_scale = 10
            t = torch.linspace(0., self.T, self.time_tick * sparse_scale) # 100 * 10 = 1000 equally-sampled tick
            t = np.random.permutation(t)[:int(self.time_tick * self.tick_rate)]
            t = torch.tensor(np.sort(t))
            t[0] = 0
            self.t = t


        if self.do_num > 0:
            while  True:
                do_index = np.random.randint(self.do_interval, int(self.time_tick * self.tick_rate) - self.do_interval, size = self.do_num)
                self.do_index = np.sort(do_index).reshape(-1)
                
                if self.do_num == 1:
                    break
                if np.diff(self.do_index).min() > self.do_interval:
                    break
        else:
            self.do_index = [None]
            self.event = torch.zeros(1, int(self.time_tick * self.tick_rate))

        if self.do_num > 0:
            event = []
            for i in range(self.do_num + 1):
                if i == 0: 
                    event.append(torch.zeros(1, self.do_index[i])) #[0,0,0,0,0,0,]
                elif i == self.do_num:
                    event.append(i *torch.ones(1, int(self.time_tick * self.tick_rate) - self.do_index[-1]))
                else:
                    event.append(i * torch.ones(1, self.do_index[i] - self.do_index[i - 1]))
            self.event = torch.cat(event,1).squeeze()

                        
        
                


    def generate_single_sample(self):
        self.generate_ralations()
        self.generate_timestamp()
        edge_list = []
        self.init_value = self.init_range * torch.rand(self.node_num,1) + 1
        t = self.t
        x0 = self.init_value
        # A = self.A
        # Adjs = A.repeat((t.shape[0],1,1))
        if self.ode_model == 'HeatDiffusion':     
            ode_model = HeatDiffusion
        elif self.ode_model == 'Kuramoto':
            ode_model = Kuramoto
        elif self.ode_model == 'Gene':
            ode_model = GeneDynamics
        elif self.ode_model == 'Mutual':
            ode_model = MutualDynamics


        for i in range(self.do_num + 1):
            if i == 0:
                self.G = remove_self_loops(self.G)[0]
                model = ode_model(self.G)
                vt = t[:self.do_index[i]]
                edge_list = edge_list + [self.G] * len(vt)
                with torch.no_grad():
                    solution_numerical = ode.odeint(model, x0, vt, method=self.method) 
                    #print(solution_numerical.shape)                 
            elif i < self.do_num and i > 0:
                self.G = add_random_edge(self.G, p = 0.2, force_undirected=True)[0]
                self.G = dropout_edge(self.G, p = 0.2, force_undirected=True)[0]
                self.G = remove_self_loops(self.G)[0]
                model = ode_model(self.G)
                vt = t[self.do_index[i - 1] - 1 :self.do_index[i]] - t[self.do_index[i - 1] - 1]
                edge_list = edge_list + [self.G] * (len(vt) - 1)
                with torch.no_grad():
                    solution_numerical_ = ode.odeint(model, solution_numerical[-1], vt, method=self.method)
                    solution_numerical = torch.cat((solution_numerical, solution_numerical_[1:]), dim=0)
            elif i == self.do_num:
                self.G = add_random_edge(self.G, p = 0.2, force_undirected=True)[0]
                self.G = dropout_edge(self.G, p = 0.2, force_undirected=True)[0]
                self.G = remove_self_loops(self.G)[0]
                model = ode_model(self.G)
                vt = t[self.do_index[i - 1] - 1:] - t[self.do_index[i - 1] - 1]
                edge_list = edge_list + [self.G] * (len(vt) - 1)
                with torch.no_grad():
                    solution_numerical_ = ode.odeint(model, solution_numerical[-1], vt, method=self.method)
                    solution_numerical = torch.cat((solution_numerical, solution_numerical_[1:]), dim=0)         

        

        if self.do_num == 0:
            Tdata = Data(x = solution_numerical.permute(1,0,2), edge_index=edge_list[0], t = self.t, event = self.event)
        else:
            Tdata = TemporalData(x = solution_numerical, edge_index = edge_list, t = self.t, event = self.event)

        return Tdata

    def plot_single_sample(self):
        Tdata = self.generate_single_sample()
        t = self.t
        if self.cos:
            solution_numerical = torch.cos(Tdata.x.permute(1,0,2))
        else:
            solution_numerical = Tdata.x.permute(1,0,2)
        print(solution_numerical.shape)    
        Feature_len = solution_numerical.shape[2]
        node_num = solution_numerical.shape[1]
        if node_num > self.plot_node_num:
            plot_node_num = self.plot_node_num
            node_index = random.sample(range(node_num), self.plot_node_num)
        else:
            plot_node_num = node_num
            node_index = range(node_num)
        do = self.do_index
        print('do_index: ', do)
        if do[0] is not None:
            print('do_t: ', t[do])
        plt.figure(figsize=(10,10))
        for i in range(Feature_len):
            col = 2
            ax = plt.subplot(Feature_len//col + 1, col, i + 1)
            for j in range(plot_node_num):
                for k in range(len(do) + 1):
                    if do[0]:
                        if k == 0:
                            ax.plot(t[:do[k]], solution_numerical[:do[k],node_index[j],i],
                                    color = self.plot_line_color[j%len(self.plot_line_color)],
                                    linestyle = self.plot_line_style[k%len(self.plot_line_style)],
                                    label = 'node_{}'.format(node_index[j]))
                        elif k < len(do) and k > 0:
                            ax.plot(t[do[k-1] -1:do[k]], solution_numerical[do[k-1] -1:do[k],node_index[j],i],
                                    color = self.plot_line_color[j%len(self.plot_line_color)],
                                    linestyle = self.plot_line_style[k%len(self.plot_line_style)])
                        elif k == len(do):
                            ax.plot(t[do[k-1] -1:], solution_numerical[do[k-1] -1:,node_index[j],i],
                                    color = self.plot_line_color[j%len(self.plot_line_color)],
                                    linestyle = self.plot_line_style[k%len(self.plot_line_style)])
                    else:
                        ax.plot(t, solution_numerical[:,node_index[j],i],
                                    color = self.plot_line_color[j%len(self.plot_line_color)],
                                    linestyle = self.plot_line_style[k%len(self.plot_line_style)],
                                    label = 'node_{}'.format(node_index[j]))
            ax.set_xlabel('time')
            ax.set_ylabel('Feature_{}'.format(i))
            ax.legend()
        plt.show()
        plt.savefig('{}/{}_sample.png'.format(self.plot_path, self.suffix))


    def generate_batch_sample(self):
        data_list = []
        for i in tqdm.tqdm(range(self.sample_num)):
            data_list.append(self.generate_single_sample())

        torch.save(data_list, self.save_path + '/data_{}.pt'.format(self.sample_num))

        



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate ODE Dataset!')
    #######ODES#######
    parser.add_argument('--method', type=str,
                        choices=['dopri5', 'adams', 'explicit_adams', 'fixed_adams','tsit5', 'euler', 'midpoint', 'rk4'],
                        default='dopri5')  # dopri5
    parser.add_argument('--rtol', type=float, default=0.01,
                        help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
    parser.add_argument('--atol', type=float, default=0.001,
                        help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')
    parser.add_argument('--time_tick', type=int, default=100) # default=10)
    parser.add_argument('--do_interval', type=int, default=5) # default=100)
    parser.add_argument('--sampled_time', type=str,
                        choices=['irregular', 'equal'], default='equal')  # 均匀采样
    parser.add_argument('--T', type=float, default=10, help='Terminal Time')  
    parser.add_argument('--do_num', type=int, default=0, help='do_operator_number < time_tick')     
    parser.add_argument('--do_operator_num', type=int, default=3, help='Max_do_operator_num')     
    parser.add_argument('--init_range', type=float, default=9, help='init_range')
    parser.add_argument('--noise', type=bool, default=None, help='noise')          
    #######Models#######
    parser.add_argument('--ode_model', type=str,
                        choices=['HeatDiffusion', 'Kuramoto', 'Mutual', 'Gene'], default='Kuramoto')
    #parser.add_argument('--freq', type=bool, default=False, help='use frequency or not')
    parser.add_argument('--cos', type=bool, default=False, help='use frequency or not')                       
    parser.add_argument('--n', type=int, default=10000, help='Number of nodes')
    parser.add_argument('--network', type=str,
                        choices=['random', 'power_law', 'small_world'], default='power_law')
    parser.add_argument('--layout', type=str, choices=['community', 'degree'], default='community')
    parser.add_argument('--sample_tick_rate', type=float, default=0.8, help='sample_rate')

    #######Others#######
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--rate', type=list, default=[0.7,0.1,0.2], help='Devision Rate')
    parser.add_argument('--sample_num', type=int, default=1, help='Sample_num')
    parser.add_argument('--save_path', type=str, default='/home/shaoqi/code/SIGN_data', help='Save_path')
    parser.add_argument('--cuda', type=str, default=None, help='cuda')
    #######Plots#######
    parser.add_argument('--plot', type=bool, default=True, help='Plot')
    parser.add_argument('--plot_path', type=str, default='/home/shaoqi/code/SIGN_data/plot', help='Plot_path')
    parser.add_argument('--plot_node_num', type=int, default=5, help='Plot_node')
    parser.add_argument('--plot_line_style', type=tuple,default=('-',':'), help='Plot_node_type')
    parser.add_argument('--plot_line_color', type=str, default='rgbcmyk', help='Plot_node_color')


    args = parser.parse_args()


    my_data = generate_dataset(args)
    print('Start Generate......')
    #my_data.plot_single_sample()
    my_data.generate_batch_sample()


    