import os
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
import sys
sys.path.append("..") 
sys.path.append(".") 
import torchdiffeq._impl as ode
from model.utils import *
import torch_geometric
from torch_geometric.nn import MessagePassing


#ode.odeint(model, x0, vt, para, method=self.method) 

# def seed_torch(seed=1029):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True

# seed_torch()


class HeatDiffusion(MessagePassing):
    #dx_i/dt = -k_{i,j} \sum A_{i,j}(xi-xj)
    def __init__(self, edge_index, edge_attr = None, aggr = 'add'):
        super(HeatDiffusion, self).__init__()
        self.aggr = aggr
        self.k = 0.3
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1

    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return -edge_attr * (x_i - x_j) * self.k

    def update(self, aggr_out):
        return aggr_out



class Kuramoto(MessagePassing):
    """
    Kuramoto model:
    dtheta_i/dt = omega_i  +  1/N *sum(k_ij * sin(theta_j - theta_i))

    """
    def __init__(self, edge_index, edge_attr = None, aggr = 'mean'):
        super(Kuramoto, self).__init__()
        self.aggr = aggr
        self.k = 0.2
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.omega = 0.5
    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr * (torch.sin(x_j - x_i)) * self.k

    def update(self, aggr_out):
        return aggr_out + self.omega





class GeneDynamics(MessagePassing):
    """
    :param t:  time tick
    :param x:  initial value:  is 2d row vector feature, n * dim
    :return: dxi(t)/dt = -b*xi^f + \sum_{j=1}^{N}Aij xj^h / (1 + xj^h)
    If t is not used, then it is autonomous system, only the time difference matters in numerical computing
    """
    def __init__(self, edge_index, edge_attr = None, aggr = 'add'):
        super(GeneDynamics, self).__init__()
        self.aggr = aggr
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.h = 2
        self.f = 1
        self.b = 0.2
        self.e = 0.1

    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_j, edge_attr):
        return self.e * edge_attr * (torch.pow(x_j, self.h)/(1 + torch.pow(x_j, self.h)))

    def update(self, aggr_out, x):
        return aggr_out - torch.pow(x, self.f) * self.b



class MutualDynamics(MessagePassing):
    """
    :param t:  time tick
    :param x:  initial value:  is 2d row vector feature, n * dim
    :return: dxi(t)/dt = bi + xi(1-xi/ki)(xi/ci-1) + \sum_{j=1}^{N}Aij *xi *xj/(di +ei*xi + hi*xj)
    If t is not used, then it is autonomous system, only the time difference matters in numerical computing
    """
    def __init__(self, edge_index, edge_attr = None, aggr = 'sum'):
        super(MutualDynamics, self).__init__()
        self.aggr = aggr
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = torch.ones(edge_index.shape[1], 1)
        self.c = 1
        self.d = 5
        self.b = 0.1
        self.k = 5
        self.e = 0.9
        self.h = 0.1


    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr * (x_i * x_j/(self.d + self.e*x_i + self.h * x_j))

    def update(self, aggr_out, x):
        return aggr_out + x * (1 - x/self.k) * (x/self.c - 1)





if "__main__" == __name__:
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", type=str, default=None, help="Cuda")
    
    # args = parser.parse_args()
    A = torch.tensor([[0,1,2,3,2,1],[3,2,1,0,3,0]])
    model = HeatDiffusion(A)
    t = torch.linspace(0, 10, 100)
    x = torch.rand(4, 1)
    y = model(t, x)
    print(y)