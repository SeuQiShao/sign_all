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
#from model.utils import *
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
    def __init__(self, edge_index, noise = 0, stre = 0.15, edge_attr = None, aggr = 'add'):
        super(Kuramoto, self).__init__()
        self.aggr = aggr
        self.k = stre
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.node_num = edge_index.max() + 1
        if noise == 0:
            self.omega = 0.3 * torch.ones(self.node_num, 1)
        else:
            self.omega = torch.normal(0.3 * torch.ones(self.node_num, 1), noise)
    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr * (torch.sin(x_j - x_i)) * self.k

    def update(self, aggr_out):
        device = aggr_out.device
        return aggr_out + self.omega.to(device)





class GeneDynamics(MessagePassing):
    """
    :param t:  time tick
    :param x:  initial value:  is 2d row vector feature, n * dim
    :return: dxi(t)/dt = -b*xi^f + \sum_{j=1}^{N}Aij xj^h / (1 + xj^h)
    If t is not used, then it is autonomous system, only the time difference matters in numerical computing
    """
    def __init__(self, edge_index, edge_attr = None, noise = 0, stre = 0.15, aggr = 'add'):
        super(GeneDynamics, self).__init__()
        self.aggr = aggr
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.h = 1
        self.f = 2
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
    def __init__(self, edge_index, edge_attr = None, noise = 0, stre = 0.15, aggr = 'add'):
        super(MutualDynamics, self).__init__()
        self.aggr = aggr
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.c = 1
        self.d = 20
        self.b = 0.1
        self.k = 0.5
        self.e = 0.9
        self.h = 0.1


    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr * (x_i * x_j/(self.d + self.e*x_i + self.h * x_j))

    def update(self, aggr_out, x):
        return aggr_out + x * (1 - x/self.k) * (x/self.c - 1) + self.b



class RosslerDynamics(MessagePassing):
    """
    :param t:  time tick
    :param x:  initial value:  is 2d row vector feature, n * dim
    :return:   dxi1/dt = -wixi2 - xi3 + e \sum Aij (xj1- xi1)
               dxi2/dt = wixi1 + axi2
               dxi3/dt = b + xi3(xi1 + c)
     
    If t is not used, then it is autonomous system, only the time difference matters in numerical computing
    """
    def __init__(self, edge_index, edge_attr = None, aggr = 'sum'):
        super(RosslerDynamics, self).__init__()
        self.aggr = aggr
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.e = 0.01
        self.a = 0.2
        self.b = 0.2
        self.c = -6
        self.w = 1



    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        device = x_i.device
        x_d1 = edge_attr * (x_j[:,[0]] - x_i[:,[0]]) * self.e
        x_d2 = torch.zeros(x_d1.shape).to(device)
        x_d3 = torch.zeros(x_d1.shape).to(device)
        return torch.cat([x_d1, x_d2, x_d3], 1)

    def update(self, aggr_out, x):
        x_d1 = -self.w * x[:,[1]] - x[:,[2]]
        x_d2 = self.w * x[:,[0]] + self.a * x[:,[1]]
        x_d3 = self.b + x[:,[2]] * (x[:,[0]] + self.c)
        return aggr_out + torch.cat([x_d1, x_d2, x_d3], 1)


class FitzHughDynamics(MessagePassing):
    """
    :param t:  time tick
    :param x:  initial value:  is 2d row vector feature, n * dim
    :return:   dxi1/dt = xi1 - xi1^3 -xi2 - e * sum Aij * (Xj1-xi1)/kin
               dxi2/dt = a + bxi1 + c xi2
     
    If t is not used, then it is autonomous system, only the time difference matters in numerical computing
    """
    def __init__(self, edge_index, noise = 0, stre = 0.01, edge_attr = None, aggr = 'add'):
        super(FitzHughDynamics, self).__init__()
        self.aggr = aggr
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.e = stre
        self.a = 0.5
        self.b = 0.3
        self.c = 0.1
        self.v = 1

    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        device = x_i.device
        x_d1 = -edge_attr * (x_j[:,[0]] - x_i[:,[0]]) * self.e
        x_d2 = torch.zeros(x_d1.shape).to(device)
        return torch.cat([x_d1, x_d2], 1)

    def update(self, aggr_out, x):
        x_d1 = x[:,[0]] - self.v * x[:,[0]]**3 - x[:,[1]] + 1
        x_d2 = self.c * (self.a  + x[:,[0]] - self.b * x[:,[1]]) #c(V + a - bw) -> cx[0] + ca - bcx[1]// 0.05, 0.1, -0.03
        return aggr_out + torch.cat([x_d1, x_d2], 1)


class HRDynamics(MessagePassing):
    """
    :param t:  time tick
    :param x:  initial value:  is 3d row vector feature, n * dim
    :return:   
     
    If t is not used, then it is autonomous system, only the time difference matters in numerical computing
    """
    def __init__(self, edge_index, edge_attr = None, aggr = 'add'):
        super(HRDynamics, self).__init__()
        self.aggr = aggr
        self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        else:
            self.edge_attr = 1
        self.e = 0.5
        self.a = 3
        self.b = 5
        self.c = 1
        self.mu = 0.01
        self.s = 4
        self.I = 3.24

    def forward(self, t, x):
        edge_index = self.edge_index


        return self.propagate(edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        device = x_i.device
        x_d1 = edge_attr * (x_j[:,[0]] - x_i[:,[0]]) * self.e 
        #x_d1 = edge_attr * 0.15 *(2 - x_i[:,[0]]) * torch.sigmoid(10 * (x_j[:, [0]] - 1))
        x_d2 = torch.zeros(x_d1.shape).to(device)
        x_d3 = torch.zeros(x_d1.shape).to(device)
        return torch.cat([x_d1, x_d2, x_d3], 1)

    def update(self, aggr_out, x):
        x_d1 = x[:,[1]] - x[:,[0]]**3 + self.a*x[:,[0]]**2 -x[:,[2]] + self.I
        x_d2 = self.c - self.b * x[:,[0]]**2 - x[:,[1]]
        x_d3 = self.mu * (self.s*(x[:,[0]] + 1.6) - x[:,[2]]) # 0.01 x0 - 0.02 -0.01x2
        return aggr_out + torch.cat([x_d1, x_d2, x_d3], 1)





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