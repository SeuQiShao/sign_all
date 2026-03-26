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
from torch_geometric.nn import MessagePassing, GCNConv
warnings.filterwarnings("ignore")



class GCN_Classifier(torch.nn.Module):
    def __init__(self, args, hidden, out):
        super(GCN_Classifier, self).__init__()
        self.args = args
        self.k = args.k
        self.time_stamp = args.time_stamp
        self.node_feature = args.dims
        self.num_nodes = args.num_atoms
        self.dim_in = args.time_stamp * args.dims
        self.tho = args.tho
        # self.mlp1 = nn.Linear(self.node_feature, 64)
        # self.mlp2 = nn.Linear(64, 1)

        self.conv1 = GCNConv(self.time_stamp, hidden)
        #torch.nn.init.uniform_(self.conv1.lin.weight) 
        self.conv2 = GCNConv(hidden, out)
        #torch.nn.init.uniform_(self.conv2.lin.weight) 
    def forward(self, x, edge_index):
        #x = x.reshape(-1,self.dim_in)
        x = x[:,:,self.k]
        x = F.relu(self.conv1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        out = torch.sigmoid(x.mean(0))

        out_t = F.softshrink(out, self.tho)
        out = torch.sign(out_t) * out
        return out.unsqueeze(1)



class MLP_Classifier(torch.nn.Module):
    def __init__(self, args, hidden, out):
        super(MLP_Classifier, self).__init__()
        self.args = args
        self.time_stamp = args.time_stamp
        self.node_feature = args.dims
        self.num_nodes = args.num_atoms
        self.dim_in = args.time_stamp * args.dims
        self.tho = args.tho
        self.mlp = nn.Sequential(nn.Linear(self.node_feature, hidden), nn.ReLU(), nn.Linear(hidden,1))
        self.class_mlp = nn.Sequential(nn.Linear(self.time_stamp, hidden), nn.ReLU(), nn.Linear(hidden,out))
        # self.conv1 = GCNConv(self.time_stamp, hidden)
        # #torch.nn.init.uniform_(self.conv1.lin.weight) 
        # self.conv2 = GCNConv(hidden, out)
        #torch.nn.init.uniform_(self.conv2.lin.weight) 


    def forward(self, x, edge_index):
        #x = x.reshape(-1,self.dim_in)
        x = self.mlp(x).squeeze()
        #x = F.relu(x)
        # x = F.relu(self.conv1(x, edge_index))
        # #x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        x = self.class_mlp(x)
        out = torch.sigmoid(x.mean(0))

        out_t = F.softshrink(out, self.tho)
        out = torch.sign(out_t) * out
        return out.unsqueeze(1)




if __name__ == '__main__':
    pass