import torch
import os
import networkx as nx
import numpy as np



network = '/home/shaoqi/code/SIGN_all/SIGN_data/HeatDiffusion_1000_small_world'
file_path = '{}/raw'.format(network)
file_names = os.listdir(file_path)
file = os.path.join(file_path,file_names[0])
data = torch.load(file)



def get_network_degree():
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(data[0].x)
    edge_index = data[0].edge_index.numpy()
    nx_graph.add_edges_from(edge_index.T)

    # 计算度分布
    degrees = dict(nx_graph.degree())

    # 计算平均度
    average_degree = sum(degrees.values()) / nx_graph.number_of_nodes()

    # 计算最大和最小度
    max_degree = max(degrees.values())
    min_degree = min(degrees.values())

    # 计算度异质性
    degree_values = list(degrees.values())
    degree_heterogeneity = sum(degree ** 2 for degree in degree_values) * nx_graph.number_of_nodes() / (sum(degree for degree in degree_values))**2 

    # 打印结果
    print(f"Average Degree: {average_degree}")
    print(f"Maximum Degree: {max_degree}")
    print(f"Minimum Degree: {min_degree}")
    print(f"Degree Heterogeneity: {degree_heterogeneity}")


def get_x_stat():
    print('graph num:', len(data))
    print('data.x',data[0].x.shape)
    print('data.t',data[0].t[:10])
    print('edges:',data[0].edge_index.shape)
    print('x_range:', data[0].x[:,0,0].max(), data[0].x[:,0,0].min())



if __name__ == '__main__':
    get_x_stat()

