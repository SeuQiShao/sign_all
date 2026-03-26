import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
# def extract_data(file_path, dataset_name):
#     with h5py.File(file_path, 'r') as f:
#         # 提取指定的数据集
#         data = f[dataset_name][:]
#         print(f"Dataset {dataset_name} shape: {data.shape}")
#         return data

# # 提取数据
# file_path = r"D:\brain-wide\Source_data_Fig.2b_d.mat"
# roidfdf_data = extract_data(file_path, 'roidfdf')
# roidfdf_clust_data = extract_data(file_path, 'roidfdf_clust')



# Step 1: Load data
def extract_data(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
        print(f"Dataset {dataset_name} shape: {data.shape}")
        return data

file_path = r"D:\brain-wide\Source_data_Fig.2b_d.mat"
roidfdf_data = extract_data(file_path, 'roidfdf')  # shape: (1470, 57813)
data_t = roidfdf_data.T  # shape: (57813, 1470)


selected_neurons = np.arange(3)

# 绘制神经元活动的时间序列图
plt.figure(figsize=(12, 6))
for i in selected_neurons:
    plt.plot(roidfdf_data[:,i].reshape(-1, 10).mean(1), label=f"Neuron {i+1}")
plt.xlabel('Time (samples)')
plt.ylabel('Activity')
plt.title('Neural Activity of Selected Neurons')
plt.legend()
plt.show()


# Step 2: KMeans clustering
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(data_t)
centroids = kmeans.cluster_centers_

print("Cluster labels:", labels.shape)
# Step 3: Intra-cluster edges
edges = []
avg_degree_target = 10
intra_k = avg_degree_target - 2  # 保留2度用于类间连边

for cluster_id in range(n_clusters):
    indices = np.where(labels == cluster_id)[0]
    sub_data = data_t[indices]
    sim_matrix = cosine_similarity(sub_data)
    np.fill_diagonal(sim_matrix, -np.inf)
    for i, row in enumerate(sim_matrix):
        top_k_idx = row.argsort()[-intra_k:]
        src = indices[i]
        for j in top_k_idx:
            tgt = indices[j]
            edges.append((src, tgt))

# Step 4: Inter-cluster edges
inter_k = 3
inter_links_per_class_pair = 10
centroid_sim = cosine_similarity(centroids)
np.fill_diagonal(centroid_sim, -np.inf)

for i in range(n_clusters):
    neighbors = centroid_sim[i].argsort()[-inter_k:]
    for j in neighbors:
        src_indices = np.random.choice(np.where(labels == i)[0], size=inter_links_per_class_pair)
        tgt_indices = np.random.choice(np.where(labels == j)[0], size=inter_links_per_class_pair)
        for s, t in zip(src_indices, tgt_indices):
            edges.append((s, t))

# Step 5: Convert to PyG format
edges = list(set(edges))
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
x = torch.tensor(data_t, dtype=torch.float)  # shape: [57813, 1470]
# aggregate every 10 time points
x = x.view(x.shape[0], -1, 10).mean(dim=2)  # shape: [57813, 147, 10]
t = torch.arange(0, x.shape[1] * 0.01, 0.01)

# Step 6: Calculate stats
g_stat = {
    'max': float(torch.max(x)),
    'min': float(torch.min(x)),
    'mean': float(torch.mean(x)),
    'std': float(torch.std(x)),
}
#x shape: [57813, 1470, 1], edge_index shape: [2, num_edges]
# Step 7: Package and save
g = Data(x=x.unsqueeze(-1), edge_index=edge_index)
g.t = t
g.stat = g_stat
g.cluster = labels
print('Graph stats:', g.stat)
print('edge_index shape:', g.edge_index.shape)
torch.save([g], 'D:/QMcode/sign_all/SIGN_lasso_true/SIGN_data/fish_57813/raw/data1.pt')
print("Saved to zebrafish_graph.pt")

# # 可视化某些神经元的活动
# # 选择前10个神经元的活动数据
# selected_neurons = np.arange(3)

# # 绘制神经元活动的时间序列图
# plt.figure(figsize=(12, 6))
# for i in selected_neurons:
#     plt.plot(roidfdf_data[:100, i], label=f"Neuron {i+1}")
# plt.xlabel('Time (samples)')
# plt.ylabel('Activity')
# plt.title('Neural Activity of Selected Neurons')
# plt.legend()
# plt.show()


# # 绘制神经元活动的时间序列图
# plt.figure(figsize=(12, 6))
# for i in selected_neurons:
#     plt.plot(roidfdf_clust_data[:100, i], label=f"Neuron {i+1}")
# plt.xlabel('Time (samples)')
# plt.ylabel('Activity')
# plt.title('Neural Activity of Selected Neurons')
# plt.legend()
# plt.show()
# # # 也可以绘制热图来查看所有神经元的活动
# # plt.figure(figsize=(12, 6))
# # plt.imshow(roidfdf_data.T, aspect='auto', cmap='viridis')
# # plt.colorbar(label='Activity')
# # plt.title('Neural Activity Heatmap')
# # plt.xlabel('Time (samples)')
# # plt.ylabel('Neuron Index')
# # plt.show()


