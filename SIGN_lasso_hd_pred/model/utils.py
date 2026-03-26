import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import networkx as nx
import scipy.sparse as sp
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.linear_model import OrthogonalMatchingPursuit, LassoCV
from itertools import combinations
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader

def my_softmax(input, axis=1):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def encode_onehot(labels):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(
    preds, num_atoms, num_edge_types, add_const=False, eps=1e-16
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (preds.size(0) * preds.size(1)) 

def edge_accuracy(preds, target, binary=True):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    _, preds = preds.max(-1)
    if binary:
        preds = (preds >= 1).long()
    correct = preds.float().data.eq(target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / target.numel()

def calc_auroc(pred_edges, GT_edges):
    pred_edges = 1 - pred_edges[:, :, :, 0]
    return roc_auc_score(
        GT_edges.cpu().detach().flatten(),
        pred_edges.cpu().detach().flatten(),  # [:, :, 1]
    )


def kl_latent(args, prob, log_prior, predicted_atoms):
    if args.prior != 1:
        return kl_categorical(prob, log_prior, predicted_atoms)
    else:
        return kl_categorical_uniform(prob, predicted_atoms, args.edge_types)



def kl_normal_reverse(prior_mean, prior_std, mean, log_std, downscale_factor=1):
    std = softplus(log_std) * downscale_factor
    d = tdist.Normal(mean, std)
    prior_normal = tdist.Normal(prior_mean, prior_std)
    return tdist.kl.kl_divergence(d, prior_normal).mean()


def sample_normal_from_latents(latent_means, latent_logsigmas, downscale_factor=1):
    latent_sigmas = softplus(latent_logsigmas) * downscale_factor
    eps = torch.randn_like(latent_sigmas)
    latents = latent_means + eps * latent_sigmas
    return latents


def softplus(x):
    return torch.log(1.0 + torch.exp(x))


def distribute_over_GPUs(args, model, num_GPU=None):
    ## distribute over GPUs
    if args.device.type != "cpu":
        if num_GPU is None:
            model = torch.nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            args.batch_size_multiGPU = args.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = torch.nn.DataParallel(model, device_ids=list(range(num_GPU)))
            args.batch_size_multiGPU = args.batch_size * num_GPU
    else:
        model = torch.nn.DataParallel(model)
        args.batch_size_multiGPU = args.batch_size

    model = model.to(args.device)

    return model, num_GPU


def create_rel_rec_send(args, num_atoms):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    if args.cuda:
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    return rel_rec, rel_send


def append_losses(losses_list, losses):
    for loss, value in losses.items():
        if type(value) == float:
            losses_list[loss].append(value)
        elif type(value) == defaultdict:
            if losses_list[loss] == []:
                losses_list[loss] = defaultdict(list)
            for idx, elem in value.items():
                losses_list[loss][idx].append(elem)
        else:
            if value.shape:
                losses_list[loss].append(value.tolist())
            else:
                losses_list[loss].append(value.item())
    return losses_list


def average_listdict(listdict, num_atoms):
    average_list = [None] * num_atoms
    for k, v in listdict.items():
        average_list[k] = sum(v) / len(v)
    return average_list


# Latent Temperature Experiment utils
def get_uniform_parameters_from_latents(latent_params):
    n_params = latent_params.shape[1]
    logit_means = latent_params[:, : n_params // 2]
    logit_widths = latent_params[:, n_params // 2 :]
    means = sigmoid(logit_means)
    widths = sigmoid(logit_widths)
    mins, _ = torch.min(torch.cat([means, 1 - means], dim=1), dim=1, keepdim=True)
    widths = mins * widths
    return means, widths


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def sample_uniform_from_latents(latent_means, latent_width):
    latent_dist = tdist.uniform.Uniform(
        latent_means - latent_width, latent_means + latent_width
    )
    latents = latent_dist.rsample()
    return latents


def get_categorical_temperature_prior(mid, num_cats, to_torch=True, to_cuda=True):
    categories = [mid * (2.0 ** c) for c in np.arange(num_cats) - (num_cats // 2)]
    if to_torch:
        categories = torch.Tensor(categories)
    if to_cuda:
        categories = categories.cuda()
    return categories


def kl_uniform(latent_width, prior_width):
    eps = 1e-8
    kl = torch.log(prior_width / (latent_width + eps))
    return kl.mean()


def get_uniform_logprobs(inferred_mu, inferred_width, temperatures):
    latent_dist = tdist.uniform.Uniform(
        inferred_mu - inferred_width, inferred_mu + inferred_width
    )
    cdf = latent_dist.cdf(temperatures)
    log_prob_default = latent_dist.log_prob(inferred_mu)
    probs = torch.where(
        cdf * (1 - cdf) > 0.0, log_prob_default, torch.full(cdf.shape, -8).cuda()
    )
    return probs.mean()


def get_preds_from_uniform(inferred_mu, inferred_width, categorical_temperature_prior):
    categorical_temperature_prior = torch.reshape(
        categorical_temperature_prior, [1, -1]
    )
    preds = (
        (categorical_temperature_prior > inferred_mu - inferred_width)
        * (categorical_temperature_prior < inferred_mu + inferred_width)
    ).double()
    return preds


def get_correlation(a, b):
    numerator = torch.sum((a - a.mean()) * (b - b.mean()))
    denominator = torch.sqrt(torch.sum((a - a.mean()) ** 2)) * torch.sqrt(
        torch.sum((b - b.mean()) ** 2)
    )
    return numerator / denominator


def get_offdiag_indices(num_nodes):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices

def batch_fft(args, x):
    #inputs.shape = B * N * T * F
    device = args.device
    out = torch.randn(x.shape[0],x.shape[1],x.shape[2],x.shape[3]*2).to(device)
    for i in range(x.shape[0]):
        N = []
        for j in range(x.shape[1]):
            temp = torch.fft.fft(x[i,j], dim = 0)
            temp = torch.stack((temp.real,temp.imag),2)
            temp = temp.reshape(x.shape[2], x.shape[3]*2)
            out[i,j,:,:] = temp 
    return out


def batch_rotation(args, input, theta):
    # input: B * N * T * F
    device = args.device
    # x = input[:,:,:,0]
    # y = input[:,:,:,1]
    # v_x = input[:,:,:,2]
    # v_y = input[:,:,:,2]
    out = torch.randn(input.shape).to(device)
    x = input[:,:,:,[0,1]]
    v = input[:,:,:,[2,3]]
    r_m = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    R_m = torch.tensor(r_m, dtype = torch.float).to(device)
    new_x = torch.einsum('ijkl,lp->ijkp',x, R_m)
    new_v = torch.einsum('ijkl,lp->ijkp',v, R_m)
    out[:,:,:,[0,1]] = new_x
    out[:,:,:,[2,3]] = new_v
    return out
    
def invariant_kl(P,Q, ep = 1e-16):
    #input B * G * 2
    #KL: SUM(P*log(P/Q))
    kl = torch.sum(P * torch.log((P + ep)/(Q + ep)))/(P.shape[0] * P.shape[1])
    return kl


def grid_8_neighbor_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    N = int(N)
    n = int(N ** 2)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = torch.zeros(n, n)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if N > newx >= 0 and N > newy >= 0:
                    index2 = newx * N + newy
                    A[index, index2] = 1
    return A.float()

def generate_node_mapping(G, type=None):
    """
    :param G:
    :param type:
    :return:
    """
    if type == 'degree':
        s = sorted(G.degree, key=lambda x: x[1], reverse=True)
        new_map = {s[i][0]: i for i in range(len(s))}
    elif type == 'community':
        cs = list(greedy_modularity_communities(G))
        l = []
        for c in cs:
            l += list(c)
        new_map = {l[i]:i for i in range(len(l))}
    else:
        new_map = None

    return new_map

def networkx_reorder_nodes(G, type=None):
    """
    :param G:  networkX only adjacency matrix without attrs
    :param nodes_map:  nodes mapping dictionary
    :return:
    """
    nodes_map = generate_node_mapping(G, type)
    if nodes_map is None:
        return G
    C = nx.to_scipy_sparse_matrix(G, format='coo')
    new_row = np.array([nodes_map[x] for x in C.row], dtype=np.int32)
    new_col = np.array([nodes_map[x] for x in C.col], dtype=np.int32)
    new_C = sp.coo_matrix((C.data, (new_row, new_col)), shape=C.shape)
    new_G = nx.from_scipy_sparse_matrix(new_C)
    return new_G

def edge_acc(classes, do):
    classes = classes.view(-1)
    do = do.view(-1)
    acc = 0
    for i in range(len(do)):
        if do[i] == classes[i]:
            acc += 1
    return acc / len(do)

def find_max_index(list):
    index = []
    max_list = np.max(list)
    for i in range(len(list)):
        if list[i] == max_list:
            index.append(i)
    return max_list, index

def find_min_index(list):
    index = []
    min_list = np.min(list)
    for i in range(len(list)):
        if list[i] == min_list:
            index.append(i)
    return min_list, index

def Frobenius(x, y):
    return torch.sqrt(torch.sum((x-y)**2))

def l1_loss(x):

    return x.abs().sum()

def MAPE(output, target):
    d = torch.abs(output - target)
    mape = d/(target.abs() + 1e-5)
    return mape.mean()

def calculate_r2(pred, target):
    """
    计算R²指标
    Args:
        pred: 预测结果 [num_nodes, time_steps, dims]
        target: 真实结果 [num_nodes, time_steps, dims]
    Returns:
        r2_score: R² 分数
    """
    pred = pred.detach().cpu().numpy().reshape(-1, pred.shape[-1])
    target = target.detach().cpu().numpy().reshape(-1, target.shape[-1])
    r2 = r2_score(target, pred, multioutput='uniform_average')
    return r2

def partition_graph_pyg(edge_index, num_nodes, num_parts=2):
    """
    Partition a large graph into `num_parts` subgraphs using PyG's ClusterData.
    
    Args:
        edge_index (torch.Tensor): Edge index of shape (2, num_edges).
        num_nodes (int): Number of nodes in the graph.
        num_parts (int): Number of partitions (default: 2).
        
    Returns:
        sub_edge_indices (list of torch.Tensor): List of edge indices for each subgraph.
        sub_num_nodes (list of int): List of number of nodes in each subgraph.
    """
    # Create a PyG Data object
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    
    # Partition the graph using ClusterData
    cluster_data = ClusterData(data, num_parts=num_parts, recursive=True)
    cluster_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)
    
    sub_edge_indices = []
    sub_num_nodes = []
    
    for batch in cluster_loader:
        sub_edge_indices.append(batch.edge_index)
        sub_num_nodes.append(batch.num_nodes)
    
    return sub_edge_indices, sub_num_nodes

def get_edgeindex_4_edges(file_path):
    """通用边数据加载器"""
    # 读取原始数据（自动检测列数）
    if file_path[-3:] == 'csv':
        raw_df = pd.read_csv(file_path, header=0)
        edge_part = raw_df.values[:, :2].astype(np.int32)   # 前两列强制转换为整型
        attr_part = raw_df.values[:, 2:].astype(np.float32)  # 后续列作为属性
        src_nodes = edge_part[:, 0]
        tgt_nodes = edge_part[:, 1]

    else:
        raw_df = pd.read_csv(
            file_path,
            sep=r'\s+',          # 匹配任意空白分隔符
            comment='%',         # 跳过注释行
            header=None,
            engine='c',
            dtype=np.float32     # 统一先读为float32节省内存
        )
    
        # 列数验证
        if len(raw_df.columns) < 2:
            raise ValueError("文件至少需要包含2列数据（源节点和目标节点）")

        # 分割数据
        edge_part = raw_df.iloc[:, :2].astype(np.int32)   # 前两列强制转换为整型
        attr_part = raw_df.iloc[:, 2:].astype(np.float32)  # 后续列作为属性
    
        # 验证节点ID合法性
        for col in [0, 1]:
            if np.any(edge_part[col] < 0):
                raise ValueError(f"第{col+1}列包含负数节点ID")

        src_nodes = edge_part[0].to_numpy()
        tgt_nodes = edge_part[1].to_numpy()

    # 删除自环
    non_self_loops = src_nodes != tgt_nodes
    src_nodes = src_nodes[non_self_loops]
    tgt_nodes = tgt_nodes[non_self_loops]

    # 合并所有节点并去重（比np.unique快30%）
    all_nodes = np.concatenate([src_nodes, tgt_nodes])
    unique_nodes, degrees = np.unique(all_nodes, return_counts=True)

    # 删除孤立点
    non_isolated_nodes = unique_nodes[degrees > 0]
    unique_nodes = non_isolated_nodes

    # 创建ID映射字典（使用向量化操作替代循环）
    node_ids = torch.arange(len(unique_nodes), dtype=torch.long)
    id_mapping = torch.full((unique_nodes.max()+1,), -1, dtype=torch.long)
    id_mapping[unique_nodes] = node_ids

    # 映射边数据到新ID（约3秒，使用GPU加速）
    src_tensor = torch.from_numpy(src_nodes)
    tgt_tensor = torch.from_numpy(tgt_nodes)

    # 使用GPU进行映射（如果可用）
    if torch.cuda.is_available():
        id_mapping = id_mapping.cuda()
        src_tensor = src_tensor.cuda()
        tgt_tensor = tgt_tensor.cuda()

    mapped_src = id_mapping[src_tensor]  # 自动并行化
    mapped_tgt = id_mapping[tgt_tensor]

    # 构建edge_index（约0.5秒）
    edge_index = torch.stack([mapped_src.cpu(), mapped_tgt.cpu()], dim=0)

    # # 改为无向图
    # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # 内存优化技巧（如需保留原始ID）
    del edge_part, attr_part, src_nodes, tgt_nodes, all_nodes  # 及时释放内存
    print('\n Graph load finished! Edgeindex shape:', edge_index.shape, ' Node_num:', len(unique_nodes))
    return edge_index, len(unique_nodes)

def fun_lib(x, poly_p, poly_n, device, activate=False, mask=None, names=False):
    """
    三维自耦合基函数库，支持1D/2D/3D输入
    
    参数:
        x (torch.Tensor): 输入张量 (N×d, d∈{1,2,3})
        poly_p (int/list): 正多项式次数，int时为统一次数，list时为各维度次数
        poly_n (int): 负多项式次数
        mask (torch.Tensor): 特征掩码张量
        names (bool): 是否返回特征名称
    """
    def _safe_divide(a, b):
        return a / (b + (b == 0)*1e-6)
    
    # 维度检测与参数处理
    dim = x.shape[1] if x is not None else (3 if isinstance(poly_p, list) else 1)
    dim = min(dim, 3)  # 最大支持3维
    poly_p = [poly_p]*dim if isinstance(poly_p, int) else poly_p[:dim]
    
    # 生成器与名称列表
    generators = []
    name_list = ["1"]  # 常数项
    
    # 常数项生成器
    generators.append(lambda x: torch.ones(x.shape[0], 1, device=device))

    # ================= 多项式项 =================
    for d in range(dim):
        # 单变量多项式
        for p in range(1, poly_p[d]):
            generators.append(
                lambda x, d=d, p=p: torch.pow(x[:, [d]], p)
            )
            name_list.append(f"x{d+1}^{p}")
    
    # 交叉项（两两组合）
    for d1, d2 in combinations(range(dim), 2):
        for p in range(1, min(poly_p[d1], poly_p[d2])):
            generators.append(
                lambda x, d1=d1, d2=d2, p=p: torch.pow(x[:, [d1]] * x[:, [d2]], p)
            )
            name_list.append(f"(x{d1+1}x{d2+1})^{p}")
    
    # 三维交叉项
    if dim == 3:
        for p in range(1, min(poly_p)):
            generators.append(
                lambda x, p=p: torch.pow(x.prod(dim=1, keepdim=True), p)
            )
            name_list.append(f"(x1x2x3)^{p}")

    # ================= 分数项 =================
    # for d in range(dim):
    #     for n in range(1, poly_n+1):
    #         generators.append(
    #             lambda x, d=d, n=n: torch.pow(x[:, [d]], -n) * (torch.abs(x[:, [d]]) > 1e-6)
    #         )
    #         name_list.append(f"x{d+1}^-{n}")
    
    # # 交叉分数项
    # for d1, d2 in combinations(range(dim), 2):
    #     generators.append(
    #         lambda x, d1=d1, d2=d2: torch.pow(_safe_divide(x[:,d1], x[:,d2]), -1)
    #     )
    #     name_list.append(f"(x{d1+1}/x{d2+1})^-1")

    # # ================= 傅里叶项 =================
    # for d in range(dim):
    #     generators.extend([
    #         lambda x, d=d: torch.sin(x[:, [d]]),
    #         lambda x, d=d: torch.cos(x[:, [d]])
    #     ])
    #     name_list.extend([f"sin(x{d+1})", f"cos(x{d+1})"])
    
    # # 交叉傅里叶项
    # for d1, d2 in combinations(range(dim), 2):
    #     generators.extend([
    #         lambda x, d1=d1, d2=d2: torch.sin(x[:, [d1]] * x[:, [d2]]),
    #         lambda x, d1=d1, d2=d2: torch.cos(x[:, [d1]] * x[:, [d2]])
    #     ])
    #     name_list.extend([f"sin(x{d1+1}x{d2+1})", f"cos(x{d1+1}x{d2+1})"])
    
    # if dim == 3:
    #     generators.extend([
    #         lambda x: torch.sin(x.prod(dim=1, keepdim=True)),
    #         lambda x: torch.cos(x.prod(dim=1, keepdim=True))
    #     ])
    #     name_list.extend(["sin(x1x2x3)", "cos(x1x2x3)"])

    # ================= 指数项 =================
    # 单变量指数
    # for d in range(dim):
    #     generators.append(
    #         lambda x, d=d: torch.exp(torch.clamp(x[:, [d]], -10, 10))
    #     )
    #     name_list.append(f"exp(x{d+1})")
    
    # 交叉指数
    # if dim >= 2:
    #     generators.append(
    #         lambda x: torch.exp(torch.clamp(x[:,:2].prod(dim=1, keepdim=True), -5, 5)
    #     ))
    #     name_list.append("exp(x1x2)")
    # if dim == 3:
    #     generators.append(
    #         lambda x: torch.exp(torch.clamp(x.prod(dim=1, keepdim=True), -5, 5)
    #     ))
    #     name_list.append("exp(x1x2x3)")

    # ================= 激活函数项 =================
    if activate:
        # 单变量激活
        for d in range(dim):
            generators.extend([
                lambda x, d=d: torch.sigmoid(x[:, [d]]),
                lambda x, d=d: x[:, [d]] / (torch.abs(x[:, [d]]) + 1)
            ])
            name_list.extend([f"sigmoid(x{d+1})", 
                              f"x{d+1}/(x{d+1}+1)"
                              ])
        # # 交叉项激活
        # if dim >= 2:
        #     generators.append(
        #         lambda x: torch.sigmoid(x[:,:2].sum(dim=1, keepdim=True))
        #     )
        #     name_list.append("sigmoid(x1+x2)")
        # if dim == 3:
        #     generators.append(
        #         lambda x: torch.sigmoid(x.sum(dim=1, keepdim=True))
        #     )
        #     name_list.append("sigmoid(x1+x2+x3)")

    # ================= 模式选择 =================
    if names:
        if mask is not None:
            mask = mask.to(device) if isinstance(mask, torch.Tensor) else torch.tensor(mask, device=device)
            selected = torch.nonzero(mask).flatten().tolist()
            return [name_list[i] for i in selected]
        return name_list

    # ================= 张量生成 =================
    assert x is not None, "需要输入张量x"
    assert x.shape[1] in [1,2,3], "输入应为1D/2D/3D张量"
    
    # 应用掩码
    if mask is not None:
        mask = mask.to(device)
        assert len(mask) == len(generators), f"掩码长度{len(mask)}不匹配特征数{len(generators)}"
        selected = torch.nonzero(mask).flatten()
    else:
        selected = torch.arange(len(generators), device=device)

    # 动态生成特征项
    lib_terms = []
    for idx in selected:
        idx = idx.item()
        try:
            term = generators[idx](x)
            term = torch.nan_to_num(term, nan=0.0, posinf=1e5, neginf=-1e5).reshape(-1, 1)
            lib_terms.append(term)
        except Exception as e:
            print(f"[WARN] 特征'{name_list[idx]}'生成失败: {str(e)}")
    
    # 拼接结果
    lib = torch.cat(lib_terms, dim=1) if lib_terms else torch.empty((x.shape[0], 0), device=device)
    return torch.clamp(lib, -1e5, 1e5)

def coupled_fun_lib(x_i, x_j, poly_p, poly_n, device, activate=False, mask=None, names=False):
    """
    动态生成基函数库，支持显存优化和特征名称输出
    
    参数:
        mask (torch.Tensor): 形状为 [F] 的掩码张量，非零元素对应保留的基函数
        names (bool): True时返回特征名称列表，False时返回特征矩阵
    """
    # 预生成所有基函数的名称和生成器
    generators = []
    name_list = []
    
    # 多项式项（含x_i项）
    for i in range(1, poly_p):
        generators.extend([
            lambda x, y, i=i: torch.pow(y, i),
            lambda x, y, i=i: torch.pow(y - x, i),
            lambda x, y, i=i: torch.pow(y * x, i)
            #lambda x, y, i=i: torch.pow(x, i)
        ])
        name_list.extend([
            f"x_j^{i}",
            f"(x_j - x_i)^{i}",
            f"(x_j * x_i)^{i}"
            #f"x_i^{i}"
        ])

    # 分数项（含安全除法）
    def _safe_divide(a, b):
        return a / (b + (b == 0)*1e-6)
    
    for i in range(1, poly_n):
        generators.extend([
            lambda x, y, i=i: torch.pow(y, -i),
            lambda x, y, i=i: torch.pow(y - x, -i),
            lambda x, y, i=i: torch.pow(y * x, -i),
            lambda x, y, i=i: torch.pow(_safe_divide(y, x), -i)
        ])
        name_list.extend([
            f"x_j^-{i}",
            f"(x_j - x_i)^-{i}",
            f"(x_j * x_i)^-{i}",
            f"(x_j/x_i)^-{i}"
        ])

    # 傅里叶项
    generators.extend([
        lambda x, y: torch.sin(y),
        lambda x, y: torch.cos(y),
        lambda x, y: torch.sin(y * x),
        lambda x, y: torch.cos(y * x),
        lambda x, y: torch.sin(y - x),
        lambda x, y: torch.cos(y - x),
        lambda x, y: torch.sin(y) * x,
        lambda x, y: torch.cos(y) * x
    ])
    name_list.extend([
        "sin(x_j)",
        "cos(x_j)",
        "sin(x_j*x_i)",
        "cos(x_j*x_i)",
        "sin(x_j-x_i)",
        "cos(x_j-x_i)",
        "sin(x_j)*x_i",
        "cos(x_j)*x_i"
    ])

    # 指数项
    generators.extend([
        lambda x, y: torch.exp(y),
        lambda x, y: torch.exp(y * x),
        lambda x, y: torch.exp(y - x),
        lambda x, y: x * torch.exp(y)
    ])
    name_list.extend([
        "exp(x_j)",
        "exp(x_j*x_i)",
        "exp(x_j-x_i)",
        "x_i*exp(x_j)"
    ])

    generators.extend([
            lambda x, y: x * torch.sigmoid(10*(y - 1)),
            lambda x, y: torch.sigmoid(10*(y - 1)),
    ])
    name_list.extend([
            "x_i*sigmoid(10(x_j-1))",
            "sigmoid(10(x_j-1))"
    ])

    # 激活函数项
    if activate:
        generators.extend([
            lambda x, y: torch.sigmoid(y),
            lambda x, y: torch.sigmoid(y * x),
            lambda x, y: torch.sigmoid(y - x),
            lambda x, y: x * torch.sigmoid(y),
            lambda x, y: torch.tanh(y),
            lambda x, y: torch.tanh(y * x),
            lambda x, y: torch.tanh(y - x),
            lambda x, y: x * torch.tanh(y),
            lambda x, y: y / (y + 1),
            lambda x, y: x * y / (x * y + 1),
            lambda x, y: (y - x) / (y - x + 1),
            lambda x, y: y * x / (y + x + 1),
        ])
        name_list.extend([
            "sigmoid(x_j)",
            "sigmoid(x_j*x_i)",
            "sigmoid(x_j-x_i)",
            "x_i*sigmoid(x_j)",
            "tanh(x_j)",
            "tanh(x_j*x_i)",
            "tanh(x_j-x_i)",
            "x_i*tanh(x_j)",
            "x_j/(x_j+1)",
            "x_i*x_j/(x_i*x_j+1)",
            "(x_j-x_i)/(x_j-x_i+1)",
            "x_j*x_i/(x_j+x_i+1)",
        ])

    # 名称模式直接返回
    if names:
        if mask is not None:
            mask = mask.to(device) if isinstance(mask, torch.Tensor) else torch.tensor(mask, device=device)
            selected = torch.nonzero(mask).flatten().tolist()
            return [name_list[i] for i in selected]
        return name_list

    # 张量生成模式
    assert x_i is not None and x_j is not None, "需要提供x_i和x_j张量"
    
    # 确定需要生成的索引
    if mask is not None:
        mask = mask.to(device) if isinstance(mask, torch.Tensor) else torch.tensor(mask, device=device)
        assert len(mask) == len(generators), "掩码长度与特征数不匹配"
        selected = torch.nonzero(mask).flatten()
    else:
        selected = torch.arange(len(generators), device=device)

    # 动态生成特征项
    lib_terms = []
    for idx in selected:
        idx = idx.item()
        try:
            term = generators[idx](x_i, x_j)
            term = torch.nan_to_num(term, nan=0.0, posinf=1e5, neginf=-1e5)
            lib_terms.append(term)
        except Exception as e:
            print(f"生成特征'{name_list[idx]}'失败: {str(e)}")
    
    # 拼接结果
    if lib_terms:
        lib = torch.cat(lib_terms, dim=1).to(device)
        return torch.clamp(lib, -1e5, 1e5)
    return torch.empty((x_i.shape[0], 0), device=device)


# 组合表达式生成函数
def functions(poly_p, poly_n, f_mask, c_mask, activate=False):
    # 获取 fun_lib, coupled_fun_lib 和 hyper_fun_lib 的基函数名称列表
    fun_names = fun_lib(torch.empty(0, 3), poly_p, poly_n, device="cpu", activate=activate, names=True)
    coupled_names = coupled_fun_lib(torch.empty(0, 1), torch.empty(0, 1), poly_p, poly_n, device="cpu", activate=activate, names=True)
        # 合并名称列表
    combined_names = fun_names + coupled_names 

    # 初始化结果列表
    expressions = []
    active_coeffs = []
    func_str = ''

    # 遍历 fun_lib 部分
    for f, name in zip(f_mask.squeeze().tolist(), fun_names):
        if np.abs(f) > 1e-4:
            expressions.append(f"{f:.4f} * {name}")
            active_coeffs.append(f)
            func_str = func_str + f"{f:.4f} * {name}" + ' + ' 

    # 遍历 coupled_fun_lib 部分
    for c, name in zip(c_mask.squeeze().tolist(), coupled_names):
        if np.abs(c) > 1e-4:
            expressions.append(f"{c:.4f} * {name}")
            active_coeffs.append(c)
            func_str = func_str + f"{c:.4f} * {name}" + ' + ' 

    # 移除最后的多余的 '+'
    func_str = func_str.rstrip(' + ')

    return func_str, expressions, active_coeffs




if __name__ == '__main__':
    print('Hello World')