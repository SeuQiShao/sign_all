import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import networkx as nx
import scipy.sparse as sp
from networkx.algorithms.community import greedy_modularity_communities

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


# def get_observed_relations_idx(num_atoms):
#     length = (num_atoms ** 2) - num_atoms * 2
#     remove_idx = np.arange(length)[:: num_atoms - 1][1:] - 1
#     idx = np.delete(np.linspace(0, length - 1, length), remove_idx)
#     return idx


# def mse_per_sample(output, target):
#     mse_per_sample = F.mse_loss(output, target, reduction="none")
#     mse_per_sample = torch.mean(mse_per_sample, dim=(1, 2, 3)).cpu().data.numpy()
#     return mse_per_sample


# def edge_accuracy_per_sample(preds, target):
#     _, preds = preds.max(-1)
#     acc = torch.sum(torch.eq(preds, target), dim=1, dtype=torch.float64,) / preds.size(
#         1
#     )
#     return acc.cpu().data.numpy()


# def auroc_per_num_influenced(preds, target, total_num_influenced):
#     preds = 1 - preds[:, :, 0]
#     preds = preds.cpu().detach().numpy()
#     target = target.cpu().detach().numpy()

#     preds_per_num_influenced = defaultdict(list)
#     targets_per_num_influenced = defaultdict(list)

#     for idx, k in enumerate(total_num_influenced):
#         preds_per_num_influenced[k].append(preds[idx])
#         targets_per_num_influenced[k].append(target[idx])

#     auc_per_num_influenced = np.zeros((max(preds_per_num_influenced) + 1))
#     for num_influenced, elem in preds_per_num_influenced.items():
#         auc_per_num_influenced[num_influenced] = roc_auc_score(
#             np.vstack(targets_per_num_influenced[num_influenced]).flatten(),
#             np.vstack(elem).flatten(),
#         )

#     return auc_per_num_influenced


# def edge_accuracy_observed(preds, target, num_atoms=5):
#     idx = get_observed_relations_idx(num_atoms)
#     _, preds = preds.max(-1)
#     correct = preds[:, idx].eq(target[:, idx]).cpu().sum()
#     return np.float(correct) / (target.size(0) * len(idx))


# def calc_auroc_observed(pred_edges, GT_edges, num_atoms=5):
#     idx = get_observed_relations_idx(num_atoms)
#     pred_edges = pred_edges[:, :, 1]
#     return roc_auc_score(
#         GT_edges[:, idx].cpu().detach().flatten(),
#         pred_edges[:, idx].cpu().detach().flatten(),
#     )


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

def batch_compression(x):
    #inputs.shape = B * N * T * F
    index = []
    for i in range(x.shape[2]):
        if i%2 == 0:
            index.append(i)
    return x[:,:,index,:]

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

def fun_fc_lib2(x, ploy_p, device):
    with torch.no_grad(): 
    ####self-coupled for 3 dim
    #x_shape: N * dims
        lib = torch.tensor([]).to(device)
        # 1 order: 2(p -1)
        for i in ploy_p:
            lib = torch.cat((lib, torch.pow(x[:,[0]], i)), 1)
            lib = torch.cat((lib, torch.pow(x[:,[1]], i)), 1)
        lib = torch.cat((lib, x[:,[0]] * x[:,[1]]), 1)
        lib = torch.clamp(lib, min=-1000, max=1000)
        ###########negetive part################
        #nlib = -lib
        # lib = torch.cat((lib, nlib), 1)
    return lib       

def fun_lib(x_i, ploy_p, ploy_n, device, activate = False, tri = True):
    with torch.no_grad(): 
        #Consitant:1
        lib = torch.ones(x_i.shape[0], 1)
        lib = lib.to(device)
        #Polynomial: p-1
        # for i in self.ploy_p:
        #     lib = torch.cat((lib, torch.pow(x_i, i)), 1)
        #Fractional: n-1
        for i in ploy_n:
            lib = torch.cat((lib, torch.pow(x_i, i)), 1)
        # fourier :3
        if tri:
            lib = torch.cat((lib, torch.sin(x_i)), 1)
            lib = torch.cat((lib, torch.cos(x_i)), 1)
            lib = torch.cat((lib, torch.tan(x_i)), 1)
        # Exponential : 1
        lib = torch.cat((lib, torch.exp(x_i)), 1)
        if activate:
        # Activation: g + 1
            lib = torch.cat((lib, torch.sigmoid(x_i)), 1)
            lib = torch.cat((lib, torch.tanh(x_i)), 1)
            lib = torch.cat((lib, torch.pow(x_i, 1)/(torch.pow(x_i, 1) + 1)), 1)
        # ###########clamp###############3#######
        lib = torch.clamp(lib, min=-1000, max=1000)
        ###########negetive part################
        # nlib = -lib
        # lib = torch.cat((lib, nlib), 1)
    return lib 
        
def coupled_fun_lib(x_i, x_j, ploy_p, ploy_n, device, activate = False, tri = True):
    with torch.no_grad(): 
        #Consitant: 1
        #lib = torch.ones(x_i.shape[0], 1)  # (E, 1)
        lib = torch.tensor([])
        lib = lib.to(device)
        #Poly:3 * p - 3
        for i in ploy_p:
            lib = torch.cat((lib, torch.pow(x_j, i)), 1)
            lib = torch.cat((lib, torch.pow(x_j - x_i, i)), 1)
            lib = torch.cat((lib, torch.pow(x_j * x_i, i)), 1)
        #Frac:4 * n - 4
        for i in ploy_n:
            lib = torch.cat((lib, torch.pow(x_j, i)), 1)
            lib = torch.cat((lib, torch.pow(x_j - x_i, i)), 1)
            lib = torch.cat((lib, torch.pow(x_j * x_i, i)), 1)
            lib = torch.cat((lib, torch.pow(x_j / x_i, i)), 1)

        # four #8
        if tri:
            lib = torch.cat((lib, torch.sin(x_j)), 1)
            lib = torch.cat((lib, torch.cos(x_j)), 1)
            lib = torch.cat((lib, torch.sin(x_j * x_i)), 1)
            lib = torch.cat((lib, torch.cos(x_j * x_i)), 1)
            lib = torch.cat((lib, torch.sin(x_j - x_i)), 1)
            lib = torch.cat((lib, torch.cos(x_j - x_i)), 1)
            lib = torch.cat((lib, torch.sin(x_j) * x_i), 1)
            lib = torch.cat((lib, torch.cos(x_j) * x_i), 1)

        # Exp:4
        lib = torch.cat((lib, torch.exp(x_j)), 1)
        lib = torch.cat((lib, torch.exp(x_j * x_i)), 1)
        lib = torch.cat((lib, torch.exp(x_j - x_i)), 1)
        lib = torch.cat((lib, x_i * torch.exp(x_j)), 1)

        #activate
        if activate:
            ##sigmoid 12
            lib = torch.cat((lib, torch.sigmoid(x_j)), 1)
            lib = torch.cat((lib, torch.sigmoid(x_j * x_i)), 1)
            lib = torch.cat((lib, torch.sigmoid(x_j - x_i)), 1)
            lib = torch.cat((lib, x_i * torch.sigmoid(x_j)), 1)
            ##tanh
            lib = torch.cat((lib, torch.tanh(x_j)), 1)
            lib = torch.cat((lib, torch.tanh(x_j * x_i)), 1)
            lib = torch.cat((lib, torch.tanh(x_j - x_i)), 1)
            lib = torch.cat((lib, x_i * torch.tanh(x_j)), 1)

            lib = torch.cat((lib, torch.pow(x_j, 1)/(torch.pow(x_j, 1) + 1)), 1)
            lib = torch.cat((lib, torch.pow(x_i * x_j, 1)/(torch.pow(x_i * x_j, 1) + 1)), 1)
            lib = torch.cat((lib, torch.pow(x_j - x_i, 1)/(torch.pow(x_j - x_i, 1) + 1)), 1)
            lib = torch.cat((lib, torch.pow(x_j, 1) * x_i/(torch.pow(x_j, 1) + 1)), 1)

        #####clamp#########
        lib = torch.clamp(lib, min=-1000, max=1000)
        #####negetive#########
        # nlib = -lib
        # lib = torch.cat((lib, nlib), 1)
    return lib




if __name__ == '__main__':
    print('Hello World')