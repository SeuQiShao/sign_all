from __future__ import division
from __future__ import print_function

from collections import defaultdict
from ctypes import util
import re
import time
import torch
from tqdm import tqdm
import random
from utils_file import arg_parser, data_loader
import numpy as np

from model.modules import *
from model import utils, model_loader
from torch_geometric.utils import degree

from sklearn import linear_model

def lasso_AIC(model, x, y):
    mse = np.mean((model.predict(x) - y)**2)
    p = (np.abs(model.coef_) > 0).sum() + np.sum(np.abs(model.intercept_) > 0)
    aic = x.shape[0] * np.log(mse) + 2 * p
    #bic = p * np.log(x.shape[0]) - x.shape[0] * np.log(mse)
    return aic


# def lasso_F(args, batchs):
#     print('Start Lasso mask...')
#     rate = 0.7
#     device = args.device
#     batchs = batchs.to(device)
#     data, edge_index, batch, t = batchs.x, batchs.edge_index, batchs.batch, batchs.t.reshape(-1,args.time_stamp)[0]
#     ploy_p = torch.arange(1, args.poly_p)
#     ploy_n = -torch.arange(1, args.poly_n)
#     # x_dot = (data[:,:,args.k].diff(1)/t.diff()[0]).reshape(-1)
#     # x1 = utils.fun_lib(data[:,:-1,args.k].reshape(-1,1), ploy_p, ploy_n, device)
#     # x2 = utils.fun_fc_lib3(data[:,:-1,:].reshape(-1,args.dims), ploy_p, device)
#     x_dot = (data[:,:,args.k].diff(1)/t.diff()[0]).reshape(-1)
#     x1 = utils.fun_lib(data[:,:-1,args.k].reshape(-1,1), ploy_p, ploy_n, device)
#     x2 = utils.fun_fc_lib3(data[:,:-1,:].reshape(-1,args.dims), ploy_p, device)
#     ####normalization
#     x_data = torch.cat((x1[:,1:], x2), 1)
#     # x_data = (x_data - x_data.min(0)[0])/(x_data.max(0)[0] - x_data.min(0)[0] + 1e-5)
#     # x_dot = (x_dot - x_dot.min())/(x_dot.max() - x_dot.min() + 1e-5)
#     model1 = linear_model.Lasso(alpha = 0.01, fit_intercept=True)
#     model2 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = 5, fit_intercept=True)
#     #model3 = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.1, fit_intercept=True)
#     model1.fit(x_data.cpu(), x_dot.cpu())
#     model2.fit(x_data.cpu(), x_dot.cpu())

#     #model3.fit(x_data.cpu(), x_dot.cpu())
#     # if model1.score(x_data.cpu(), x_dot.cpu()) > model2.score(x_data.cpu(), x_dot.cpu()):
#     #     coef = model1.coef_
#     # else:
#     #     coef = model2.coef_
#     coef = rate * model2.coef_ + (1 - rate) * model1.coef_
#     return coef

def lasso_CF(args, batchs):
    print('Start Lasso mask...')
    device = args.device
    batchs = batchs.to(device)
    data, edge_index, batch, t = batchs.x, batchs.edge_index, batchs.batch, batchs.t.reshape(-1,args.time_stamp)[0]
    ploy_p = torch.arange(1, args.poly_p)
    ploy_n = -torch.arange(1, args.poly_n)
    #num_atoms
    # x_dot = (data[:,:,args.k].diff(1)/t.diff()[0]).reshape(-1)
    # x1 = utils.fun_lib(data[:,:-1,args.k].reshape(-1,1), ploy_p, ploy_n, device)
    # x2 = utils.fun_fc_lib3(data[:,:-1,:].reshape(-1,args.dims), ploy_p, device)
    nums = max(0.005 * data.shape[0], args.lasso_node_num)
    nums_ = min(data.shape[0], nums)
    random_index = random.sample(list(np.arange(data.shape[0])), int(nums_))
    x_data = torch.tensor([]).to(device)
    x_dot = torch.tensor([]).to(device)
    for i in random_index:
        neighbor_i = edge_index[1][edge_index[0] == i]
        neighbor_num = len(neighbor_i)
        if neighbor_num > args.lasso_neighbor_num:
            neighbor_index = random.sample(list(neighbor_i), args.lasso_neighbor_num)
            neighbor_coef = neighbor_num/args.lasso_neighbor_num
        else:
            neighbor_index = neighbor_i
            neighbor_coef = 1
        x_neighbor = data[neighbor_index, :-1, args.k]
        x_i = data[i, :-1, args.k]
        x_c = 0
        for j in x_neighbor:
            x_c += utils.coupled_fun_lib(x_i.reshape(-1,1), j.reshape(-1,1), ploy_p, ploy_n, device, activate=args.activate)
        if args.agg == 'mean':
            x_c = x_c/x_neighbor.shape[0]
        x_c = x_c * neighbor_coef
        x_dot0 = (data[i,:,args.k].diff(1)/t.diff()[0]).reshape(-1)
        x1 = utils.fun_lib(data[i,:-1,args.k].reshape(-1,1), ploy_p, ploy_n, device, activate=args.activate)
        #x2 = utils.fun_fc_lib3(data[i,:-1,:].reshape(-1,args.dims), ploy_p, device)
        ####normalization
        x_data0 = torch.cat((x1[:,1:], x_c), 1)
        x_dot = torch.cat((x_dot, x_dot0),0)
        x_data = torch.cat((x_data, x_data0),0)
    model1 = linear_model.Lasso(alpha = 0.01, fit_intercept=True)
    model2 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = 5, fit_intercept=True)
    model1_0 = linear_model.Lasso(alpha = 0.01, fit_intercept=False)
    model2_0 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = 5, fit_intercept=False)
    model1.fit(x_data.cpu(), x_dot.cpu())
    model2.fit(x_data.cpu(), x_dot.cpu())
    model1_0.fit(x_data.cpu(), x_dot.cpu())
    model2_0.fit(x_data.cpu(), x_dot.cpu())
    model = [model1, model2, model1_0, model2_0]
    aic = [lasso_AIC(model1, x_data.cpu(), x_dot.cpu().numpy()),
           lasso_AIC(model2, x_data.cpu(), x_dot.cpu().numpy()),
           lasso_AIC(model1_0, x_data.cpu(), x_dot.cpu().numpy()),
           lasso_AIC(model2_0, x_data.cpu(), x_dot.cpu().numpy()),
           ]
    model_index = np.argmin(aic)
    return model[model_index].coef_, model[model_index].intercept_



def forward_pass_and_eval(
    args,
    encoder,
    decoder,
    batchs,
    epoch,
    c_mask = None,
    f_mask = None,
    save = False
):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))
    steps = 0

    #################### INPUT DATA ####################
    device = args.device
    batchs = batchs.to(device)
    data, edge_index, batch, t = batchs.x, batchs.edge_index, batchs.batch, batchs.t.reshape(-1,args.time_stamp)[0]

    #######sample data#######
    if args.decoder == 'CGSI':
        data = data[:,::10,:]
        t = t[::10]
        batchs.t = t
        batchs.x = data


    degree_in = degree(batchs.edge_index[0])
    degree_loss_coef = torch.log2(degree_in + 1)
    data = data.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)
    t = t.to(device)
    batch_size = batch.max()
    if len(data.shape) == 2:
        data = data.unsqueeze(2)
    target = data[:, 1:, [args.k]]

    # #################### DATA WITH UNOBSERVED TIME-SERIES ####################
    predicted_atoms = args.num_atoms




    #################### ENCODER ####################
    # if use_encoder:
   


    ##############Dynamic l1
    if epoch < args.l1_epochs:
       lambda_l1 = 0.001
    else:
       lambda_l1 = 0


    ################### DECODER ####################
    if args.decoder is not None:
        output,wc, wf = decoder(
            t,
            batchs,
            c_mask = c_mask,
            f_mask = f_mask
        )

    if save:
        true_data = target[:10,:,0].cpu().detach().numpy().T
        pred_data = output[:10,:,0].cpu().detach().numpy().T
        for i in range(true_data.shape[-1]):
            np.savetxt('true_{}_dim_{}.csv'.format(args.ode_model,args.k), true_data, delimiter=',')
            np.savetxt('pred_{}_dim_{}.csv'.format(args.ode_model,args.k), pred_data, delimiter=',')




    
    #################### MAIN LOSSES ####################
    if args.UseTD:
        time_delay = torch.arange(start = len(t), end = 1, step = -1)
        time_delay = time_delay.to(device)
    else:
        time_delay = torch.ones(len(t) -1).to(device)
    losses['loss_wc'] = utils.l1_loss(wc)
    losses['loss_wf'] = utils.l1_loss(wf)
    losses["loss_mse"] = F.mse_loss(output, target)
    losses["loss_mape"] = utils.MAPE(output, target)
    losses['weight_loss_mse'] = (time_delay * (degree_loss_coef * ((output - target)**2).mean(-1).permute(1,0)).mean(-1)).mean()
    total_loss = losses["weight_loss_mse"] + lambda_l1 * (args.lam_f * losses['loss_wf'] + args.lam_c * losses['loss_wc'] )
    losses["loss"] = total_loss

    losses["inference time"] = time.time() - start

    return losses, wc, wf

if __name__ == '__main__':
    pass


