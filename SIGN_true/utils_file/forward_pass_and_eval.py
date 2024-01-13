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
        else:
            x_c = x_c * neighbor_coef
        x_dot0 = (data[i,:,args.k].diff(1)/t.diff()[0]).reshape(-1)
        x1 = utils.fun_lib(data[i,:-1,args.k].reshape(-1,1), ploy_p, ploy_n, device, activate=args.activate)
        #x2 = utils.fun_fc_lib3(data[i,:-1,:].reshape(-1,args.dims), ploy_p, device)
        ####normalization
        if args.t_basis:
            x_t = utils.t_fun_lib(t,args.T_max_k, device)[:-1]
            x_data0 = torch.cat((x1[:,1:], x_t), 1)
        x_data0 = torch.cat((x_data0, x_c), 1)
        x_dot = torch.cat((x_dot, x_dot0),0)
        x_data = torch.cat((x_data, x_data0),0)    
    model1 = linear_model.Lasso(alpha = 0.01, fit_intercept=True)
    model2 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = 10, fit_intercept=True)
    #model1_0 = linear_model.Lasso(alpha = 0.01, fit_intercept=False)
    #model2_0 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = 10, fit_intercept=False)
    model1.fit(x_data.cpu(), x_dot.cpu())
    model2.fit(x_data.cpu(), x_dot.cpu())
    #model1_0.fit(x_data.cpu(), x_dot.cpu())
    #model2_0.fit(x_data.cpu(), x_dot.cpu())
    #model = [model1, model2, model1_0, model2_0]
    model = [model1, model2]
    aic = []
    for i in model:
        aic.append(lasso_AIC(i, x_data.cpu(), x_dot.cpu().numpy()))
    # aic = [lasso_AIC(model1, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model2, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model1_0, x_data.cpu(), x_dot.cpu().numpy()),
    #        lasso_AIC(model2_0, x_data.cpu(), x_dot.cpu().numpy()),
    #        ]
    model_index = np.argmin(aic)
    print('Use model:', model_index)
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
    lon = batchs.lon
    lat = batchs.lat
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
        '''
        Ali 	33.391	79.703
        Ali01	33.435	79.746
        Ali02	33.457	79.629
        Ali03	33.456	79.632
        WATERNET-18	38.093	100.282	3792
        WATERNET-20	38.105	100.371	3242
        WATERNET-21	38.061	100.22	3557
        WATERNET-22	38.178	100.198	3050
        WATERNET-23	38.057	100.319	3270
        WATERNET-24	37.949	100.758	3513
        WATERNET-25	38.037	100.227	3846
        WATERNET-27	38.067	100.564	3414
        WATERNET-28	38.023	100.231	3930
        CST 01	33.886	102.142
        CST 02	33.677	102.14
        CST 03	33.903	101.973
        CST 04	33.768	101.733
        CST 05	33.677	101.891
        NST 01	33.888	102.143
        NST 02	33.883	102.144
        NST 03	33.765	102.116
        NST 04	33.629	102.059
        MS3533	31.58681	91.79339
        MS3545	31.57351	91.91269
        MS3552	31.54569	91.98467
        MS3559	31.52711	92.04956
        MS3576	31.41003	91.97082
        MS3593	31.30087	91.84779
        MS3603	31.259	91.79928
        MS3614	31.17451	91.7597
        PL01	28.08	89.28
        PL02	28.06	89.28
        PL03	28.04	89.27
        PL04	28.02	89.29
        PL05	28.01	89.24
        PL06	27.98	89.26
        PL07	27.98	89.29
        SQ01	32.497	80.061
        SQ02	32.503	80.028
        SQ03	32.507	79.983
        SQ04	32.511	79.957
        SQ05	32.513	79.921
        SQ06	32.506	79.878
        SQ07	32.532	79.845
        SQ08	32.559	79.838
        SQ09	32.456	80.053
        '''

        if args.ode_model == 'tb':
            position = [(33.25, 79.75),(33.5, 79.75),(33.5, 79.5),(33.25, 79.5),
                        (38,100.25),(38,100),(38.25,100.25),(38.25,100),
                        (33.75,102),(33.5,101.75),(33.75,101.75),(33.5,102),
                        (31.5, 91.75),(31.25, 92),(31.5, 92),(31.25, 91.75),
                        (28, 89.25),(27.75, 89),(28, 89),(27.75, 89.25),
                        (32.5, 80),(32.25, 79.75), (32.25, 80),(32.5, 79.75)]
            index = []
            for i in position:
                mask = (batchs.lat == i[0]) & (batchs.lon == i[1])
                index.append(torch.nonzero(mask, as_tuple=False)[0,0].item())
            true_data = target[index,:,0].cpu().detach().numpy().T
            pred_data = output[index,:,0].cpu().detach().numpy().T
        elif args.ode_model == 'climate':
            true_data = target[5000:5100,:,0].cpu().detach().numpy().T
            pred_data = output[5000:5100,:,0].cpu().detach().numpy().T
        elif args.ode_model == 'enso':
            true_data = target[5000:5100,:,0].cpu().detach().numpy().T
            pred_data = output[5000:5100,:,0].cpu().detach().numpy().T
            np.savetxt('All_enso_true_{}_dim_{}.csv'.format(args.ode_model,args.k), target[:,:,0].cpu().detach().numpy().T, delimiter=',')
            np.savetxt('All_enso_pred_{}_dim_{}.csv'.format(args.ode_model,args.k), output[:,:,0].cpu().detach().numpy().T, delimiter=',')
            xy = batchs.xy.cpu().detach().numpy()
            np.savetxt('enso_xy.csv', xy, delimiter=',')

        for i in range(true_data.shape[-1]):
            np.savetxt('true_{}_dim_{}.csv'.format(args.ode_model,args.k), true_data, delimiter=',')
            np.savetxt('pred_{}_dim_{}.csv'.format(args.ode_model,args.k), pred_data, delimiter=',')
            # np.savetxt('All_true_{}_dim_{}.csv'.format(args.ode_model,args.k), target[:,:,0].cpu().detach().numpy().T, delimiter=',')
            # np.savetxt('All_pred_{}_dim_{}.csv'.format(args.ode_model,args.k), output[:,:,0].cpu().detach().numpy().T, delimiter=',')
        mse_error = ((output[:,:,0] - target[:,:,0])**2).mean(1).cpu().detach().numpy().T
        np.savetxt('error_{}_node_{}.csv'.format(args.ode_model,args.num_atoms), mse_error, delimiter=',')
        # np.savetxt('diff_c.csv', C_coef[:,:,0].cpu().detach().numpy(), delimiter=',')
        # np.savetxt('diff_f.csv', F_coef[:,:,0].cpu().detach().numpy(), delimiter=',')
        # np.savetxt('diffp.csv', (F_coef+C_coef)[:,:,0].cpu().detach().numpy(), delimiter=',')

    
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


