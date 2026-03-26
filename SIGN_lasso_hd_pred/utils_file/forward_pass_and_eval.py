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
import copy


def forward_pass_and_eval(
    args,
    decoder,
    batchs,
    epoch,
    c_mask = None,
    f_mask = None,
    save = False,
    part = 0
):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))
    steps = 0

    #################### INPUT DATA ####################
    device = args.device
    batchs = batchs.to(device)
    data, edge_index, batch, t = batchs.x, batchs.edge_index, batchs.batch, batchs.t.reshape(-1,args.time_stamp)[0]


    # degree_in = degree(batchs.edge_index[0])
    # degree_loss_coef = torch.log2(degree_in + 1)
    data = data.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)
    t = t.to(device)
    batch_size = batch.max()
    if len(data.shape) == 2:
        data = data.unsqueeze(2)
    #target = data[:, 1:, :]

    # #################### DATA WITH UNOBSERVED TIME-SERIES ####################
    if args.decoder == 'CGSI':
        input_batch = copy.deepcopy(batchs)
        data = data[:,::10,:]
        t = t[::10]
        input_batch.t = t
        input_batch.x = data
        target = data
    else:
        input_batch = batchs


    #################### ENCODER ####################
    # if use_encoder:
   

    ################### DECODER ####################
    if args.decoder is not None:
        output,wc, wf, start_step = decoder(
            t,
            input_batch,
            c_mask = c_mask,
            f_mask = f_mask
        )
    target = data[:,start_step:start_step+output.shape[1],:]


    if save:
        nodes = output.shape[0]
        idx = torch.randperm(nodes)[:1000]
        true_data = target[idx,:,:].cpu().detach().permute(1,0,2).numpy()
        pred_data = output[idx,:,:].cpu().detach().permute(1,0,2).numpy()
        for i in range(true_data.shape[-1]):
            np.savetxt('true_{}_dim_{}_part_{}.csv'.format(args.ode_model,i,part), true_data[:,:,i], delimiter=',')
            np.savetxt('pred_{}_dim_{}_part_{}.csv'.format(args.ode_model,i,part), pred_data[:,:,i], delimiter=',')




    
    #################### MAIN LOSSES ####################
    losses['loss_wc'] = utils.l1_loss(wc)
    losses['loss_wf'] = utils.l1_loss(wf)
    losses["loss_mse"] = F.mse_loss(output, target)
    losses["loss_mape"] = utils.MAPE(output, target)
    losses["loss"] = losses["loss_mse"]
    losses["inference time"] = time.time() - start
    # losses['wc'] = wc
    # losses['wf'] = wf
    return losses, wc, wf

if __name__ == '__main__':
    pass


