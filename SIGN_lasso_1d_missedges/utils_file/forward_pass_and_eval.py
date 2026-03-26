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
    save = False
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
    target = data[:, 1:, [args.k]]

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
        output,wc, wf = decoder(
            t,
            input_batch,
            c_mask = c_mask,
            f_mask = f_mask
        )

    if save:
        true_data = target[:10,:,0].cpu().detach().numpy().T
        pred_data = output[:10,:,0].cpu().detach().numpy().T
        for i in range(true_data.shape[-1]):
            np.savetxt('true_{}_dim_{}.csv'.format(args.suffix, args.k), true_data, delimiter=',')
            np.savetxt('pred_{}_dim_{}.csv'.format(args.suffix, args.k), pred_data, delimiter=',')




    
    #################### MAIN LOSSES ####################
    losses['loss_wc'] = utils.l1_loss(wc)
    losses['loss_wf'] = utils.l1_loss(wf)
    losses["loss_mse"] = F.mse_loss(output, target)
    losses["loss_mape"] = utils.MAPE(output, target)
    losses["loss"] = losses["loss_mse"]
    losses["inference time"] = time.time() - start
    losses['wc'] = wc
    losses['wf'] = wf
    return losses

if __name__ == '__main__':
    pass


