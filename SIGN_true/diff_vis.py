from __future__ import division
from __future__ import print_function
import os
from collections import defaultdict
import time
import numpy as np
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
from model.modules import *
from utils_file import arg_parser, logger, data_loader, forward_pass_and_eval
from model import utils, model_loader
from torch_geometric.data import DataLoader
import tqdm
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")
import copy




def test(encoder, decoder, epoch, c_mask = None, f_mask = None):
    args.shuffle_unobserved = False
    test_losses = defaultdict(list)
    save = False

    if args.load_folder == "":
        ## load model that had the best validation performance during training
        if args.use_encoder:
            encoder.load_state_dict(torch.load(args.encoder_file))
        decoder.load_state_dict(torch.load(args.decoder_file))

    if args.use_encoder:
        encoder.eval()
    decoder.eval()
    for batch_idx, batchs in enumerate(All_loader):
        with torch.no_grad():
            losses, wc, wf = forward_pass_and_eval.forward_pass_and_eval(
                args,
                encoder,
                decoder,
                batchs,
                epoch,
                c_mask = c_mask,
                f_mask = f_mask,
                save = save   
            )
            losses["wc"] = wc
            if args.UseF:
                losses["wf"] = wf
        test_losses = utils.append_losses(test_losses, losses)

    string = logs.result_string("test", epoch, test_losses)
    logs.write_to_log_file(string)
    logs.append_test_loss(test_losses)
    

    logs.create_log(
        args,
        decoder=decoder,
        encoder=encoder,
        optimizer=optimizer,
        final_test=True,
        test_losses=test_losses,
    )


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = arg_parser.parse_args()
    logs = logger.Logger(args)
    args.load_folder = "logs/2023-10-24T070837.816577"
    if args.GPU_to_use is not None:
        logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

    dataset = data_loader.SimulationDynamic(args.root)

    All_loader =  DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    #all_batchs = copy.deepcopy(All_loader.dataset.data)
    
    for batch_idx, batchs in tqdm.tqdm(enumerate(All_loader)):
        all_batchs = copy.deepcopy(batchs)
        break

    args.k = 0
    diffs = all_batchs.x[:,:,0].diff().cpu().numpy()/0.1
    np.savetxt('diffs.csv', diffs, delimiter=',')
    args.T_max_k = utils.generate_season(args, all_batchs.x[:,:,0].cpu().numpy(), all_batchs.t.cpu().numpy())
    print(args.T_max_k)

    c_mask = torch.tensor([[ 0.0000,  1.6080,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000, -0.0177,  0.0000,  0.0000,  0.0000,
          0.0000]]).T
    f_mask = torch.tensor([[-0.6082,  0.0000,  0.0000,  0.0000,  0.5936,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0479,
         -0.0286,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0550,
          0.0000,  0.0158,  0.0000, -0.0451, -0.0464,  0.0000,  0.0000,  0.0000,
          0.0144,  0.0000]]).T

    encoder, decoder, optimizer, scheduler = model_loader.load_model(args)




    if args.test:
        test(encoder, decoder,  1, c_mask, f_mask)

        #break