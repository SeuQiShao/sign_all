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



def train(c_mask, f_mask):
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        t_epoch = time.time()
        train_losses = defaultdict(list) 
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()        

        for batch_idx, batchs in tqdm.tqdm(enumerate(All_loader)):
            batchs = batchs.to(args.device)
            # Loss & back
            # batchs = batchs.to(args.device)
            losses, wc, wf = forward_pass_and_eval.forward_pass_and_eval(
                args,
                encoder,
                decoder,
                batchs,
                epoch,
                c_mask = c_mask,
                f_mask = f_mask                  
            )
            decoder0 = copy.deepcopy(decoder)
            encoder0 = copy.deepcopy(encoder)
            losses["wc"] = wc
            losses["wf"] = wf
            optimizer.zero_grad()
            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            train_losses = utils.append_losses(train_losses, losses)
        
        
        string = logs.result_string("train", epoch, train_losses, t=t_epoch)
        logs.write_to_log_file(string)
        logs.append_train_loss(train_losses)
        scheduler.step()
        if epoch > args.l1_epochs:
            #c_mask = torch.zeros(wc.shape[0], 1)
            c_mask[wc.detach().abs() < max(0.025 * wc.detach().abs().max(), 0.005)] = 0
            #f_mask = torch.zeros(wf.shape[0], 1)
            f_mask[wf.detach().abs() < max(0.025 * wf.detach().abs().max(), 0.005)] = 0

        val_loss = np.mean(train_losses["loss_mse"]) + 1e-4 * ((wc.detach().abs() > 0.0001).sum() + (wf.detach().abs() > 0.0001).sum())
        if val_loss < best_loss:
            print("Best model so far, saving...")
            logs.create_log(
                args,
                encoder=encoder0,
                decoder=decoder0,
                optimizer=optimizer,
            )
            best_loss = val_loss
            best_epoch = epoch

        logs.draw_loss_curves()

    return best_epoch, epoch, best_loss, c_mask, f_mask



def test(encoder, decoder, epoch, c_mask = None, f_mask = None):
    args.shuffle_unobserved = False
    test_losses = defaultdict(list)
    save = args.save
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
    
    with open('result.log', 'a+') as f:
        f.write(args.root + ' ' + str(args.seed) + args.decoder +'\n')
        f.write(string)
        f.write('\n')
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

    if args.GPU_to_use is not None:
        logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

    dataset = data_loader.SimulationDynamic(args.root)

    All_loader =  DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    all_batchs = copy.deepcopy(All_loader.dataset.data)
    



    for k in range(args.dims):
        c_mask = None
        f_mask = None
        args.k = 1
        ###### mask     
        if args.UseLasso:
            c_num = 3 * (args.poly_p - 1) + 4 * (args.poly_n - 1) + 4 
            f_num = (args.poly_n + 1) + 3 * args.poly_p
            if args.activate:
                c_num += 12
                f_num += 3
            if args.tri:
                c_num += 8
                f_num += 3
            c_mask = torch.zeros(c_num, 1)
            f_mask = torch.zeros(f_num, 1)
            coef, inter = forward_pass_and_eval.lasso_CF(args, all_batchs)
            coef = torch.tensor(coef, dtype=torch.float32, requires_grad=False)
            if np.abs(inter) > 0.05:
                f_mask[0, 0] = inter
            else:
                f_mask[0, 0] = 0
            f_mask[1:,0] = coef[:(f_num-1)]
            c_mask[:,0] = coef[(f_num-1):]
        # c_mask = torch.zeros(c_num, 1)
        # f_mask = torch.zeros(f_num, 1)
        #c_mask[1] = 0.01
        # f_mask[0] = 0.99
        # f_mask[3] = -0.99
        # f_mask[5] = -4.99
        with open('result.log', 'a+') as f:
            f.write(args.root + ' ' + str(args.seed) + args.decoder +'\n')
            f.write(str(c_mask.T))
            f.write('\n')
            f.write(str(f_mask.T))
            f.write('\n')
        print('c_mask:', c_mask.T)
        print('f_mask:', f_mask.T)

        encoder, decoder, optimizer, scheduler = model_loader.load_model(args)
        try:
            best_epoch, epoch, mse1, c_mask, f_mask = train(c_mask, f_mask)

        except KeyboardInterrupt:
            best_epoch, epoch = -1, -1

        print("dim_{} Optimization Finished!".format(k + 1))
        logs.write_to_log_file("Best Epoch: {:04d}".format(best_epoch))

        if args.test:
            test(encoder, decoder,  epoch, c_mask, f_mask)

        #break