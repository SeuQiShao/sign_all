import os
from collections import defaultdict
import time
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from model.modules import *
from utils_file import arg_parser, logger, data_loader, primary_mask, forward_pass_and_eval
from model import utils, model_loader
from torch_geometric.data import DataLoader
import tqdm
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")



def train():
    #stage 1, generate mask
    f_mask, c_mask = primary_mask.generate_primary_mask_all(args, train_bactchs)

    #stage 2, training SIGN
    # load separator
    print('Start SIGN Training...')
    if args.load_folder == "":
        ## load model that had the best validation performance during training
        best_loss = np.inf
        best_epoch = 0
        soft_mask_c = torch.ones_like(c_mask, device=args.device)  # 初始化软掩码
        soft_mask_f = torch.ones_like(f_mask, device=args.device)  # 初始化软掩码

        for epoch in range(args.epochs):
            t_epoch = time.time()
            train_losses = defaultdict(list)
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            batchs = train_bactchs.to(args.device)
            batchs.train = True
            # 前向传播与损失计算
            losses, wc, wf = forward_pass_and_eval.forward_pass_and_eval(args, decoder, batchs, epoch, c_mask=c_mask * soft_mask_c, f_mask=f_mask * soft_mask_f)
            #wf, wc = losses['wf'], losses['wc']
            train_losses = utils.append_losses(train_losses, losses)
            string = logs.result_string("train", epoch, train_losses, t=t_epoch)
            logs.write_to_log_file(string)
            logs.append_train_loss(train_losses)
            for i in range(args.dims):
                #expression = utils.functions(args.poly_p, args.poly_n, losses['wf'][:,i],  losses['wc'][:,i], activate=args.activate)[0]
                expression = utils.functions(args.poly_p, args.poly_n, wf[:,i],  wc[:,i], activate=args.activate)[0]
                logs.write_to_log_file(expression)
            mae_loss = np.mean(train_losses["loss_mse"]) 
            
            if mae_loss < best_loss:
                print("Best model so far, saving...")
                logs.create_log(args, decoder=decoder, optimizer=optimizer)
                best_loss = mae_loss

            optimizer.zero_grad()
            loss = losses["loss"]
            loss.backward()
            optimizer.step()
            logs.draw_loss_curves()
            # decay coef:
            decay_factor = 0.9  # 衰减系数，可根据需要调整
            # 对于绝对值较小的权重，降低软掩码值
            soft_mask_c[wc.detach().abs() < max(0.001 * wc.detach().abs().max(), 0.005)] *= decay_factor
            soft_mask_f[wf.detach().abs() < max(0.001 * wf.detach().abs().max(), 0.005)] *= decay_factor

        decoder.load_state_dict(torch.load(args.decoder_file))
    else:
        decoder.load_state_dict(torch.load(args.decoder_file))
        print('Successed loading model.')
    decoder.eval()

    #stage3 Eval SIGN
    for i in range(0, N_part):
        for batch_idx, test_batchs in enumerate(All_loader):
            print('Data shape:', test_batchs.x.shape)
        test_batchs.x = test_batchs.x[:, i * N: (i + 1) * N, :].clone()
        test_losses = defaultdict(list)
        test_batchs0 = test_batchs.to(args.device)
        test_batchs0.train = False
        fianal_loss, wc, wf = forward_pass_and_eval.forward_pass_and_eval(args, decoder, test_batchs0, epoch, c_mask=c_mask * soft_mask_c, f_mask=f_mask * soft_mask_f, save = args.save, part = i)
        test_losses = utils.append_losses(test_losses, fianal_loss)
        string = logs.result_string("test", epoch, test_losses)
        logs.write_to_log_file(string)
        logs.append_test_loss(test_losses)
        with open('result.log', 'a+') as f:
            f.write(args.root + ' ' + str(args.seed) + args.decoder +'\n')
            f.write(string)
            f.write('\n')
    #expression = utils.functions(args.poly_p, args.poly_n, fianal_loss['wf'],  fianal_loss['wc'], activate=args.activate)[0]
    for i in range(args.dims):
        #expression = utils.functions(args.poly_p, args.poly_n, losses['wf'][:,i],  losses['wc'][:,i], activate=args.activate)[0]
        expression = utils.functions(args.poly_p, args.poly_n, wf[:,i],  wc[:,i], activate=args.activate)[0]
        logs.write_to_log_file(expression)
        with open('result.log', 'a+') as f:
            f.write(expression)
            f.write('\n')

    logs.create_log(
        args,
        decoder=decoder,
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
    print('device:', args.device)
    dataset = data_loader.SimulationDynamic(args.root)

    All_loader =  DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    for batch_idx, onebatchs in enumerate(All_loader):
        print('True dynamics:', onebatchs.para)
        break
    train_bactchs = onebatchs
    N_part = 20
    N_train = 16
    N = train_bactchs.x.shape[1]//N_part
    train_bactchs.x = onebatchs.x[:, :N* N_train, :].clone()
    
    # for batch_idx, test_batchs in enumerate(All_loader):
    #     print('Data shape:', test_batchs.x.shape)
    # test_batchs.x = test_batchs.x[:, N:2*N, :].clone()

    # for i in range(args.dims):
    #     args.k = i
    #     encoder, decoder, optimizer, scheduler = model_loader.load_model(args)
    #     train()
    encoder, decoder, optimizer, scheduler = model_loader.load_model(args)
    train()
  