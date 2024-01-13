import argparse
import torch
import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3, help="Random seed.")
    parser.add_argument(
        "--GPU_to_use", type=int, default=None, help="GPU to use for training"
    )

    ############## GSI hyperparameter ##############
    parser.add_argument(
        "--poly_p", type=int, default=4, help="Polynomial of Library."
    )
    parser.add_argument(
        "--poly_n", type=int, default=2, help="Neg_Polynomial of Library."
    )
    parser.add_argument(
        "--activate", type=bool, default=False, help="activation of Library."
    )



    parser.add_argument("--teacher", type=int, default=5, help="add teacher every t.")

    ############## training hyperparameter ##############

    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train."
    )
    parser.add_argument(
        "--l1_epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=40, help="Number of samples per batch."  # default=32
    )
    parser.add_argument("--lam_c", type=float, default=1, help="lambda of w_f, w_c.")
    parser.add_argument("--lam_f", type=float, default=1, help="lambda of w_f, w_c.")
    parser.add_argument("--lam_s", type=float, default=1, help="lambda of similarity.")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_decay",
        type=int,
        default=20,
        help="After how epochs to decay LR by a factor of gamma.",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="LR decay factor.")  
    parser.add_argument(
        "--UseTD",
        type=bool,
        default=False,
        help='use time-delay'
    )

    parser.add_argument(
        "--lasso_neighbor_num",
        type=int,
        default=100,
        help='lasso_neighbor_num'
    )
    parser.add_argument(
        "--lasso_node_num",
        type=int,
        default=100,
        help='lasso_node_num'
    )
    ############## DataSet ##############
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of Workers."
    )
    parser.add_argument('--network', type=str,
                    choices=['random', 'power_law', 'small_world','oregon', 'git', 'Twitch', 'email', 'CA'], default='power_law')
    parser.add_argument('--ode_model', type=str,
                    choices=['HeatDiffusion', 'Kuramoto'], default='Kuramoto')
    parser.add_argument('--time_stamp', type=int, default=80, help="number of timesteps.")
    parser.add_argument('--num_atoms', type=int, default=100000) # 10670 37700 168114 36692 23133
    parser.add_argument('--dims', type=int, default=1)
    parser.add_argument('--save', type=bool, default=True)
    

    ############## architecture ##############
    parser.add_argument(
        "--decoder",
        type=str,
        default='CGSI',
        help="Type of decoder model (DGSI, CGSI).",
    )
    parser.add_argument(
        "--adjoint",
        type=bool,
        default='False',
        help="Type of decoder model (DGSI, CGSI).",
    )

    parser.add_argument(
        "--agg",
        type=str,
        default='add',
        help="agg function: add/mean.",
    )

    parser.add_argument(
        "--UseF",
        type=bool,
        default=True,
        help='use self upgrade'
    )
    parser.add_argument("--F_coef", type=float, default=1, help="Coefs of F(x).")

    parser.add_argument(
        "--UseEdgeAttr",
        type=bool,
        default=False,
        help='use edge A'
    )

    parser.add_argument(
        "--UseLasso",
        type=bool,
        default=True,
        help='use Lasso'
    )

    ########### Different variants for variational distribution q ###############
    parser.add_argument(
        "--dont_use_encoder",
        action="store_true",
        # default=False,
        help="If true, replace encoder with distribution to be estimated",
        default=True,
    )
    parser.add_argument(
        "--lr_z",
        type=float,
        default=0.1,
        help="Learning rate for distribution estimation.",
    )
   

   
    ############## loading and saving ##############  

    parser.add_argument(
        "--save_folder",
        type=str,
        default="logs",
        help="Where to save the trained model, leave empty to not save anything.",
    )
    parser.add_argument(
        "--expername",
        type=str,
        default="",
        help="If given, creates a symlinked directory by this name in logdir"
        "linked to the results file in save_folder"
        "(be careful, this can overwrite previous results)",
    )
    parser.add_argument(
        "--sym_save_folder",
        type=str,
        default="../logs",
        help="Name of directory where symlinked named experiment is created."
    )
    parser.add_argument(
        "--load_folder",
        type=str,
        default='',
        # default="logs/k5_0.817",
        help="Where to load pre-trained model if finetuning/evaluating. "
        + "Leave empty to train from scratch",
    )

   
    ############## almost never change these ##############
    parser.add_argument(
        "--no_validate", action="store_true", default=False, help="Do not validate results throughout training."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument("--var", type=float, default=1e-3, help="Output variance.")

    parser.add_argument(
        "--invariant",
        type=bool,
        default=True,
        help="Use invariant data.",
    )


    args = parser.parse_args()
    args.test = True



    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.validate = not args.no_validate
    args.use_encoder = not args.dont_use_encoder
    args.time = datetime.datetime.now().isoformat().replace(':','')
    args.root = '/home/shaoqi/code/SIGN_all/SIGN_data/' + args.ode_model +'_' + str(args.num_atoms) + '_' + args.network
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.type != "cpu":
        if args.GPU_to_use is not None:
            torch.cuda.set_device(args.GPU_to_use)
        torch.cuda.manual_seed(args.seed)
        args.num_GPU = 1  # torch.cuda.device_count()
        args.batch_size_multiGPU = args.batch_size * args.num_GPU
    else:
        args.num_GPU = None
        args.batch_size_multiGPU = args.batch_size

    return args
