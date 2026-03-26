from model import utils, model_loader
from torch_geometric.data import DataLoader
from utils_file import arg_parser, logger, data_loader
import multiprocessing as mp


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = arg_parser.parse_args()
    logs = logger.Logger(args)

    dataset = data_loader.SimulationDynamic(args.root)

    All_loader =  DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    for batch_idx, onebatchs in enumerate(All_loader):
        print('True dynamics:', onebatchs.para)
        syn = utils.R_syn_modified(onebatchs.x)
        print('syn of {}: {}'.format(args.stre, syn))
        break
    
