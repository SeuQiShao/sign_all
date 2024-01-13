import numpy as np
import torch
from torch_geometric.data import DataLoader
import os
import sys
sys.path.append("..")
sys.path.append(".")
from utils_file import arg_parser
from model.utils import *
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import InMemoryDataset, Dataset, Data
from itertools import repeat, product, chain

class SimulationDynamic(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SimulationDynamic, self).__init__(root)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data.x.max(), self.data.x.mean(), self.data.x.std())
        
        self.data.x = self.data.x/self.data.x.max()
        #self.data.x = self.data.x - 273.15

        #self.data.x = (self.data.x - self.data.x.mean())/self.data.x.std()
    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def get(self, idx):
        if self.slices:
            data = Data()
            for key in self.data.keys:
                item, slices = self.data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        else:
            data = self.data
        return data


    def process(self):
        # Read data into huge `Data` list.
        data_list = torch.load(self.raw_paths[0])
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



# def get_off_diag_idx(num_atoms):
#     return np.ravel_multi_index(
#         np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
#         [num_atoms, num_atoms],
#     )


# def get_dataset(args):
#     path = args.datadir
#     path = os.path.join(path, args.dataset)
    
#     data_names = os.listdir(path)
#     ##train_loader
#     for i in data_names:
#         if i.startswith('data_Adjs_train'):
#             train_adj_path = os.path.join(path,i)
#             train_Adj = np.load(train_adj_path)
#         elif i.startswith('data_train'):
#             train_data_path = os.path.join(path,i)
#             train_data = np.load(train_data_path)
#         elif i.startswith('data_do_train'):
#             train_do_path = os.path.join(path,i)
#             train_do = np.load(train_do_path)
#         elif i.startswith('data_t_train'):
#             train_t_path = os.path.join(path,i)
#             train_t = np.load(train_t_path)
#         elif i.startswith('data_Adjs_test'):
#             test_adj_path = os.path.join(path,i)
#             test_Adj = np.load(test_adj_path)
#         elif i.startswith('data_test'):
#             test_data_path = os.path.join(path,i)
#             test_data = np.load(test_data_path)
#         elif i.startswith('data_do_test'):
#             test_do_path = os.path.join(path,i)
#             test_do = np.load(test_do_path)
#         elif i.startswith('data_t_test'):
#             test_t_path = os.path.join(path,i)
#             test_t = np.load(test_t_path)
#         elif i.startswith('data_Adjs_val'):
#             val_adj_path = os.path.join(path,i)
#             val_Adj = np.load(val_adj_path)
#         elif i.startswith('data_val'):
#             val_data_path = os.path.join(path,i)
#             val_data = np.load(val_data_path)
#         elif i.startswith('data_do_val'):
#             val_do_path = os.path.join(path,i)
#             val_do = np.load(val_do_path)
#         elif i.startswith('data_t_val'):
#             val_t_path = os.path.join(path,i)
#             val_t = np.load(val_t_path)
        
#     num_atoms = train_Adj.shape[2]  # 1400,1,5,5
#     train_Adj = train_Adj.reshape(train_Adj.shape[0],train_Adj.shape[1],-1)
#     test_Adj = test_Adj.reshape(test_Adj.shape[0],test_Adj.shape[1],-1)
#     val_Adj = val_Adj.reshape(val_Adj.shape[0],val_Adj.shape[1],-1)

#     off_diag_idx = get_off_diag_idx(num_atoms)
#     train_Adj = train_Adj[:,:,off_diag_idx]  # 1400,1,25
#     test_Adj = test_Adj[:,:,off_diag_idx]
#     val_Adj = val_Adj[:,:,off_diag_idx]


#     scaler = MinMaxScaler()
#     train_Adj = torch.from_numpy(train_Adj).float()
#     test_Adj = torch.from_numpy(test_Adj).float()
#     val_Adj = torch.from_numpy(val_Adj).float()
#     train_data = scaler.fit_transform(train_data.reshape(-1,1)).reshape(train_data.shape[0],train_data.shape[1],-1)
#     test_data = scaler.transform(test_data.reshape(-1,1)).reshape(test_data.shape[0],test_data.shape[1],-1)
#     val_data = scaler.transform(val_data.reshape(-1,1)).reshape(val_data.shape[0],val_data.shape[1],-1)
#     train_data = torch.from_numpy(train_data).float()
#     test_data = torch.from_numpy(test_data).float()
#     val_data = torch.from_numpy(val_data).float()
#     train_do = torch.tensor(train_do, dtype=torch.int64)
#     test_do = torch.tensor(test_do, dtype=torch.int64)
#     val_do = torch.tensor(val_do, dtype=torch.int64)
#     train_t = torch.from_numpy(train_t).float()
#     test_t = torch.from_numpy(test_t).float()
#     val_t = torch.from_numpy(val_t).float()

#     train_dataset = TensorDataset(train_data, train_Adj, train_do, train_t)
#     test_dataset = TensorDataset(test_data, test_Adj, test_do, test_t)
#     val_dataset = TensorDataset(val_data, val_Adj, val_do, val_t)

#     train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
#     test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
#     val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

#     return train_data_loader, test_data_loader, val_data_loader

# def unpack_batch(args, batch):
#     device = args.device
#     data, adj, do, t = batch
#     data = data.to(device)
#     adj = adj.to(device)
#     do = do.to(device)
#     t = t.to(device)
#     return data, adj, do, t


if __name__ == '__main__':
    SimulationDynamic('/home/shaoqi/code/SIGN/dataset/HeatDiffusion_10_power_law')


