import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
from pandas import DataFrame
import datetime
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch.utils import data
import torch_geometric.data as geo_data
from torch_geometric.nn import radius_graph

# File paths
class sstg_process():
    def __init__(self):
        self.path = "/home/shaoqi/SSTG/"
        self.lat = []
        self.lon = []
        self.poolSize = 6 #0.5
        self.save_path = 'enso_sstg.npy'
        if os.path.exists(self.save_path):
            self.data = np.load(self.save_path)
            self.lon = np.load('enso_lon.npy')
            self.lat = np.load('enso_lat.npy')
            print(self.data.shape)
        else:
            self.concat_sstg()
            print(self.data.shape)
        self.t_coef = 0.1
        self.r = 0.5



    def get_pacific_data(self,year,path):
        img_path = "/home/shaoqi/SSTG/{}/{}img".format(year,path)
        hdr_path = "/home/shaoqi/SSTG/{}/{}hdr".format(year,path)  # Assuming the .hdr file has the same name as the .img file

        # Open the ENVI file using rasterio
        with rasterio.open(img_path) as src:
            # Read the data as a NumPy array
            data = src.read(1)  # Assuming there is only one band (band index 1)
        # sns.heatmap(data)
        # plt.savefig('sst.png')
        #plt.show()
        # You can now work with the data as a NumPy array
        #print(data.shape)  # Print the shape of the data
        with open(hdr_path, 'r') as hdr_file:
            for line in hdr_file:
                if line.strip().startswith("map info"):
                    map_info = line.strip().split("= ")[1]
                    map_info = map_info.strip("{}").split(", ")
                    coordinate_system = map_info[-1]
                    ul_x = float(map_info[3])
                    ul_y = float(map_info[4])
                    x_resolution = float(map_info[5])
                    y_resolution = float(map_info[6])
        # Define coordinates for the Pacific Ocean region
        lat_min = -5.0
        lat_max = 5.0
        lon_min = 180 - 170
        lon_max = 180 - 120
        # Calculate the indices for the Pacific Ocean region
        row_min = int((90 - lat_max) / x_resolution)
        row_max = int((90 - lat_min) / x_resolution)
        col_min = int((lon_min) / y_resolution)
        col_max = int((lon_max) / y_resolution)
        # Subset the data to the Pacific Ocean region
        if lon_min * lon_max < 0:
            pacific_sst_data_e = data[row_min:row_max, col_min-1:]
            pacific_sst_data_w = data[row_min:row_max, :col_max + 1]
            pacific_sst_data = np.concatenate((pacific_sst_data_e, pacific_sst_data_w), axis=1)
        else:
            pacific_sst_data = data[row_min:row_max, col_min:col_max]

        #print(pacific_sst_data.shape)
        if len(self.lon) == 0:
            lat = []
            lon = []
            for row in range(row_min, row_max):
                lat.append(ul_y - row * y_resolution)
            for col in range(col_min, col_max):
                lon.append(col * x_resolution)
            self.lat = np.array(lat)[::self.poolSize]
            self.lon = np.array(lon)[::self.poolSize]
            np.save('enso_lon.npy', self.lon)
            np.save('enso_lat.npy', self.lat)
        sns.heatmap(pacific_sst_data)
        plt.savefig('enso.png')
        pacific_sst_data[pacific_sst_data < -300] = -1000000000
        return pacific_sst_data
    
    def pooling_sst(self, pacific_sst_data):
        in_row = pacific_sst_data.shape[0]
        in_col = pacific_sst_data.shape[1]
        if in_row % self.poolSize:
            padding_in_row = self.poolSize - in_row % self.poolSize
        else: 
            padding_in_row = 0 
        if in_col % self.poolSize:
            padding_in_col = self.poolSize - in_col % self.poolSize
        else:
            padding_in_col = 0 
        padding_data = np.lib.pad(pacific_sst_data,((padding_in_row,0),(padding_in_col,0)))
        pool_out = padding_data.reshape(padding_data.shape[0]//self.poolSize, self.poolSize, padding_data.shape[1]//self.poolSize, self.poolSize)
        pool_out = pool_out.mean(axis=(1, 3))
        print(pool_out.shape)
        return pool_out




    def concat_sstg(self):
        files = os.listdir(self.path)
        files.sort()
        #print(files)
        pool_sst_data = []
        for i in files:
            file_names = os.listdir(os.path.join(self.path, i))
            file_names.sort()
            for j in file_names:
                if j[-3:] == 'img':
                    pacific_sst_data = self.get_pacific_data(i,j[:-3])
                    pool_sst_data.append(self.pooling_sst(pacific_sst_data))           
                    print(i,j,' finished!')
        pacific_sst_data = np.stack(pool_sst_data)
        self.data = pacific_sst_data
        np.save('enso_sstg.npy', self.data)

    def build_graph_data(self):
        data = geo_data.Data()
        x = torch.tensor(self.data.reshape(self.data.shape[0], -1).T[:,:,None], dtype=torch.float32) #[192, 192, 192..] #self.graph_data.x.reshape(94,192,-1)
        x = x[:,:120]
        lon = torch.tensor(self.lon, dtype=torch.float32) 
        lat = torch.tensor(self.lat, dtype=torch.float32)
        xy = torch.cartesian_prod(lat, lon)
        data.lon = lon
        data.lat = lat
        print(xy.shape)
        index = x[:,:,0].mean(1) > -300
        data.x = x[index]
        data.xy = xy[index]
        data.edge_index = radius_graph(data.xy, r = self.r)
        data.t = self.t_coef * torch.arange(data.x.shape[1])
        self.graph_data = data
        print(data.x.shape)
        print(data.edge_index.shape)
        path = '/home/shaoqi/code/SIGN_all/SIGN_data/enso_{}'.format(data.x.shape[0])
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path,'raw'))
        self.graph_savepath = os.path.join(path,'raw/data1.pt')   
        torch.save([data], self.graph_savepath)





if __name__ == '__main__':
    sstgs = sstg_process()
    sstgs.build_graph_data()
    #sstgs.get_pacific_data('2019_SSTG','20191201_20191231.')
