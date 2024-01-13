from netCDF4 import Dataset
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
import datetime
import math
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric as pyg
from torch.utils import data
import torch_geometric.data as geo_data
from torch_geometric.nn import radius_graph



# poolsize
# file_path = os.path.join(self.path, self.fname)
class DataProcess():
    def __init__(self, path, poolSize, time_bin):
        self.path = path
        self.poolSize = poolSize
        # self.time_bin = time_bin
        self.loader()
        self.processtime(time_bin)

        # self.file_path = os.path.join(self.path, self.fname)

    # nc.variables.keys() ['lat', 'lon', 'time', 'air']
    def loader(self):
        nc = Dataset(self.path)
        cut = np.abs(nc.variables['lat'][:].data) < 66.34
        self.data = nc.variables['air'][:].data[:,cut,:]
        self.lon = nc.variables['lon'][:]
        self.lat = nc.variables['lat'][:][cut]
        self.time = nc.variables['time'][:].data

    def processtime(self, time_bin):
        if time_bin == 'day':
            self.time_bin = 4
        elif time_bin == 'year':
            self.time_bin = self.data.shape[0]
        elif time_bin == 'month':
            self.time_bin = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if int(self.path[-7 : -3]) % 4 == 0:
                self.time_bin[1] += 1
            self.time_bin = np.array(self.time_bin) * 4
        else:
            self.time_bin = time_bin

    def temporal_processing(self, data):
        if hasattr(self.time_bin, 'shape'):
            cumsum_time = np.cumsum(self.time_bin, 0)
            pool_data = []
            for i in range(len(self.time_bin)):
                if i == 0:
                    if self.data.shape[0] < cumsum_time[i]:
                        pool_data.append(data[:data.shape[0],:,:].mean(0))
                        break
                    else:
                        pool_data.append(data[:cumsum_time[i],:,:].mean(0))
                else:
                    if self.data.shape[0] < cumsum_time[i]:
                        pool_data.append(data[cumsum_time[i-1]:,:,:].mean(0))
                        break
                    else:
                        pool_data.append(data[cumsum_time[i-1]:cumsum_time[i],:,:].mean(0))

            pool_data = np.stack(pool_data)
            

        else:
            if data.shape[0] % self.time_bin == 0:
                pool_data = data.reshape(self.time_bin,-1,data.shape[1], data.shape[2]).mean(0)
            else:
                temp_len = data.shape[0]//self.time_bin * self.time_bin
                pool_data0 = data[:temp_len].reshape(-1,self.time_bin,data.shape[1], data.shape[2]).mean(1)
                pool_data1 = data[temp_len:].mean(0)[None,:,:]
                pool_data = np.vstack((pool_data0, pool_data1))

        return pool_data  


    
    def spatial_processing(self):
        in_row = self.data.shape[1]
        in_col = self.data.shape[2]
        if in_row % self.poolSize:
            padding_in_row = self.poolSize - in_row % self.poolSize
        else: 
            padding_in_row = 0 
        if in_col % self.poolSize:
            padding_in_col = self.poolSize - in_col % self.poolSize
        else:
            padding_in_col = 0 
        padding_data = np.lib.pad(self.data,((0,0),(padding_in_row,0),(padding_in_col,0)))
        pool_out = padding_data.reshape(padding_data.shape[0], padding_data.shape[1]//self.poolSize, self.poolSize, padding_data.shape[2]//self.poolSize, self.poolSize)
        pool_out = pool_out.mean(axis=(2, 4))
        return pool_out

    
    def ST_block(self):
        spatial_out = self.spatial_processing()
        pool_data = self.temporal_processing(spatial_out)
        np.save('two_meter_lon.npy', self.lon.data)
        np.save('two_meter_lat.npy', self.lat.data)
        return pool_data
    

class climate_graph():
    def __init__(self, timebin) -> None:
        self.data_name = 'two_meter_1_{}.npy'.format(timebin)
        self.lon_name = 'two_meter_lon.npy'
        self.lat_name = 'two_meter_lat.npy'
        self.data = np.load(self.data_name)
        self.lon = np.load(self.lon_name)
        self.lat = np.load(self.lat_name)
        self.t_coef = 0.05
        nodes = len(self.lon) * len(self.lat)
        self.savepath = '/home/shaoqi/code/SIGN_all/SIGN_data/climate_{}/raw/data1.pt'.format(nodes)


    def build_graph_data(self):
        data = geo_data.Data()
        data.x = torch.tensor(self.data.reshape(self.data.shape[0], -1).T[:,:,None], dtype=torch.float32) #[192, 192, 192..] #self.graph_data.x.reshape(94,192,-1)
        data.lon = torch.tensor(self.lon, dtype=torch.float32) #self.graph_data.lon.reshape(60,140)
        data.lat = torch.tensor(self.lat, dtype=torch.float32)
        XY = torch.cartesian_prod(data.lat, data.lon)
        print(XY.shape)
        data.edge_index = radius_graph(XY, r = 3)
        data.t = self.t_coef * torch.arange(data.x.shape[1])
        self.graph_data = data
        print(data.x.shape)
        print(data.edge_index.shape)        
        torch.save([data], self.savepath)




def generate_climate_seq():
    path = '/home/shaoqi/DataSet/two_meter/'
    FLAG = True
    poolSize = 1
    time_bin = 10 * 4
    data_dir = os.listdir(path)
    # data_dir.sort(key = lambda x: int(x[-7:-3]), reverse= False)
    data_dir.sort()
    # time_index = ['year','month','day']
    for file_path in data_dir[-8:]:
        file_name = os.path.join(path, file_path)
        if file_name[-3:] =='.nc':
            Climate_process = DataProcess(file_name, poolSize, time_bin)
            # Climate_process.temporal_processing()
            # Climate_process.spatial_processing()
            pool_data = Climate_process.ST_block()
            print(file_name,'finished')
            if FLAG:
                final_data = pool_data
                FLAG = False
            else:
                final_data = np.vstack((final_data,pool_data))
    print(final_data.shape)
    np.save('two_meter_'+str(poolSize)+'_'+str(time_bin)+'.npy', final_data)

if __name__ == '__main__':
    generate_climate_seq()
    climate_data = climate_graph(40)
    climate_data.build_graph_data()