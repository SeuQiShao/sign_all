import numpy as np
from geotiff import GeoTiff
import os
import torch_geometric.data as geo_data
import matplotlib
import torch
from torch_geometric.nn import radius_graph
import copy

class tif_processor:
    def __init__(self):
        self.dirpath = '/home/shaoqi/DataSet//SM_Rec'
        self.savepath = '/home/shaoqi/code/SIGN_all/SIGN_data/tb_8400/raw/data1.pt'
        self.file_years = os.listdir(self.dirpath)
        self.file_years.sort()
        self.data_name = 'tifdata.npy'
        if os.path.exists(self.data_name):
            self.data = np.load(self.data_name)
            self.lon = np.load('lon_' + self.data_name)
            self.lat = np.load('lat_' + self.data_name)
        else:
            self.get_tifs()
        self.time_stamp = 511
        self.pool_size = 30
        self.time_pooling(self.pool_size)
        self.t_coef = 0.1
        ############check 0
        zeros = (self.data == 0).sum(0)
        np.savetxt('zeros.txt', zeros)
        for i in range(self.data.shape[0]):
            self.data[i][self.data[i] == 0] = self.data[i].mean()

    def time_pooling(self, pool_size = 5):
        time_length, H, W = self.data.shape
        length = time_length//pool_size * pool_size
        self.data = self.data[:length].reshape(-1, pool_size, H, W).mean(1)

    def get_tif_data(self,tiff_file):
        #tiff_file = os.path.join(dirpath, '2002_001_ECV_filled.tif')
        geo_tiff = GeoTiff(tiff_file)
        zarr_array = geo_tiff.read()
        data = np.array(zarr_array)
        #lon, lat = geo_tiff.get_coord_arrays()
        #print(data.shape)
        ###check
        if data.shape == (60,140):
            return data
        else:
            print('error size!')
    
    def get_range(self,tiff_file):
        geo_tiff = GeoTiff(tiff_file)
        lon, lat = geo_tiff.get_coord_arrays()
        self.lon = lon
        self.lat = lat
        
    def get_tifs(self):
        data = []
        for i in self.file_years:
            files = os.path.join(self.dirpath, i)
            file_names = os.listdir(files)
            file_names.sort()
            for j in file_names:
                tiff_file = os.path.join(files,j)
                data.append(self.get_tif_data(tiff_file))
                print(tiff_file + ' finished!')
        data = np.stack(data)

        print(data.shape)
        self.get_range(tiff_file)
        self.data = data    
        np.save(self.data_name,data)
        np.save('lon_' + self.data_name,self.lon)
        np.save('lat_' + self.data_name,self.lat)
    
    def build_graph_data(self):
        data = geo_data.Data()
        data.x = torch.tensor(self.data.reshape(self.data.shape[0], -1).T[:,:,None], dtype=torch.float32) #[140,140,..,140] #self.graph_data.x.reshape(60,140,-1)
        data.lon = torch.tensor(self.lon.reshape(-1,1), dtype=torch.float32) #self.graph_data.lon.reshape(60,140)
        data.lat = torch.tensor(self.lat.reshape(-1,1), dtype=torch.float32)
        XY = torch.cat([data.lon,data.lat],1)
        print(XY.shape)
        data.edge_index = radius_graph(XY, r = 0.5)
        data.t = self.t_coef * torch.arange(data.x.shape[1])
        self.graph_data = data
        print(data.x.shape)
        torch.save([data], self.savepath)
    
    def devide_graph(self):
        data_list = []
        time_length = self.graph_data.x.shape[1]
        for i in range(time_length//self.time_stamp + 1):
            #data0 = geo_data.Data()
            if i == time_length//self.time_stamp:
                end = time_length
                start = time_length - self.time_stamp
            else:
                start = i * self.time_stamp                    
                end = (i + 1) * self.time_stamp
            data0 = copy.copy(self.graph_data)
            data0.x = data0.x[:,start:end]
            data0.t = data0.t[start:end]
        data_list.append(data0)
        print(data0.x.shape)
        torch.save(data_list, self.savepath)
        
        
if __name__ == '__main__':
    tifs = tif_processor()
    tifs.build_graph_data()
    #tifs.devide_graph()