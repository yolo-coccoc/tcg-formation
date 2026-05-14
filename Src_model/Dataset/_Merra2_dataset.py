import sys
sys.path.insert(0, '/N/slate/tnn3/DucHGA/TC-formation/Src_model')
import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
import lightning as L
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# from torchvision.transforms import v2
from Utils.New_features import *

SINGLE_VAR = ["PS", "SLP", 'PHIS']
PRESS_VAR = ["H", "OMEGA", "QI", "QL", "QV", "RH", "T", "U", "V"]
ADD_VAR = ["VOR", "DIV"]
PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825,
               800, 775, 750, 725, 700, 650, 600, 550,
               500, 450, 400, 350, 300, 250, 200, 150, 100]

LEVEL = len(PRESS_LEVEL)

# number of channels after concat
INP_CHANNELS = len(SINGLE_VAR) + LEVEL * (len(PRESS_VAR)+len(ADD_VAR))

LIST_VAR = [var + '0' for var in SINGLE_VAR]
LIST_VAR.extend([var + str(level) for var in PRESS_VAR for level in PRESS_LEVEL])
LIST_VAR.extend([var + str(level) for var in ADD_VAR for level in PRESS_LEVEL])
print(len(LIST_VAR))
#===============================
# Description: 
#   - Fully loaded dataset before train progress
#   - Apply standard normalization to dataset
#   - Apply aggregate (OPTIONAL)
#===============================

class Merra2Full(Dataset):
    def __init__(self, 
                 merra_path: Path,
                 stat_path: Path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/data_train_statistics.xlsx',
                 dataset = 'train',
                 ratio = 30,
                 num_workers: int = 1,):
        
        super().__init__()
        self.data_path = pd.read_csv(merra_path)#.iloc[:100]
        self.data_path = self.data_path[~ self.data_path['Label'].isna()].reset_index(drop=True)
        
        self.stat = pd.read_excel(stat_path)
        self.stat['variable_name'] = self.stat['variable'] + self.stat['level'].astype(str)
        self.stat.set_index('variable_name', inplace=True)
        self.mean = self.stat['mean'].loc[LIST_VAR].to_numpy()[:, np.newaxis, np.newaxis]
        self.std = self.stat['std'].loc[LIST_VAR].to_numpy()[:, np.newaxis, np.newaxis]
        self.dataset = dataset
        
        
    def read_data(self, row):
        nc_path = row['FullPath']
        input = []
        ds = xr.open_dataset(nc_path)

        for var in SINGLE_VAR:
            arr = ds.variables[var].data.squeeze()
            input.append(arr)

        for var in PRESS_VAR:
            arr = ds.variables[var].data.squeeze()[: LEVEL]

            input.extend(arr)
            
        U = ds.variables['U'].data.squeeze()[: LEVEL]
        V = ds.variables['V'].data.squeeze()[: LEVEL]
        lon = ds.coords['longitude'].data
        lat = ds.coords['latitude'].data[:: -1]
        lat_grid, lon_grid = meshgrid(lat, lon, LEVEL)
        VOR = vorticity(U, V, lat_grid, lon_grid)
        input.extend(VOR)
        DIV = divergence(U, V, lat_grid, lon_grid)
        input.extend(DIV)
        
        ds.close()
        input = np.array(input)
        res = (input - self.mean) / self.std
        res[np.isnan(res)] = 0
        
        return res
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        input = self.read_data(self.data_path.iloc[idx])
        input = torch.tensor(input, dtype=torch.float)
        
        label = self.data_path.iloc[idx]['Label']
        label = torch.tensor(label, dtype=torch.float).type(torch.LongTensor)
        
        return input, label

class LData(L.LightningDataModule):
    def __init__(self,
                 dataset_class = Merra2Full,
                 train_path: Path = 'path',
                 val_path: Path = 'path',
                 test_path: Path = 'path',
                 predict_path: Path = '/N/slate/tnn3/DucHGA/map_test.csv',
                 batch_size: int = 32,
                 pin_memory: bool = torch.cuda.is_available(),
                 num_workers: int = os.cpu_count(),
                 **kwargs):
        
        super().__init__()
        self.train_dataset = dataset_class(merra_path=train_path, dataset='train', num_workers=num_workers, **kwargs)
        self.val_dataset = dataset_class(merra_path=val_path, dataset='val', num_workers=num_workers, **kwargs)
        self.test_dataset = dataset_class(merra_path=test_path, dataset='test', num_workers=num_workers, **kwargs)
        self.predict_dataset = dataset_class(merra_path=predict_path, dataset='test', num_workers=num_workers, **kwargs)
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size = self.batch_size,
                          pin_memory = self.pin_memory,
                          num_workers = self.num_workers,
                          shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size = self.batch_size,
                          pin_memory = self.pin_memory,
                          num_workers = self.num_workers,
                          shuffle = False)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size = self.batch_size,
                          pin_memory = self.pin_memory,
                          num_workers = self.num_workers,
                          shuffle = False)
        
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size = self.batch_size,
                          pin_memory = self.pin_memory,
                          num_workers = self.num_workers,
                          shuffle = False)

if __name__ == '__main__':
    data = LData(DatasetClass=Merra2Full,
                      train_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',
                      val_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',
                      test_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',
                      predict_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',)
    
    trainLoader = data.train_dataloader()
    for i, (x, y) in enumerate(trainLoader):
        print(x.shape, y.shape)
