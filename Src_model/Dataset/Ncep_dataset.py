import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

SINGLE_VAR = ["PS", "SLP", "PHIS"]
PRESS_VAR = ["H", "OMEGA", "QI", "QL", "QV", "RH", "T", "U", "V"]
ADD_VAR = ["VOR", "DIV"]
PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825,
               800, 775, 750, 725, 700, 650, 600, 550,
               500, 450, 400, 350, 300, 250, 200, 150, 100]
LEVEL = len(PRESS_LEVEL)

INP_CHANNELS = len(SINGLE_VAR) + LEVEL * (len(PRESS_VAR) + len(ADD_VAR))

LIST_VAR = [f"{var}0" for var in SINGLE_VAR]
LIST_VAR += [f"{var}{level}" for var in PRESS_VAR for level in PRESS_LEVEL]
LIST_VAR += [f"{var}{level}" for var in ADD_VAR for level in PRESS_LEVEL]


OMEGA = 7.29 * 1e-5

def vorticity(u, v, lat, lon):
    x = lon# * 111 * 1000
    y = lat# * 111 * 1000
    
    y[y == 0] = 0.5
    
    lat = np.deg2rad(lat)
    
    # vx = np.divide(v, x, out=np.full_like(v, np.nan, dtype=np.float64), where=(x != 0))
    # uy = np.divide(u, y, out=np.full_like(u, np.nan, dtype=np.float64), where=(y != 0))
    vx = v / x
    uy = u / y
    
    return vx - uy + 2 * OMEGA * np.sin(lat) * 111 * 1000

def divergence(u, v, lat, lon):
    x = lon# * 111 * 1000
    y = lat# * 111 * 1000
    
    y[y == 0] = 0.5
    
    # ux = np.divide(u, x, out=np.full_like(u, np.nan, dtype=np.float64), where=(x != 0))
    # vy = np.divide(v, y, out=np.full_like(v, np.nan, dtype=np.float64), where=(y != 0))
    ux = u / x
    vy = v / y
    
    return ux + vy

def meshgrid(lat, lon, lvl = 1):
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lat_grid = np.repeat(lat_grid[np.newaxis, :, :], lvl, axis=0)
    lon_grid = np.repeat(lon_grid[np.newaxis, :, :], lvl, axis=0)
    
    return lat_grid, lon_grid

class Merra2Full(Dataset):
    """Dataset for MERRA-2 features with normalization from summary statistics."""

    def __init__(self,
                 merra2_path: Path,
                 stat_path: Path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/data_train_statistics.xlsx',
                 dataset: str = "train",):
        
        super().__init__()

        self.data_path = (pd.read_csv(merra2_path)
                          .dropna(subset=["Label"])
                          .reset_index(drop=True))

        stat = pd.read_excel(stat_path)
        stat["variable_name"] = stat["variable"] + stat["level"].astype(str)
        stat = stat.set_index("variable_name")

        self.mean = stat.loc[LIST_VAR, "mean"].to_numpy().reshape(-1, 1, 1)
        self.std = stat.loc[LIST_VAR, "std"].to_numpy().reshape(-1, 1, 1)
        self.dataset = dataset

    def _read_data(self, row) -> np.ndarray:
        with xr.open_dataset(row["FullPath"]) as ds:
            feature_list = []

            for var in SINGLE_VAR:
                feature_list.append(ds[var].squeeze().data)

            for var in PRESS_VAR:
                feature_list.extend(ds[var].squeeze().data[:LEVEL])

            u = ds["U"].squeeze().data[:LEVEL]
            v = ds["V"].squeeze().data[:LEVEL]
            lon = ds["longitude"].data
            lat = ds["latitude"].data[::-1]

            lat_grid, lon_grid = meshgrid(lat, lon, LEVEL)
            feature_list.extend(vorticity(u, v, lat_grid, lon_grid))
            feature_list.extend(divergence(u, v, lat_grid, lon_grid))

        features = np.stack(feature_list, axis=0)
        features = (features - self.mean) / self.std
        return np.nan_to_num(features)

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data_path.iloc[idx]
        input = self._read_data(row)
        input = torch.tensor(input, dtype=torch.float)

        label = int(row["Label"])
        label = torch.tensor(label, dtype=torch.float).type(torch.LongTensor)

        return input, label


class LData(L.LightningDataModule):
    def __init__(self,
                 dataset_class=Merra2Full,
                 train_path: Path = Path("path"),
                 val_path: Path = Path("path"),
                 test_path: Path = Path("path"),
                 predict_path: Path = Path("/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv"),
                 batch_size: int = 32,
                 pin_memory: bool = torch.cuda.is_available(),
                 num_workers: int = os.cpu_count(),
                 **kwargs,):
        
        super().__init__()

        self.train_dataset = dataset_class(merra2_path=train_path, dataset="train", **kwargs)
        self.val_dataset = dataset_class(merra2_path=val_path, dataset="val", **kwargs)
        self.test_dataset = dataset_class(merra2_path=test_path, dataset="test", **kwargs)
        self.predict_dataset = dataset_class(merra2_path=predict_path, dataset="predict", **kwargs)

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          shuffle=True,)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          shuffle=False,)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          shuffle=False,)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers,
                          shuffle=False,)


if __name__ == "__main__":
    data = LData(
        dataset_class=Merra2Full,
        train_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',
        val_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',
        test_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',
        predict_path='/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Base/merra2_domain_sample.csv',
    )

    train_loader = data.train_dataloader()
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
        break
