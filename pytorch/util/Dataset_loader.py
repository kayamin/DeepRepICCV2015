import numpy as np
import pandas as pd
import scipy as sp
import h5py
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Dataset_loader(Dataset):

    def __init__(self, filename_df, transforms=None):

        # Create whole image path list

        self.data_path = filename_df['filename'].values

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        filename = self.data_path[idx]
        f = h5py.File(filename)
        data_x = f['data_x'].value
        data_y = f['data_y'].value
        f.close()

        data_x = data_x.reshape(-1, 20, 50, 50)
        data_y = data_y.astype('int')

        return [data_x, data_y]
