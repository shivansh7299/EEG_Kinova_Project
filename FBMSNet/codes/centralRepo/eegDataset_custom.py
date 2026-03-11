import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch.utils.data import random_split, DataLoader


TIME_LEN = 248  # FBMSNet requires this to be divisible by 4


class EEGMultiBandDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.X = np.load(os.path.join(data_dir, f"X_{split}_fbmsnet.npy"))
        self.y = np.load(os.path.join(data_dir, f"y_{split}_fbmsnet.npy"))
        # Crop to 248 if necessary
        if self.X.shape[-1] > TIME_LEN:
            self.X = self.X[..., :TIME_LEN]
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (9, 8, 248)
        y = torch.tensor(self.y[idx])
        return x, y


def create_dataloaders(data_dir, batch_size=32, shuffle=True, num_workers=0):
    train = EEGMultiBandDataset(data_dir, split='train')
    test = EEGMultiBandDataset(data_dir, split='test')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
