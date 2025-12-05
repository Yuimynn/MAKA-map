'''
This code defines a custom PyTorch Dataset class (DistDataset) designed to load and preprocess data for protein residue-residue distance prediction models.
'''
from torch.utils.data import Dataset, DataLoader
from dataio import *
import torch
import numpy as np
epsilon = 1e-7


class DistDataset(Dataset):
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, pad_size, expected_n_channels,
                 label_engineering=None):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.dim = dim
        self.pad_size = pad_size
        self.distmap_path = distmap_path
        self.expected_n_channels = expected_n_channels
        self.label_engineering = label_engineering
        self.on_epoch_begin()

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.pdb_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.pdb_id_list)

    def __getitem__(self, index):
        batch_list = [self.pdb_id_list[index]]
        X, Y = get_input_output_dist(batch_list, self.features_path, self.distmap_path, self.pad_size, self.dim,
                                     self.expected_n_channels)

        # Ensure X and Y are numpy arrays before converting to tensors
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().numpy()

        if self.label_engineering is None:
            X = torch.from_numpy(X[0]).float().permute(2, 0, 1)  # Change shape to (channels, height, width)
            Y = torch.from_numpy(Y[0]).float().permute(2, 0, 1).squeeze(
                -1)  # Change shape to (1, height, width) and remove the last dimension if it's singleton
        elif self.label_engineering == '100/d':
            X = torch.from_numpy(X[0]).float().permute(2, 0, 1)
            Y = torch.from_numpy(100.0 / Y[0]).float().permute(2, 0, 1).squeeze(-1)
        else:
            try:
                t = float(self.label_engineering)
                Y[Y > t] = t
            except ValueError:
                print('ERROR!! 未知标签工程参数!!')
                return None
            X = torch.from_numpy(X[0]).float().permute(2, 0, 1)
            Y = torch.from_numpy(Y[0]).float().permute(2, 0, 1).squeeze(-1)
        return X, Y
