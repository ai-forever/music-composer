import numpy as np
import torch
from torch.utils.data import Dataset


class EncodedDataset(Dataset):
    """
    Dataset class for training and evaluating the model.
    
    Parameters
    ----------
    ds_files : str
        path to file 'ds_files.pt'. The file contains the list of paths to encoded sequences (samples of dataset).
    prefix_path : str
        prefix_path will be added to paths in 'ds_files.pt'. Used sometimes for convenience.
    transform : MusicAugmentations
        in-fly augmentations for sequences.
    """
    def __init__(self, ds_files, prefix_path='', transform=None):
        self.transform = transform
        self.files = torch.load(ds_files)
        self.prefix_path = prefix_path
        self.genre2id = {'classic':0, 'jazz':1, 'calm':2, 'pop':3}
        self.genre = [self.genre2id.get(f.split('/')[1], 0) for f in self.files]  # 1 for 'encoded_data/GENRE/xxx.pt'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.prefix_path + self.files[idx])
        if self.transform:
            x = torch.from_numpy(self.transform(x))
        genre = self.genre[idx]
        return x, genre, idx
