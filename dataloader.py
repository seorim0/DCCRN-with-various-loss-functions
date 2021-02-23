import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg


# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def create_dataloader(mode):
    if mode == 'train':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch,  # max 3696 * snr types
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )    # max 1152


def create_dataloader_for_test(mode, type, snr):
    if mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset_for_test(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )    # max 192


class Wave_Dataset(Dataset):
    def __init__(self, mode):
        # load data
        if mode == 'train':
            print('<Training dataset>')
            print('Load the data...')
            self.input_path = './input/train_dataset.npy'
        elif mode == 'valid':
            print('<Validation dataset>')
            print('Load the data...')
            self.input_path = './input/validation_dataset.npy'

        self.input = np.load(self.input_path)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx][0]
        labels = self.input[idx][1]

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        return inputs, labels


class Wave_Dataset_for_test(Dataset):
    def __init__(self, mode, type, snr):
        # load data
        if mode == 'test':
            print('<Test dataset>')
            print('Load the data...')
            self.input_path = './input/recon_test_dataset.npy'

        self.input = np.load(self.input_path)
        self.input = self.input[type][snr]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx][0]
        labels = self.input[idx][1]

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        return inputs, labels
