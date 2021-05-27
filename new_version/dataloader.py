import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg


def create_dataloader(mode, type=0, snr=0):
    if mode == 'train':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )
    elif mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )


class Wave_Dataset(Dataset):
    def __init__(self, mode, type, snr):
        # load data
        if mode == 'train':
            print('<Training dataset>')
            print('Load the data...')
            self.input_path = './input/C1_train_dataset.npy'
            self.input = np.load(self.input_path)
        elif mode == 'valid':
            print('<Validation dataset>')
            print('Load the data...')
            self.input_path = './input/C1_validation_dataset.npy'
            self.input = np.load(self.input_path)

            # # if you want to use a part of the dataset
            # self.input = self.input[:500]
        elif mode == 'test':
            print('<Test dataset>')
            print('Load the data...')
            self.input_path = './input/C1_test_dataset.npy'

            self.input = np.load(self.input_path)
            self.input = self.input[type][snr]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx][0]
        targets = self.input[idx][1]

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        return inputs, targets
