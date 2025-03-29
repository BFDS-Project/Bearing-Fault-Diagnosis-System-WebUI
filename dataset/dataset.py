import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.SequenceDatasets import dataset
from dataset.sequence_aug import *
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List
from PIL import Image
from utils.wavelet import ContinuousWaveletTransform


def get_dataset(data_set, subset, split):
    ds = load_dataset(data_set, subset)
    return ds[split].to_pandas()


class SignalDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, transform: transforms.Compose = None):
        # 这里我暂时没有用小波变换因为跑起来有点麻烦我后面想一下怎么解决
        self.data = torch.tensor(data_frame.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(data_frame.iloc[:, -1].values, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple:
        sample = self.data[index].unsqueeze(0)
        label = self.labels[index]

        # FIXME 这里还有问题先不transform
        # if self.transform:
        # sample = self.transform(sample)

        return sample, label


class SignalDatasetCreator:
    # FIXME 修改参数位置
    num_classes = 6
    inputchannel = 1

    def __init__(self, data_set, conditions, labels, transfer_task, normlizetype="0-1"):
        self.data_set = data_set
        # FIXME 可能用不上
        self.conditions = conditions
        self.labels = labels
        self.source = transfer_task[0]
        self.target = transfer_task[1]
        # FIXME 可能用不上
        self.normlizetype = normlizetype
        self.data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def data_split(self, batch_size, num_workers, device, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            data_frame = get_dataset(self.data_set, self.source[0], self.source[1])
            data_set = SignalDataset(data_frame, transform=self.data_transforms)
            lengths = [round(0.8 * len(data_set)), len(data_set) - round(0.8 * len(data_set))]
            train_data, eval_data = random_split(data_set, lengths)
            source_train = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
            source_val = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
            # get target train and val
            get_dataset(self.data_set, self.target[0], self.target[1])
            data_set = SignalDataset(data_frame, transform=self.data_transforms)
            lengths = [round(0.8 * len(data_set)), len(data_set) - round(0.8 * len(data_set))]
            train_data, eval_data = random_split(data_set, lengths)
            target_train = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
            target_val = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
            return source_train, source_val, target_train, target_val
        else:
            # get source train and val
            data_frame = get_dataset(self.data_set, self.source[0], self.source[1])
            data_set = SignalDataset(data_frame, transform=self.data_transforms)
            lengths = [round(0.8 * len(data_set)), len(data_set) - round(0.8 * len(data_set))]
            train_data, eval_data = random_split(data_set, lengths)
            source_train = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
            source_val = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

            # get target val
            get_dataset(self.data_set, self.target[0], self.target[1])
            data_frame = get_dataset(self.data_set, self.transfer_task[0, 0], self.transfer_task[0, 1])
            data_set = SignalDataset(data_frame, transform=self.data_transforms)
            target_val = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
            return source_train, source_val, target_val
