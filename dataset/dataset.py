import os
import pandas as pd
import numpy as np

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Literal


def get_dataset(data_set, subset, split):
    ds = load_dataset(data_set, subset)
    return ds[split].to_pandas()


def get_owned_dataset(data_path):
    # 提供更多读取方式，和预测一起整理一下
    df = pd.read_csv(data_path).dropna()
    data = df.values
    if data.size % 224 != 0:
        raise ValueError(f"数据大小 {data.size} 不能被 224 整除，无法重塑为 (-1, 224)")
    # 重塑数据为 (-1, 224)
    reshaped_data = data.reshape(-1, 224)
    # 创建一个全是 0 的列
    zero_column = np.zeros((reshaped_data.shape[0], 1))
    # 将 reshaped_data 和 zero_column 拼接成新的数组
    new_data = np.hstack((reshaped_data, zero_column))
    # 将新的数组转换为 DataFrame
    owned_df = pd.DataFrame(new_data, columns=[f"col_{i}" for i in range(225)])
    return owned_df


class SignalDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, num_classes, normalize_type: Optional[Literal["mean-std", "min-max"]] = None):
        if normalize_type == "mean-std":
            data_frame = (data_frame - data_frame.mean()) / data_frame.std()
        elif normalize_type == "min-max":
            data_frame = (data_frame - data_frame.min()) / (data_frame.max() - data_frame.min())
        self.data = torch.tensor(data_frame.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = F.one_hot(torch.tensor(data_frame.iloc[:, -1].values, dtype=torch.long), num_classes=num_classes)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple:
        sample = self.data[index].unsqueeze(0)
        label = self.labels[index]
        return sample, label


class SignalDatasetCreator:
    def __init__(self, data_set, transfer_task, num_classes):
        self.data_set = data_set
        self.source = transfer_task[0]
        self.target = transfer_task[1]
        self.num_classes = num_classes

    def data_split(self, batch_size, num_workers, device):
        # 这里源域和目标域都是我们提供用来验证迁移学习的正确性
        # get source train and val
        data_frame_source = get_dataset(self.data_set, self.source[0], self.source[1])
        data_set_source = SignalDataset(data_frame_source, self.num_classes)
        lengths_source = [round(0.8 * len(data_set_source)), len(data_set_source) - round(0.8 * len(data_set_source))]
        train_data_source, eval_data_source = random_split(data_set_source, lengths_source)
        source_train = DataLoader(dataset=train_data_source, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        source_val = DataLoader(dataset=eval_data_source, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        # get target train and val
        data_frame_target = get_dataset(self.data_set, self.target[0], self.target[1])
        data_set_target = SignalDataset(data_frame_target, self.num_classes)
        lengths_target = [round(0.8 * len(data_set_target)), len(data_set_target) - round(0.8 * len(data_set_target))]
        train_data_target, eval_data_target = random_split(data_set_target, lengths_target)
        target_train = DataLoader(dataset=train_data_target, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        target_val = DataLoader(dataset=eval_data_target, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        return source_train, source_val, target_train, target_val

    def owned_data_split(self, data_path, batch_size, num_workers, device):
        # 这里目标域是用户自己提供的数据集
        # get source train and val
        data_frame_source = get_dataset(self.data_set, self.source[0], self.source[1])
        data_set_source = SignalDataset(data_frame_source, self.num_classes)
        lengths_source = [round(0.8 * len(data_set_source)), len(data_set_source) - round(0.8 * len(data_set_source))]
        train_data_source, eval_data_source = random_split(data_set_source, lengths_source)
        source_train = DataLoader(dataset=train_data_source, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        source_val = DataLoader(dataset=eval_data_source, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        # get target train and val
        data_frame_target = get_owned_dataset(data_path)
        data_set_target = SignalDataset(data_frame_target, self.num_classes)
        lengths_target = [round(0.8 * len(data_set_target)), len(data_set_target) - round(0.8 * len(data_set_target))]
        train_data_target, eval_data_target = random_split(data_set_target, lengths_target)
        target_train = DataLoader(dataset=train_data_target, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        target_val = DataLoader(dataset=eval_data_target, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        return source_train, source_val, target_train, target_val
    
    def local_data_split(self, data_path, batch_size, num_workers, device):
        # 源域与目标域均采用本地提供
        # get source train and val
        data_frame_source = pd.read_csv(os.path.join(data_path, self.source[0], f"{self.source[1]}.csv")).dropna()
        data_set_source = SignalDataset(data_frame_source, self.num_classes)
        lengths_source = [round(0.8 * len(data_set_source)), len(data_set_source) - round(0.8 * len(data_set_source))]
        train_data_source, eval_data_source = random_split(data_set_source, lengths_source)
        source_train = DataLoader(dataset=train_data_source, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        source_val = DataLoader(dataset=eval_data_source, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        # get target train and val
        data_frame_target = pd.read_csv(os.path.join(data_path, self.target[0], f"{self.target[1]}.csv")).dropna()
        data_set_target = SignalDataset(data_frame_target, self.num_classes)
        lengths_target = [round(0.8 * len(data_set_target)), len(data_set_target) - round(0.8 * len(data_set_target))]
        train_data_target, eval_data_target = random_split(data_set_target, lengths_target)
        target_train = DataLoader(dataset=train_data_target, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        target_val = DataLoader(dataset=eval_data_target, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        return source_train, source_val, target_train, target_val
