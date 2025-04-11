from typing import Literal, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from dataset.get_data import get_huggingface_dataset, get_local_dataset, get_user_dataset


class SignalDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, normalize_type: Optional[Literal["mean-std", "min-max"]] = None):
        if normalize_type == "mean-std":
            data_frame = (data_frame - data_frame.mean()) / data_frame.std()
        elif normalize_type == "min-max":
            data_frame = (data_frame - data_frame.min()) / (data_frame.max() - data_frame.min())
        self.data = torch.tensor(data_frame.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(data_frame.iloc[:, -1].values, dtype=torch.long)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple:
        sample = self.data[index].unsqueeze(0)
        label = self.labels[index]
        return sample, label


class SignalDatasetCreator:
    def __init__(self, data_set, labels, transfer_task, stratified_sampling):
        self.data_set = data_set
        self.labels = labels
        self.source = transfer_task[0]
        self.target = transfer_task[1]
        self.stratified_sampling = stratified_sampling

    def stratified_split(self, dataset, test_size=0.2):
        """
        使用 PyTorch 实现分层抽样，将数据集按标签分布划分为训练集和验证集。
        """
        labels = dataset.labels  # 获取标签
        num_classes = torch.unique(labels).size(0)  # 获取类别数
        train_indices = []
        val_indices = []

        # 遍历每个类别，按比例分配索引
        for class_idx in range(num_classes):
            class_indices = torch.where(labels == class_idx)[0]  # 获取当前类别的索引
            num_val = int(len(class_indices) * test_size)  # 验证集样本数
            num_train = len(class_indices) - num_val  # 训练集样本数

            # 随机打乱索引
            shuffled_indices = torch.randperm(len(class_indices))
            train_indices.extend(class_indices[shuffled_indices[:num_train]].tolist())
            val_indices.extend(class_indices[shuffled_indices[num_train:]].tolist())

        # 返回训练集和验证集的子集
        train_data = Subset(dataset, train_indices)
        val_data = Subset(dataset, val_indices)
        return train_data, val_data

    def data_split(self, batch_size, num_workers, device):
        # 获取源域数据集
        data_frame_source = get_huggingface_dataset(self.data_set, self.source[0], self.source[1])
        data_set_source = SignalDataset(data_frame_source)
        # 根据 stratified_sampling 参数选择分层抽样或随机划分
        if self.stratified_sampling:
            train_data_source, eval_data_source = self.stratified_split(data_set_source, test_size=0.2)
        else:
            lengths_source = [round(0.8 * len(data_set_source)), len(data_set_source) - round(0.8 * len(data_set_source))]
            train_data_source, eval_data_source = random_split(data_set_source, lengths_source)
        source_train = DataLoader(dataset=train_data_source, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        source_val = DataLoader(dataset=eval_data_source, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)

        # 获取目标域数据集
        data_frame_target = get_huggingface_dataset(self.data_set, self.target[0], self.target[1])
        data_set_target = SignalDataset(data_frame_target)
        # 根据 stratified_sampling 参数选择分层抽样或随机划分
        if self.stratified_sampling:
            train_data_target, eval_data_target = self.stratified_split(data_set_target, test_size=0.2)
        else:
            lengths_target = [round(0.8 * len(data_set_target)), len(data_set_target) - round(0.8 * len(data_set_target))]
            train_data_target, eval_data_target = random_split(data_set_target, lengths_target)
        target_train = DataLoader(dataset=train_data_target, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        target_val = DataLoader(dataset=eval_data_target, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)

        return source_train, source_val, target_train, target_val

    def user_data_split(self, data_path, batch_size, num_workers, device):
        # 获取源域数据集
        data_frame_source = get_huggingface_dataset(self.data_set, self.source[0], self.source[1])
        data_set_source = SignalDataset(data_frame_source)
        # 根据 stratified_sampling 参数选择分层抽样或随机划分
        if self.stratified_sampling:
            train_data_source, eval_data_source = self.stratified_split(data_set_source, test_size=0.2)
        else:
            lengths_source = [round(0.8 * len(data_set_source)), len(data_set_source) - round(0.8 * len(data_set_source))]
            train_data_source, eval_data_source = random_split(data_set_source, lengths_source)
        source_train = DataLoader(dataset=train_data_source, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        source_val = DataLoader(dataset=eval_data_source, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)

        # 获取目标域数据集
        data_frame_target = get_user_dataset(data_path)
        data_set_target = SignalDataset(data_frame_target)
        # 根据 stratified_sampling 参数选择分层抽样或随机划分
        if self.stratified_sampling:
            train_data_target, eval_data_target = self.stratified_split(data_set_target, test_size=0.2)
        else:
            lengths_target = [round(0.8 * len(data_set_target)), len(data_set_target) - round(0.8 * len(data_set_target))]
            train_data_target, eval_data_target = random_split(data_set_target, lengths_target)
        target_train = DataLoader(dataset=train_data_target, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        target_val = DataLoader(dataset=eval_data_target, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)

        return source_train, source_val, target_train, target_val
