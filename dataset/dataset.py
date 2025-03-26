import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List
from utils.wavelet import ContinuousWaveletTransform


def get_data_set(filepath: str) -> pd.DataFrame:
    data_set = pd.read_csv(filepath, header=None)
    return data_set


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class SignalDataset(Dataset):
    def __init__(self, data_set: np.ndarray, fs, num_classes, transform=None):
        """
        Args:
            data_set (pd.DataFrame): 数据集，最后一列是标签
            fs (int): 采样频率
            num_classes(int): 分类数
            transform (callable, optional): 图像变换
        """
        self.datas = data_set[:, :-1]  # 取出时间序列数据
        self.labels = data_set[:, -1]  # 取出标签
        self.fs = fs
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        """从 DataFrame 读取时间序列，进行 CWT 变换，并返回处理后的图像和标签"""
        signal = self.datas[index]  # 取出 1D 信号
        cwt_result = ContinuousWaveletTransform(self.fs, signal).cwt_results[0]  # 计算 CWT 结果，取 batch 里第一个
        image = torch.tensor(cwt_result, dtype=torch.float32)

        label = self.labels[index]
        label_onehot = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return image, label_onehot


class MatrixDataset(Dataset):
    """
    自定义 PyTorch 数据集类，用于存储 N*x*y 形状的矩阵数据及对应的标签。

    Args:
        data (numpy.ndarray | torch.Tensor): 形状为 (N, x, y) 的输入数据。
        labels (numpy.ndarray | torch.Tensor): 形状为 (N,) 的标签向量。
        transform (callable, optional): 可选的预处理转换函数，默认 None。

    Returns:
        dataset: (sample (torch.Tensor), label (torch.Tensor))
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform: transforms.Compose = None):
        self.data = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        self.labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple:
        sample = self.data[index]
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def get_data_loader(data_set: Dataset, BATCH_SIZE: int = 16, NUM_WORKERS: int = 0) -> List[DataLoader]:  # 我想这个函数对多种dataset都适用，所以我改成输入dataset而不是pd.DataFrame
    """
    用于获取数据集的 DataLoader。

    Args:
        data_set (_torch的Dataset_): 传入的数据集
    Returns:
        元组 (train_loader, eval_loader): 训练集和评估集的dataloader
    """

    lengths = [round(0.8 * len(data_set)), len(data_set) - round(0.8 * len(data_set))]
    train_data, eval_data = random_split(data_set, lengths)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    return train_loader, eval_loader


if __name__ == "__main__":
    sample = np.random.rand(20, 28, 28)
    label = np.random.randint(0, 10, 20)
    data_set = MatrixDataset(sample, label)
    train_loader, eval_loader, test_loader = get_data_loader(data_set)
    for i, (data, label) in enumerate(train_loader):
        print(data.shape, label.shape)
