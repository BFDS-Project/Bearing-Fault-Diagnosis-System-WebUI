import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List
from PIL import Image

DATAPATH = "./make-data/Dataset/written-num/data"
LABELPATH = "./make-data/Dataset/written-num/label/label.txt"
SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2


def get_data_set(filepath: str) -> pd.DataFrame:
    data_set = pd.read_csv(filepath, header=None)
    return data_set


transform = transforms.Compose(
    [
        transforms.Resize([SIZE, SIZE]),
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class SignalDataset(Dataset):
    def __init__(self, data_set: pd.DataFrame):
        # 最后一列是标签
        self.datas = data_set.iloc[:, :-1]
        self.labels = data_set.iloc[:, -1]

    # 返回数据集大小
    def __len__(self):
        return self.datas.shape[0]

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        # # 需要将图片大小统一并变成tensor形式
        Image_data=Image.fromarray(np.array(self.datas.iloc[index,:]).reshape(SIZE,SIZE))
        data = transform(Image_data)#我查transform作用对象是PIL.Image.Image，所以这里需要先将图片转化为numpy.ndarray，然后再reshape为(SIZE,SIZE)
        label = self.labels[index]
        return data, label

import torch
from torch.utils.data import Dataset

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
    
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform:transforms.Compose=None):
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


def get_data_loader(data_set:Dataset) -> List[DataLoader]:#我想这个函数对多种dataset都适用，所以我改成输入dataset而不是pd.DataFrame
    """
    用于获取数据集的 DataLoader。
    Args:
        data_set(torchd的Dataset):传入的数据集
        
    Returns:
        train_loader(torch.utils.data.DataLoader) 训练集的 DataLoader
        
        eval_loader(torch.utils.data.DataLoader)  验证集的 DataLoader
        
        test_loader(torch.utils.data.DataLoader)  测试集的 DataLoader
    """
    data_trained = data_set
    lengths = [
        round(0.6 * len(data_trained)),
        round(0.2 * len(data_trained)),
        len(data_trained) - round(0.6 * len(data_trained)) - round(0.2 * len(data_trained)),
    ]

    train_data, eval_data, test_data = random_split(data_trained, lengths)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    return train_loader, eval_loader, test_loader


if __name__ == "__main__":
    sample=np.random.rand(20,28,28)
    label=np.random.randint(0,10,20)
    data_set=MatrixDataset(sample,label)
    train_loader, eval_loader, test_loader = get_data_loader(data_set)
    for i, (data, label) in enumerate(train_loader):
        print(data.shape, label.shape)  
