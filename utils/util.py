import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List

DATAPATH = "./make-data/Dataset/written-num/data"
LABELPATH = "./make-data/Dataset/written-num/label/label.txt"
SIZE = 224
BATCH_SIZE = 24
NUM_WORKERS = 2


def get_data_set(filepath: str) -> pd.DataFrame:
    """
    This function reads a csv file and returns a pandas dataframe.
    """
    pass


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
        data = transform(self.datas[index][0])
        label = self.labels[index]
        return data, label


def get_data_loader(data_set: pd.DataFrame) -> List[DataLoader]:
    data_trained = SignalDataset()
    lengths = [
        round(0.6 * len(data_trained)),
        round(0.2 * len(data_trained)),
        len(data_trained) - round(0.8 * len(data_trained)) - round(0.2 * len(data_trained)),
    ]

    train_data, eval_data, test_data = random_split(data_trained, lengths)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    return train_loader, eval_loader, test_loader


if __name__ == "__main__":
    pass
