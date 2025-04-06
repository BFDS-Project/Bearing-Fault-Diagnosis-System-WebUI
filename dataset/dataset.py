import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Literal


def get_dataset(data_set, subset, split):
    # TODO 换源
    ds = load_dataset(data_set, subset)
    return ds[split].to_pandas()


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
    def __init__(self, data_set, labels, transfer_task):
        self.data_set = data_set
        self.labels = labels
        self.source = transfer_task[0]
        self.target = transfer_task[1]

    def data_split(self, batch_size, num_workers, device, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            data_frame = get_dataset(self.data_set, self.source[0], self.source[1])
            data_set = SignalDataset(data_frame)
            lengths = [round(0.8 * len(data_set)), len(data_set) - round(0.8 * len(data_set))]
            train_data, eval_data = random_split(data_set, lengths)
            source_train = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"),drop_last=True)
            source_val = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"),drop_last=True)
            # get target train and val
            data_frame=get_dataset(self.data_set, self.target[0], self.target[1])
            data_set = SignalDataset(data_frame)
            lengths = [round(0.8 * len(data_set)), len(data_set) - round(0.8 * len(data_set))]
            train_data, eval_data = random_split(data_set, lengths)
            target_train = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"),drop_last=True)
            target_val = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"),drop_last=True)
            return source_train, source_val, target_train, target_val
        else:
            # get source train and val
            data_frame = get_dataset(self.data_set, self.source[0], self.source[1])
            data_set = SignalDataset(data_frame)
            lengths = [round(0.8 * len(data_set)), len(data_set) - round(0.8 * len(data_set))]
            train_data, eval_data = random_split(data_set, lengths)
            source_train = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
            source_val = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
            # get target val
            get_dataset(self.data_set, self.target[0], self.target[1])
            data_frame = get_dataset(self.data_set, self.target[0, 0], self.target[0, 1])
            data_set = SignalDataset(data_frame)
            target_val = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
            return source_train, source_val, target_val
