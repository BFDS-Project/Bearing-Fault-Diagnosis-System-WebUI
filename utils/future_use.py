# 暂时不使用的代码
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os


class ContinuousWaveletTransform:
    def __init__(self, fs, signals, save_path=None, wavelet="cmor1.5-1.0", freqNum=224):
        """
        连续小波变换 (CWT) 计算类，支持单个信号或批量信号输入，并保存为 .npy 文件。

        Args:
            fs (_int_): 采样频率
            signals (_np.array_): 输入信号，形状可以是 (signal_length,) 或 (batch_size, signal_length)
            save_path (_str_): 如果提供，将保存 CWT 变换后的数据到 .npy 文件，默认为None，为None时不保存
            wavelet (_str_): 连续小波类型（默认 'cmor1.5-1.0'）
            freqNum (_int_): 频率点个数（默认 224）
        """
        self.fs = fs
        self.save_path = save_path

        # 确保路径存在
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # 确保输入是 NumPy 数组
        signals = np.asarray(signals, dtype=np.float32)  # 使用 float32 节省内存

        # 处理 batch 维度
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]  # (signal_length,) -> (1, signal_length)

        self.batch_size, self.signal_length = signals.shape
        self.time = np.arange(0, self.signal_length) / fs  # 时间轴
        self.widths = np.geomspace(1, 512, num=freqNum).astype(np.float32)  # 频率尺度

        # 预分配 CWT 结果矩阵
        self.cwt_results = np.empty((self.batch_size, freqNum, self.signal_length), dtype=np.float32)

        for i in range(self.batch_size):
            signal = signals[i] - np.mean(signals[i])  # 去均值（去直流分量）
            cwtmatr, freqs = pywt.cwt(signal, self.widths, wavelet, sampling_period=1 / fs)
            cwt_result = np.abs(cwtmatr).astype(np.float32)  # 取模值，转换为 float32
            self.cwt_results[i] = cwt_result

            # 保存 CWT 结果到 .npy 文件
            if save_path:
                np.save(os.path.join(save_path, f"cwt_{i:04d}.npy"), cwt_result)
                print(f"CWT 结果已保存到 {os.path.join(save_path, f'cwt_{i:04d}.npy')}")

        self.freqs = freqs.astype(np.float32)  # 存储频率信息，节省内存

    def plot(self, index=0, logspace=True, save_path=None):
        """
        绘制并可选保存 CWT 结果。

        Args:
            index (_int_): 选择绘制的信号索引
            logspace (_bool_): 是否使用对数坐标绘制频率轴
            save_path (_str_ 或 None): 如果提供路径，则保存 `.npy` 文件，否则不保存
        """
        if index >= self.batch_size:
            raise ValueError(f"Index 超出范围！batch_size = {self.batch_size}, 但 index = {index}")

        # 获取 CWT 结果
        cwt_matrix = self.cwt_results[index]

        # 选择是否保存 .npy
        if save_path:
            np.save(save_path, cwt_matrix)
            print(f"CWT 结果已保存到 {save_path}")

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 5))
        pcm = ax.pcolormesh(self.time, self.freqs, cwt_matrix, shading="auto")

        ax.set_yscale("log" if logspace else "linear")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"CWT Scaleogram (Signal {index})")
        fig.colorbar(pcm, ax=ax)

        plt.show()  # 显示图像


if __name__ == "__main__":
    fs = 1e3
    N = 1e4
    noise_power = 1e-3 * fs
    time = np.arange(N) / float(fs)
    mod = 2 * np.pi * 20 * np.cos(time)
    carrier = np.sin(2 * np.pi * 100 * time + mod)  # 频率调制

    rng = np.random.default_rng()
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier + noise

    CWT = ContinuousWaveletTransform(fs, x)
    CWT.plot(0, logspace=False)

    # plt.plot(time, x)
    # plt.show()

# ================================================================
# %%
import torch
import torch.nn as nn
import pandas as pd
from dataset.get_dataset import SignalDatasetCreator
from pathlib import Path

data_set = "BFDS-Project/Bearing-Fault-Diagnosis-System"  # 数据集huggingface地址
labels = {"Normal Baseline Data": 0, "Ball": 1, "Inner Race": 2, "Outer Race Centered": 3, "Outer Race Opposite": 4, "Outer Race Orthogonal": 5}  # 标签
transfer_task = [["CWRU224", "12kDriveEnd"], ["CWRU224", "12kFanEnd"]]  # 迁移方向


signal_dataset_creator = SignalDatasetCreator(data_set, labels, transfer_task, stratified_sampling=True)
dataloaders = {}
dataloaders["source_train"], dataloaders["source_val"], dataloaders["target_train"], dataloaders["target_val"] = signal_dataset_creator.data_split(
    64, 0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# %%
import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = getattr(models, "ResNet")().to(device)
bottleneck_layer = nn.Sequential(
    nn.Linear(model.output_num(), 256),
    nn.ReLU(inplace=True),
    nn.Dropout(),
).to(device)
classifier_layer = nn.Linear(256, len(labels)).to(device)
model_all = nn.Sequential(model, bottleneck_layer, classifier_layer).to(device)
model_all.load_state_dict(torch.load("checkpoint/150_0/149-0.3942-best_model.bin"))  # 加载模型参数
model_without_head = nn.Sequential(*list(model_all.children())[:-1])


# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 定义一个固定的颜色映射
num_classes = len(set(label for dataloader in dataloaders.values() for _, labels in dataloader for label in labels.numpy()))
colors = plt.cm.get_cmap("tab10", num_classes)  # 使用 "tab10" 颜色映射
cmap = ListedColormap(colors.colors)


def plot_tsne(dataloader, title, ax):
    model_all.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_without_head(inputs)
            # Collect all points across all batches
            if i == 0:
                all_points = outputs.cpu().numpy()
                all_labels = labels.cpu().numpy()
            else:
                all_points = np.concatenate((all_points, outputs.cpu().numpy()), axis=0)
                all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=0)

        # Apply t-SNE to reduce dimensions to 2D
        tsne = TSNE(n_components=2, random_state=42)
        reduced_points = tsne.fit_transform(all_points)

        # Plot the reduced points
        scatter = ax.scatter(reduced_points[:, 0], reduced_points[:, 1], c=all_labels, cmap=cmap, s=10)
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        return scatter, reduced_points, all_labels


# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot each dataloader
# sc1,_ = plot_tsne(dataloaders["source_train"], "Source Train", axes[0, 0])
sc2, reduced_points2, all_labels2 = plot_tsne(dataloaders["source_val"], "Source Val", axes[0, 1])
# sc3,_ = plot_tsne(dataloaders["target_train"], "Target Train", axes[1, 0])
# sc4,_ = plot_tsne(dataloaders["target_val"], "Target Val", axes[1, 1])

# Add a colorbar to the figure
# cbar = fig.colorbar(sc1, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
# cbar.set_label("Labels")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# %%
reduced_points2, all_labels2

# %%
import pandas as pd

df = pd.DataFrame(reduced_points2)
df["label"] = all_labels2  # 将标签添加为新列

# 保存为 CSV 文件
df.to_csv("checkpoint/reduced_points_with_labels.csv", index=False)
