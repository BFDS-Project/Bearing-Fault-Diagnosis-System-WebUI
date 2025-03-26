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
