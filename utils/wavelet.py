import numpy as np
import pywt
import matplotlib.pyplot as plt

class ContinuousWaveletTransform:
    def __init__(self, fs, signal, wavelet='cmor1.5-1.0', freqNum = 224):
        '''
        Args:
            fs(_int_): 信号频率
            signal(_np.array_): 原始信号
            wavelet(_str_): 连续小波种类
            freqNum(_int_): 频率域细分点个数
        '''
        signal = np.squeeze(np.array(signal))
        signal = signal - np.mean(signal) # 去直流分量
        
        widths = np.geomspace(1, 1024, num = freqNum)
        self.time = np.arange(0, len(signal))/fs

        self.cwtmatr, self.freqs = pywt.cwt(signal, widths, wavelet, sampling_period=1/fs)
        self.cwtmatr = np.abs(self.cwtmatr[:-1, :-1]) # cwt结果为复数，要取模

    def plot(self, logspace = True):
        fig, axs = plt.subplots(1, 1)
        pcm = axs.pcolormesh(self.time, self.freqs, self.cwtmatr)
        
        if logspace: # 频率域绘制选择线性or对数坐标
            axs.set_yscale("log")
        else:
            axs.set_yscale("linear")
        
        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Frequency (Hz)")
        axs.set_title("Continuous Wavelet Transform (Scaleogram)")
        fig.colorbar(pcm, ax=axs)
        plt.show()


if __name__ == "__main__":
    fs = 1e3
    N = 1e4
    noise_power = 1e-3 * fs
    time = np.arange(N) / float(fs)
    mod = 2 * np.pi * 20 * np.cos(time)
    carrier = np.sin(2*np.pi*100 * time + mod) # 频率调制

    rng = np.random.default_rng()
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + noise

    CWT = ContinuousWaveletTransform(fs, x)
    CWT.plot(logspace=False)
    
    # plt.plot(time, x)
    # plt.show()
