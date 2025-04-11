import pandas as pd
import numpy as np
from datasets import load_dataset
import librosa
import mimetypes

# ===============================================================
# 加载有标签的数据集(n , m + 1)最后一列是标签


def get_huggingface_dataset(data_set, subset, split):
    """
    Loads a dataset from the Hugging Face Hub, converts the specified split to a pandas DataFrame, and returns it.
    Args:
        data_set (str): The name of the dataset to load from the Hugging Face Hub.
        subset (str): The subset or configuration of the dataset to load.
        split (str): The split of the dataset to retrieve (e.g., 'train', 'test', 'validation').
    Returns:
        pandas.DataFrame: The specified split of the dataset converted to a pandas DataFrame.
    Example:
        >>> df = get_huggingface_dataset("imdb", "plain_text", "train")
        >>> print(df.head())
    """

    ds = load_dataset(data_set, subset)
    return ds[split].to_pandas()


def get_local_dataset(data_path):
    """
    读取本地 CSV 文件并将其内容作为 pandas DataFrame 返回。
    此函数主要用于调试目的以读取本地文件。
    通常情况下，数据集通常从 Hugging Face 或其他外部来源加载。
    参数:
        data_path (str): 本地 CSV 文件的文件路径。
    返回:
        pandas.DataFrame: CSV 文件内容的 DataFrame 表示。
    """
    df = pd.read_csv(data_path)
    return df


# ===============================================================
# 加载无标签的数据集(n , m)作为目标域,首先读取文件在转换为numpy一维数组在传入核心函数


def audio_to_signal(audio_file, sr=None):
    signal, _ = librosa.load(audio_file, sr=sr)
    signal = signal.flatten()
    return signal


def csv_to_signal(signal_file):
    signal = pd.read_csv(signal_file).to_numpy().flatten()
    return signal


def txt_to_signal(signal_file):
    signal = pd.read_csv(signal_file, sep="\t").to_numpy().flatten()
    return signal


def xlsx_to_signal(signal_file):
    signal = pd.read_excel(signal_file).to_numpy().flatten()
    return signal


def get_user_dataset(data_path, target_length=224):
    mime_type, _ = mimetypes.guess_type(data_path)
    supported_types = {
        "text/csv": "CSV 文件",
        "text/plain": "TXT 文件",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel 文件 (.xlsx)",
        "application/vnd.ms-excel": "旧版 Excel 文件 (.xls)",
        "audio/": "音频文件 (如 WAV, MP3, FLAC 等)",
    }

    if mime_type in ["text/csv", "application/vnd.ms-excel"]:
        signal = csv_to_signal(data_path)
    elif mime_type == "text/plain":
        signal = txt_to_signal(data_path)
    elif mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        signal = xlsx_to_signal(data_path)
    elif mime_type and mime_type.startswith("audio/"):
        signal = audio_to_signal(data_path)
    else:
        supported_list = ", ".join([f"{key} ({value})" for key, value in supported_types.items()])
        raise ValueError(f"不支持的文件类型: {mime_type}。支持的文件类型包括: {supported_list}")
    padding_size = 0 if signal.size % target_length == 0 else target_length - (signal.size % target_length)
    signal = np.pad(signal, (0, padding_size), mode="constant", constant_values=0)
    reshaped_data = signal.reshape(-1, target_length)
    zero_column = np.zeros((reshaped_data.shape[0], 1))
    new_data = np.hstack((reshaped_data, zero_column))
    reshaped_df = pd.DataFrame(new_data, columns=[f"col_{i}" for i in range(target_length + 1)])
    return reshaped_df


if __name__ == "__main__":
    data_path = r"C:\Users\Administrator\Desktop\demo.csv"
    df = get_user_dataset(data_path)
    print(df.head())
    print(df.shape)
