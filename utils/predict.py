import torch
import torch.nn as nn
import pandas as pd
import librosa
from pathlib import Path
import models


def audio_to_signal(audio_file, sr=None):
    signal, _ = librosa.load(audio_file, sr=sr)
    return signal


def csv_to_signal(signal_file):
    signal = pd.read_csv(signal_file).to_numpy().flatten()
    return signal


# 修改backbone
def predict(model_state_dict, signal_file, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, args.model_name)().to(device)
    bottleneck_layer = nn.Sequential(
        nn.Linear(model.output_num(), args.bottleneck_num),
        nn.ReLU(inplace=True),
        nn.Dropout(),
    ).to(device)
    classifier_layer = nn.Linear(args.bottleneck_num, len(args.labels)).to(device)
    model_all = nn.Sequential(model, bottleneck_layer, classifier_layer).to(device)
    model_all.load_state_dict(model_state_dict)
    # 模型预测
    model_all.eval()
    with torch.no_grad():
        # 根据文件后缀选择处理方式
        file_extension = Path(signal_file).suffix
        if file_extension == ".csv":
            signal = csv_to_signal(signal_file).reshape(-1, 1, 224)
        elif file_extension in [".wav", ".mp3"]:
            signal = audio_to_signal(signal_file).reshape(-1, 1, 224)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        signal = torch.tensor(signal, dtype=torch.float32).to(device)
        output = model_all(signal)
        predictions = output.mean(dim=0)
    return predictions
