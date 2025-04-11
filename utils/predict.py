import torch
import torch.nn as nn
import models
from dataset.get_dataset import get_user_dataset
import torch.nn.functional as F


def predict(model_state_dict, signal_file, args):
    signal = get_user_dataset(signal_file)
    signal = torch.tensor(signal.to_numpy(), dtype=torch.float32).unsqueeze(1)
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
    model_all.eval()
    with torch.no_grad():
        signal = signal.to(device)
        output = model_all(signal)
        output = F.softmax(output, dim=1)
        predictions = output.mean(dim=0).cpu()
        predictions = predictions.numpy()
        predictions = predictions / predictions.sum()
        predictions = [round(value, 3) for value in predictions]
        diff = 1.0 - sum(predictions)
        predictions[predictions.index(max(predictions))] += diff
    return predictions
