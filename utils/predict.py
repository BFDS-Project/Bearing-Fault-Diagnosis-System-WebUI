from models.CNN import cnn_features
import torch
import torch.nn as nn

# from huggingface_hub import hf_hub_download
from utils.args import Argument

# FIXME 这里的 Argument 类看是直接导入还是后期需要main传进来
args = Argument()


def predict(model_state_dict, signal):
    model = cnn_features()
    bottleneck_layer = nn.Sequential(
        nn.Linear(model.output_num(), args.bottleneck_num),
        nn.ReLU(inplace=True),
        nn.Dropout(),
    )
    classifier_layer = nn.Linear(args.bottleneck_num, 10)
    model_all = nn.Sequential(model, bottleneck_layer, classifier_layer)
    model_all.load_state_dict(model_state_dict)

    # 设置为评估模式
    model_all.eval()
    # 进行预测
    with torch.no_grad():
        # FIXME signal的处理还没有写
        output = model_all(torch.randn(10, 1, 1024))
    return torch.argmax(output, dim=1)
