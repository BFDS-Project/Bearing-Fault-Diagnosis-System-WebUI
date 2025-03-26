from models.CNN import cnn_features
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from utils.args import Argument

args = Argument()

# 下载模型文件
model_file = hf_hub_download(repo_id="BFDS-Project/Bearing-Fault-Diagnosis-System-classifer", filename="model_CNN_demo.bin")
model = cnn_features()
bottleneck_layer = nn.Sequential(
    nn.Linear(model.output_num(), args.bottleneck_num),
    nn.ReLU(inplace=True),  # mark: 此处为节省内存，采用了inplace操作
    nn.Dropout(),
)
classifier_layer = nn.Linear(args.bottleneck_num, 10)
model_all = nn.Sequential(model, bottleneck_layer, classifier_layer)
model_all.load_state_dict(torch.load(model_file))

# 设置为评估模式
model_all.eval()
# 进行预测
with torch.no_grad():
    output = model_all(torch.randn(1, 1, 1024))

print(output.shape)
