import torch
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=15),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )  # 128, 4,4

        self.layer5 = nn.Sequential(nn.Linear(128 * 4, 256), nn.ReLU(inplace=True), nn.Dropout())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        return x


# FIXME 看要不要东西
# convnet without the last layer
class cnn_features(nn.Module):
    def __init__(self, pretrained=False):
        super(cnn_features, self).__init__()
        self.model_cnn = CNN(pretrained)
        self.__in_features = 256

    def forward(self, x):
        x = self.model_cnn(x)
        return x

    def output_num(self):
        return self.__in_features


if __name__ == "__main__":
    model = cnn_features()
    input = torch.randn(1, 1, 224)
    out = model(input)
    print(out.shape)
