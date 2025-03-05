import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        # 第一个1x1卷积层，用于通道变换
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个3x3卷积层，可能进行下采样
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第三个1x1卷积层，恢复通道数
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # 当输入和输出维度不一致时，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet12(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet12, self).__init__()
        self.in_planes = 64  # 初始通道数

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 四个残差块层
        self.layer1 = self._make_layer(64, stride=1)  # 第1层，不降采样
        self.layer2 = self._make_layer(128, stride=2)  # 第2层，降采样
        self.layer3 = self._make_layer(256, stride=2)  # 第3层，降采样
        self.layer4 = self._make_layer(512, stride=2)  # 第4层，降采样

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, stride):
        # 每个残差块层包含一个Block
        layers = []
        layers.append(Block(self.in_planes, planes, stride))
        self.in_planes = planes  # 更新输入通道数为当前层的输出通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet12(num_classes=5).to(device)
    summary(model, (1, 224, 224))
