import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    expansion = 1  # 扩展倍数，用于调整输出通道数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # 激活函数
        # 第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample  # 下采样模块，用于匹配维度

    def forward(self, x):
        identity = x  # 残差连接的输入
        if self.downsample is not None:
            identity = self.downsample(x)  # 如果需要下采样，调整维度

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, block=BasicBlock1D, layers=[2, 2, 2, 2]):
        super(ResNet1D, self).__init__()
        self.in_channels = 64  # 初始输入通道数
        self.__in_features = 256  # 固定输出特征维度
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 自适应池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, self.__in_features)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 如果需要调整通道数或步幅不为1，则定义下采样层
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        # 第一个残差块
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # 后续残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入经过初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 经过残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化和全连接层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def output_num(self):
        # 返回输出特征维度
        return self.__in_features


def resnet1d18():
    # 构建 ResNet1D-18 模型
    return ResNet1D(layers=[2, 2, 2, 2])


if __name__ == "__main__":
    model = resnet1d18()
    print(model)
    input_tensor = torch.randn(8, 1, 1024)
    output = model(input_tensor)
    print("Output shape:", output.shape)
