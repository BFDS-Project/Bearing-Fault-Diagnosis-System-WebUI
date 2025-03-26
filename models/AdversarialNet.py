import torch
from torch import nn
import numpy as np


def calc_coeff(iter_num: int, high: float, low: float, alpha: float, max_iter: float) -> np.float64:
    """
    动态计算对抗训练中的权重系数以调整梯度反转的强度
    Args:
        iter_num(_int_): 迭代轮次
        high(_float_): 轮次较大时系数接近high
        low(_float_): 轮次较小时系数接近low
        alpha(_float_): 随轮次增加时系数的衰减参数
        max_iter(_float_): 最大轮次
    Returns:
        (_np.float64_): 权重系数
    """
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff: float):
    """
    实现GRL(Gradient Reverse Layer 梯度反转层)
    Args:
        coeff(_float_): 梯度反转系数
    Returns:
        (_function_): 输入梯度值, 返回反转后乘coeff所得值
    """

    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def Entropy(input_prob: torch.tensor) -> torch.tensor:
    """
    计算输入样本的熵
    Args:
        input(_torch.tensor_): 输入概率分布, 形状为 (batch_size, num_classes)
    Returns:
        entropy(_torch.tensor_): 输出熵值, 长度为 batch_size
    """
    input_prob.size(0)
    epsilon = 1e-5
    entropy = -input_prob * torch.log(input_prob + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class AdversarialNet(nn.Module):
    """
    对抗网络, 用于减小域间差异
    Args:
        in_feature(_int_): 输入特征维数
        hidden_size(_int_): 隐藏层维数
        grl_option(_float_): 梯度反转层权重选择静态or动态更新
        grl_lambda(_float_): 静态反转时梯度所乘的系数
        high(_float_): 轮次较大时系数接近high
        low(_float_): 轮次较小时系数接近low
        alpha(_float_): 随轮次增加时系数的衰减参数
        max_iter(_float_): 最大迭代轮数, 影响梯度衰减因子
    """

    def __init__(
        self,
        in_feature: int,
        hidden_size: int,
        grl_option: str = "Step",
        grl_lambda: float = 1.0,
        high: float = 1.0,
        low: float = 0.0,
        alpha: float = 10.0,
        max_iter: float = 10000.0,
    ):
        super().__init__()  # 继承nn.Module的参数
        self.ad_layer1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),  # mark: 此处为节省内存，采用了inplace操作
            nn.Dropout(),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # 参数
        self.high = high
        self.low = low
        self.alpha = alpha
        self.max_iter = max_iter
        self.grl_option = grl_option
        self.grl_lambda = grl_lambda
        self.iter_num = 0  # 当前迭代轮数
        self.__in_features = 1  # 输出特征维数（伪私有化变量）

    def forward(self, x):
        # 训练模式下，迭代轮次+1
        if self.training:
            self.iter_num += 1

        # 计算对抗损失权重，可选动态更新还是保持常数
        if self.grl_option == "Cons":
            coeff = self.grl_lambda
        elif self.grl_option == "Step":
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        else:
            raise Exception("loss not implement")

        # 前向传播
        x = x * 1.0
        x.register_hook(grl_hook(coeff))  # 反转对抗层之前的梯度，以保证最大化领域对抗损失
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return self.__in_features
