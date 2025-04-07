import os
import logging
import warnings
import json
from datetime import datetime
import requests

if __name__ == "__main__":
    try:
        # 这里尝试连接hugging face连接不上就换国内镜像源
        response = requests.get("https://huggingface.co", timeout=5)
        if response.status_code == 200:
            print("成功连接到 Hugging Face")
        else:
            print(f"连接失败，状态码: {response.status_code}")
    except requests.exceptions.RequestException:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"无法连接到 Hugging Face:换源到{os.environ['HF_ENDPOINT']}")

from utils.logger import setlogger
from utils.train import train_utils
from utils.fetch_conditions import fetch_all_conditions_from_huggingface


class Argument:
    """
    训练过程中全部超参数
    """

    def __init__(self):
        # 数据集
        self.data_set = "BFDS-Project/Bearing-Fault-Diagnosis-System"  # 数据集huggingface地址
        self.conditions = fetch_all_conditions_from_huggingface(self.data_set)  # 数据集的配置和分割信息如果想要知道明确的信息来确定迁移方向请自行运行fetch_conditions.py
        self.labels = {"Normal Baseline Data": 0, "Ball": 1, "Inner Race": 2, "Outer Race Centered": 3, "Outer Race Opposite": 4, "Outer Race Orthogonal": 5}  # 标签
        self.transfer_task = [["CWRU", "CWRU_12k_Drive_End_Bearing_Fault_Data"], ["CWRU", "CWRU_12k_Fan_End_Bearing_Fault_Data"]]  # 迁移方向

        # 预处理
        self.normalize_type = None  # 归一化方式, mean-std/min-max/None

        # 模型
        self.model_name = "ResNet_1d"  # 模型名
        self.bottleneck = True  # 是否使用bottleneck层
        self.bottleneck_num = 256  # bottleneck层的输出维数

        # 训练
        self.batch_size = 64  # 批次大小
        self.cuda_device = "0"  # 训练设备
        self.max_epoch = 2  # 训练最大轮数
        self.num_workers = 0  # 训练设备数

        # 数据记录
        self.checkpoint_dir = "./checkpoint"  # 参数保存路径
        self.print_step = 50  # 参数打印间隔

        # 优化器
        self.opt = "adam"  # 优化器 sgd/adam
        self.momentum = 0.9  # sgd优化器动量参数
        self.weight_decay = 1e-5  # 优化器权重衰减

        # 学习率调度器
        self.lr = 1e-3  # 初始学习率
        self.lr_scheduler = "step"  # 学习率调度器 step/exp/stepLR/fix
        self.gamma = 0.1  # 学习率调度器参数
        self.steps = [150, 250]  # 学习率衰减轮次

        # 迁移学习参数
        self.middle_epoch = 5  # 引入目标域数据的起始轮次

        # 基于映射
        self.distance_option = True  # 是否采用基于映射的损失
        self.distance_loss = "MK-MMD"  # 损失模型 MK-MMD/JMMD/CORAL
        self.distance_tradeoff = "Step"  # 损失的trade_off参数 Cons/Step
        self.distance_lambda = 1  # 若调整模式为Cons，指定其具体值

        # 基于领域对抗
        self.adversarial_option = False  # 是否采用领域对抗
        self.adversarial_loss = "CDA"  # 领域对抗损失
        self.hidden_size = 1024  # 对抗网络的隐藏层维数
        self.grl_option = "Step"  # 梯度反转层权重选择静态or动态更新
        self.grl_lambda = 1  # 梯度静态反转时梯度所乘的系数
        self.adversarial_tradeoff = "Step"  # 损失的trade_off参数 Cons/Step
        self.adversarial_lambda = 1  # 若调整模式为Cons，指定其具体值

        # 输出可视化
        self.wavelet = "cmor1.5-1.0"  # 小波类型

    def update_params(self, **kwargs):
        """
        使用 **kwargs 动态更新 args 的参数。
        """
        for param_name, param_value in kwargs.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
            else:
                print(f"警告: Parameter '{param_name}' does not exist.")

    def set_recommended_params(self):
        # 给用户设定的推荐参数
        recommended_params = {
            "data_set": "BFDS-Project/Bearing-Fault-Diagnosis-System",
            "conditions": fetch_all_conditions_from_huggingface("BFDS-Project/Bearing-Fault-Diagnosis-System"),
            "labels": {"Normal Baseline Data": 0, "Ball": 1, "Inner Race": 2, "Outer Race Centered": 3, "Outer Race Opposite": 4, "Outer Race Orthogonal": 5},
            "transfer_task": [["CWRU", "CWRU_12k_Drive_End_Bearing_Fault_Data"], ["CWRU", "CWRU_12k_Fan_End_Bearing_Fault_Data"]],
            "normalize_type": None,
            "model_name": "CNN",
            "bottleneck": True,
            "bottleneck_num": 256,
            "batch_size": 64,
            "cuda_device": "0",
            "max_epoch": 2,
            "num_workers": 0,
            "checkpoint_dir": "./checkpoint",
            "print_step": 50,
            "opt": "adam",
            "momentum": 0.9,
            "weight_decay": 1e-5,
            "lr": 1e-3,
            "lr_scheduler": "step",
            "gamma": 0.1,
            "steps": [150, 250],
            "middle_epoch": 0,
            "distance_option": True,
            "distance_loss": "JMMD",
            "distance_tradeoff": "Step",
            "distance_lambda": 1,
            "adversarial_option": False,
            "adversarial_loss": "CDA",
            "hidden_size": 1024,
            "grl_option": "Step",
            "grl_lambda": 1,
            "adversarial_tradeoff": "Step",
            "adversarial_lambda": 1,
            "wavelet": "cmor1.5-1.0",
        }
        self.update_params(**recommended_params)


if __name__ == "__main__":
    args = Argument()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()
    warnings.filterwarnings("ignore")

    save_dir = os.path.join(args.checkpoint_dir, args.model_name + "_" + datetime.strftime(datetime.now(), "%m%d-%H%M%S"))
    setattr(args, "save_dir", save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 设定日志
    setlogger(os.path.join(args.save_dir, "train.log"))

    # 保存超参数
    for k, v in args.__dict__.items():
        if k[-3:] != "dir":
            logging.info(f"{k}: {v}")

    # 训练
    trainer = train_utils(args)
    trainer.setup()
    trainer.train()
    trainer.plot()
