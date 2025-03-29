import json


class Argument:
    """
    训练过程中全部超参数
    """

    def __init__(self):
        # 数据集
        self.data_dir = "E:/Data/CWRU_old/"  # 数据集路径
        self.data_name = "CWRU"  # 数据集名称

        # 预处理
        self.normalize_type = "mean-std"  # 归一化方式
        self.wavelet = "cmor1.5-1.0"  # 小波类型

        # 模型
        self.model_name = "CNN"  # 模型名
        self.bottleneck = True  # 是否使用bottleneck层
        self.bottleneck_num = 256  # bottleneck层的输出维数
        self.pretrained = False  # 是否使用预训练模型

        # 训练
        self.batch_size = 64  # 批次大小
        self.cuda_device = "0"  # 训练设备
        self.last_batch = False  # 是否保留最后的不完整批次
        self.max_epoch = 200  # 训练最大轮数
        self.num_workers = 0  # 训练设备数
        self.pretrained = False  # 是否加载预训练模型
        self.transfer_task = [[0], [1]]  # 迁移方向

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
        # 基于映射
        self.distance_option = False  # 是否采用基于映射的损失
        self.distance_loss = "JMMD"  # 损失模型 MK-MMD/JMMD/CORAL
        self.distance_tradeoff = "Step"  # 损失的trade_off参数 Cons/Step
        self.distance_lambda = 1  # 若调整模式为Cons，指定其具体值

        # 基于领域对抗
        self.adversarial_option = True  # 是否采用领域对抗
        self.adversarial_loss = "CDA"  # 领域对抗损失
        self.hidden_size = 1024  # 对抗网络的隐藏层维数
        self.middle_epoch = 100  # 引入对抗的起始轮次
        self.grl_option = "Step"  # 梯度反转层权重选择静态or动态更新
        self.grl_lambda = 1  # 梯度静态反转时梯度所乘的系数
        self.adversarial_tradeoff = "Step"  # 损失的trade_off参数 Cons/Step
        self.adversarial_lambda = 1  # 若调整模式为Cons，指定其具体值

    def update_param(self, param_name, param_value):
        if hasattr(self, param_name):
            setattr(self, param_name, param_value)
        else:
            raise AttributeError(f"Parameter '{param_name}' does not exist.")


def update_param(args, batch_size, optimizer, learning_rate, scheduler, transfer_method, distance_loss):
    if transfer_method not in ["基于映射", "基于领域对抗"]:
        return "错误: 迁移学习方式无效，请选择 '基于映射' 或 '基于领域对抗'。"
    args.update_param("batch_size", batch_size)
    args.update_param("opt", optimizer.lower())
    args.update_param("lr", learning_rate)
    args.update_param("lr_scheduler", scheduler)
    if transfer_method == "基于映射":
        args.update_param("adversarial_option", False)
        args.update_param("distance_option", True)
    elif transfer_method == "基于领域对抗":
        args.update_param("adversarial_option", True)
        args.update_param("distance_option", False)
    args.update_param("distance_loss", distance_loss)
    # 返回所有参数
    # FIXME __dict__
    all_params = {attr: getattr(args, attr) for attr in dir(args) if not attr.startswith("__") and not callable(getattr(args, attr))}
    return json.dumps(all_params, indent=2)


if __name__ == "__main__":
    args = Argument()
    print(args.data_dir)
    print(args.batch_size)
    args.update_param("data_dir", "E:/Data/CWRU_new/")
    args.update_param("batch_size", 128)
    print(args.data_dir)
    print(args.batch_size)
