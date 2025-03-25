import os
import logging
import warnings
from datetime import datetime

from utils.logger import setlogger
from utils.train import train_utils


class Argument:
    '''
    训练过程中全部超参数
    '''
    def __init__(self):
        # 模型和数据集
        self.bottleneck     = True              # 是否使用bottleneck层
        self.bottleneck_num = 256               # bottleneck层的输出维数
        self.data_dir       = 'D:/Data/CWRU/'   # 数据集路径
        self.data_name      = 'CWRU'            # 数据集名称
        self.model_name     = 'ResNet12'        # 模型名
        self.normalize_type = 'mean-std'        # 归一化方式
        self.transfer_task  = [0, 1]            # 迁移方向
        
        # 预处理
        self.wavelet        = 'cmor1.5-1.0'     # 小波类型
        
        # 训练
        self.batch_size     = 64                # 批次大小
        self.checkpoint_dir = './checkpoint'    # 参数保存路径
        self.cuda_device    = '0'               # 训练设备
        self.last_batch     = False             # 是否保留最后的不完整批次
        self.max_epoch      = 300               # 训练最大轮数
        self.pretrained     = False             # 是否加载预训练模型
        
        # 优化器
        self.momentum       = 0.9               # sgd优化器动量参数
        self.opt            = 'adam'            # 优化器
        self.weight_decay   = 1e-5              # 优化器权重衰减
        
        # 学习率调度器
        self.gamma          = 0.1               # 学习率调度器参数
        self.lr             = 1e-3              # 初始学习率
        self.lr_scheduler   = 'step'            # 学习率调度器
        self.steps          = [150, 250]        # 学习率衰减轮次

        # 迁移学习
        # 基于映射
        self.distance_loss          = 'JMMD'    # 损失模型
        self.distance_metric        = False     # 是否采用基于映射的损失
        self.trade_off_distance     = 'Step'    # 基于映射的损失的trade_off参数 Cons/Step
        self.lam_distance           = 1         # 若调整模式为Cons，指定其具体值
        
        # 基于领域对抗
        self.adversarial_loss       = 'CDA'     # 领域对抗损失
        self.domain_adversarial     = True      # 是否采用领域对抗
        self.hidden_size            = 1024      # 对抗网络的隐藏层维数
        self.lam_adversarial        = 1         # 梯度静态反转时梯度所乘的系数
        self.middle_epoch           = 100       # 引入对抗的起始轮次
        self.trade_off_adversarial  = 'Step'    # 梯度反转层权重选择静态or动态更新
        
         


if __name__ == '__main__':
    args = Argument()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    warnings.filterwarnings('ignore')

    save_dir = os.path.join(args.checkpoint_dir, 
                            args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S'))
    setattr(args, 'save_dir', save_dir)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 设定日志
    setlogger(os.path.join(args.save_dir, 'train.log'))

    # 保存超参数
    for k, v in args.__dict__.items():
        if k[-3:] != 'dir':
            logging.info("{}: {}".format(k, v))

    # 训练
    trainer = train_utils(args)
    trainer.setup()
    trainer.train()
    trainer.plot()
