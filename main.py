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
        # 模型和数据集参数
        self.data_dir       = 'D:/Data/CWRU/'   # 数据集路径
        self.data_name      = 'CWRU'            # 数据集名称
        self.model_name     = 'ResNet12'        # 模型名
        self.normalize_type = 'mean-std'        # 归一化方式
        self.transfer_task  = [0, 1]            # 迁移方向
        
        # 预处理参数
        self.wavelet        = 'cmor1.5-1.0'     # 小波类型
        
        # 训练参数
        self.batch_size     = 64                # 批次大小
        self.cuda_device    = '0'               # 训练设备
        self.checkpoint_dir = './checkpoint'    # 模型保存路径
        self.last_batch     = False             # 是否保留最后的不完整批次


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
