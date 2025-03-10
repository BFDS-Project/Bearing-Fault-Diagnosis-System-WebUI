import logging
import warnings

import torch
from torch import nn

import models
import datasets

class train_utils:
    def __init__(self, args):
        self.args = args
    
    def setup(self):
        args = self.args

        # 判断训练设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
        
        
        # 加载数据集
        # TODO: datasets尚未完成，编写时注意调用格式一致
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        self.datasets['source_train'], 
        self.datasets['source_val'], 
        self.datasets['target_train'], 
        self.datasets['target_val'] = Dataset(args.data_dir,
                                              args.transfer_task,
                                              args.normlizetype).data_split(transfer_learning=True)

        self.dataloaders = {
            x:  torch.utils.data.DataLoader(self.datasets[x], 
                                            batch_size  = args.batch_size,
                                            shuffle     = (x.split('_')[1] == 'train'),
                                            num_workers = args.num_workers,
                                            pin_memory  = (self.device == 'cuda'),
                                            drop_last   = (args.last_batch and x.split('_')[1] == 'train')
                                            )
            for x in ['source_train', 'source_val', 'target_train', 'target_val']
            }
        
        
        # 定义模型
        # TODO: models尚未完成，编写时注意考虑预训练模型加载
        self.model = getattr(models, args.model_name)(pretrained = args.pretrained)
        if args.bottleneck:
            # bottleneck层由线性层、ReLU层与Dropout层组成
            # 分类层为bottleneck的最后一层到输出的线性层
            # TODO: 注意Datasets要有分类的类数属性num_classes
            self.classifier_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), 
                                                  nn.Dropout(),
                                                  nn.Linear(args.bottleneck_num, Dataset.num_classes))
        else:
            # 若不启用bottleneck层，则仅包含最后一层
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        # 模型实际上为原模型加入bottleneck层与输出层的结构
        self.model_all = nn.Sequential(self.model, self.classifier_layer)

        # 定义领域对抗网络
        if args.domain_adversarial:
            # 引入对抗后的最大迭代轮数 = 源域全部数据 * (最大轮数 - 引入对抗的轮数)
            self.max_iter = len(self.dataloaders['source_train'])*(args.max_epoch-args.middle_epoch)

            # 若对抗损失为CDA或CDA+E，需引入对抗网络
            # TODO: 对抗网络未完成
            if args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                if args.bottleneck:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(
                                            in_feature              = args.bottleneck_num * Dataset.num_classes,
                                            hidden_size             = args.hidden_size, max_iter=self.max_iter,
                                            trade_off_adversarial   = args.trade_off_adversarial,
                                            lam_adversarial         = args.lam_adversarial
                                            )
                else:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(
                                            in_feature              = self.model.output_num()*Dataset.num_classes,
                                            hidden_size             = args.hidden_size, max_iter=self.max_iter,
                                            trade_off_adversarial   = args.trade_off_adversarial,
                                            lam_adversarial         = args.lam_adversarial
                                            )
            else:
                if args.bottleneck_num:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(
                                            in_feature              = args.bottleneck_num,
                                            hidden_size             = args.hidden_size, max_iter=self.max_iter,
                                            trade_off_adversarial   = args.trade_off_adversarial,
                                            lam_adversarial         = args.lam_adversarial
                                            )
                else:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(
                                            in_feature              = self.model.output_num(),
                                            hidden_size             = args.hidden_size, max_iter=self.max_iter,
                                            trade_off_adversarial   = args.trade_off_adversarial,
                                            lam_adversarial         = args.lam_adversarial
                                            )

    def train(self):
        pass
    
    def plot(self):
        pass