import logging
import time
import warnings

import torch
from torch import nn
from torch import optim

import datasets
import models
import loss


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
            x: torch.utils.data.DataLoader(self.datasets[x], 
                                           batch_size   = args.batch_size,
                                           shuffle      = (x.split('_')[1] == 'train'),
                                           num_workers  = args.num_workers,
                                           pin_memory   = (self.device == 'cuda'),
                                           drop_last    = (args.last_batch and x.split('_')[1] == 'train')
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
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace = True), # mark: 此处为节省内存，采用了inplace操作
                                                  nn.Dropout()
                                                  )
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
            self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        else:
            # 若不启用bottleneck层，则仅包含最后一层
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)
            self.model_all = nn.Sequential(self.model, self.classifier_layer)

        # 定义领域对抗网络
        if args.domain_adversarial:
            # 引入对抗后的最大迭代轮数 = 源域全部数据 * (最大轮数 - 引入对抗的轮数)
            self.max_iter = len(self.dataloaders['source_train'])*(args.max_epoch-args.middle_epoch)

            # 若对抗损失为CDA或CDA+E，对抗网络将输入全部展开
            if args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                if args.bottleneck:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(
                                            in_feature              = args.bottleneck_num * Dataset.num_classes,
                                            hidden_size             = args.hidden_size,
                                            max_iter                = self.max_iter,
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

     
        # 加载模型
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)
        
        # 定义模型参数
        parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                          {"params": self.classifier_layer.parameters(), "lr": args.lr}]
        
        if args.bottleneck:
            parameter_list.append({"params": self.bottleneck_layer.parameters(), "lr": args.lr})
        
        if args.domain_adversaial:
            parameter_list.append({"params": self.AdversarialNet.parameters(), "lr": args.lr})


        # 定义优化器
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, 
                                       lr = args.lr,
                                       momentum = args.momentum,
                                       weight_decay = args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, 
                                        lr = args.lr,
                                        weight_decay = args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # 定义学习率调度器
        if args.lr_scheduler == 'step':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, args.steps, gamma = args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, args.steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")
                
        # 定义交叉损失（与对抗网络二选一）
        if args.distance_metric:
            if args.distance_loss == 'MK-MMD':
                self.distance_loss = loss.DAN
            elif args.distance_loss == "JMMD":
                # mark: 附加网络
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.distance_loss = loss.JAN
            elif args.distance_loss == "CORAL":
                self.distance_loss = loss.CORAL
            else:
                raise Exception("loss not implement")
        else:
            self.distance_loss = None

        # 定义对抗损失
        if args.domain_adversarial:
            if args.adversarial_loss == 'DA':
                self.adversarial_loss = nn.BCELoss()
            elif args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                # 附加网络
                self.softmax_layer_ad = nn.Softmax(dim=1)
                self.softmax_layer_ad = self.softmax_layer_ad.to(self.device)
                self.adversarial_loss = nn.BCELoss()
            else:
                raise Exception("loss not implement")
        else:
            self.adversarial_loss = None

        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        args = self.args
        
        batch_acc = 0
        batch_count = 0
        batch_loss = 0.0
        best_acc = 0.0
        
        iter_num = 0
        
        step = 0
        step_start = time.time()
        
        # 记录训练轮次与学习率
        for epoch in range(args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))
            
            step_target = 0
            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])
            
            # 每轮分为三个阶段：源域训练、源域测试、目标域测试
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0
                
                # 设置模型为训练/测试模式
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()
                
                # 遍历每个batch训练
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    # 只有源域训练轮次大于middle_epoch后才在训练过程中引入目标域数据训练
                    if phase != 'source_train' or epoch <= args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, _ = next(iter_target) # 无监督学习，目标域数据无标签
                        step_target += 1
                        
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    
                    # 若目标域训练数据已加载完，重新初始化迭代器
                    if step_target % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])
                    
                    
                    # 只有源域训练时需要计算梯度
                    with torch.set_grad_enabled(phase == 'source_train'):
                        # 前向传播
                        features = self.model(inputs)
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            # 未引入目标域数据训练，直接计算损失
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else: 
                            pass
        
        
    
    def plot(self):
        pass