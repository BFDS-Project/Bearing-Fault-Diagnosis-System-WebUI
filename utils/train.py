import logging
import warnings

import torch

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
    
    def train(self):
        pass
    
    def plot(self):
        pass