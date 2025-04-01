import logging
import math
import time
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

import models
from models.AdversarialNet import AdversarialNet, calc_coeff, grl_hook, Entropy
from dataset.dataset import SignalDatasetCreator
from .loss import DAN, JAN, CORAL


class train_utils:
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        self.save_dir = os.path.join(args.checkpoint_dir, args.model_name + "_" + datetime.strftime(datetime.now(), "%m%d-%H%M%S"))

        # 判断训练设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info(f"using {self.device_count} gpus")
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info(f"using {self.device_count} cpu")

        # 加载数据集
        signal_dataset_creator = SignalDatasetCreator(args.data_set, args.labels, args.transfer_task)
        self.dataloaders = {}
        self.dataloaders["source_train"], self.dataloaders["source_val"], self.dataloaders["target_train"], self.dataloaders["target_val"] = signal_dataset_creator.data_split(
            args.batch_size, args.num_workers, self.device, transfer_learning=True
        )
        # 定义模型
        self.model = getattr(models, args.model_name)()
        if args.bottleneck:
            # bottleneck层由线性层、ReLU层与Dropout层组成
            # 分类层为bottleneck的最后一层到输出的线性层
            self.bottleneck_layer = nn.Sequential(
                nn.Linear(self.model.output_num(), args.bottleneck_num),
                nn.ReLU(inplace=True),  # mark: 此处为节省内存，采用了inplace操作
                nn.Dropout(),
            )
            self.classifier_layer = nn.Linear(args.bottleneck_num, len(args.labels))
            self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        else:
            # 若不启用bottleneck层，则仅包含最后一层
            self.classifier_layer = nn.Linear(self.model.output_num(), len(args.labels))
            self.model_all = nn.Sequential(self.model, self.classifier_layer)

        # 定义领域对抗网络
        if args.adversarial_option:
            # 引入对抗后的最大迭代轮数 = 源域全部数据 * (最大轮数 - 引入对抗的轮数)
            self.max_iter = len(self.dataloaders["source_train"]) * (args.max_epoch - args.middle_epoch)

            # 若对抗损失为CDA或CDA+E，对抗网络将输入全部展开
            if args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                if args.bottleneck:
                    self.AdversarialNet = AdversarialNet(
                        in_feature=args.bottleneck_num * len(args.labels),
                        hidden_size=args.hidden_size,
                        max_iter=self.max_iter,
                        grl_option=args.grl_option,
                        grl_lambda=args.grl_lambda,
                    )
                else:
                    self.AdversarialNet = AdversarialNet(
                        in_feature=self.model.output_num() * len(args.labels),
                        hidden_size=args.hidden_size,
                        max_iter=self.max_iter,
                        grl_option=args.grl_option,
                        grl_lambda=args.grl_lambda,
                    )
            else:
                if args.bottleneck_num:
                    self.AdversarialNet = AdversarialNet(
                        in_feature=args.bottleneck_num, hidden_size=args.hidden_size, max_iter=self.max_iter, grl_option=args.grl_option, grl_lambda=args.grl_lambda
                    )
                else:
                    self.AdversarialNet = AdversarialNet(
                        in_feature=self.model.output_num(), hidden_size=args.hidden_size, max_iter=self.max_iter, grl_option=args.grl_option, grl_lambda=args.grl_lambda
                    )

        # 加载模型
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.adversarial_option:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)

        # 定义模型参数
        parameter_list = [{"params": self.model.parameters(), "lr": args.lr}, {"params": self.classifier_layer.parameters(), "lr": args.lr}]

        if args.bottleneck:
            parameter_list.append({"params": self.bottleneck_layer.parameters(), "lr": args.lr})

        if args.adversarial_option:
            parameter_list.append({"params": self.AdversarialNet.parameters(), "lr": args.lr})

        # 定义优化器
        if args.opt == "sgd":
            self.optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == "adam":
            self.optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # 定义学习率调度器
        if args.lr_scheduler == "step":
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, args.steps, gamma=args.gamma)
        elif args.lr_scheduler == "exp":
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == "stepLR":
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, args.steps, args.gamma)
        elif args.lr_scheduler == "fix":
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # 定义基于映射的损失（与对抗方法二选一）
        if args.distance_option:
            if args.distance_loss == "MK-MMD":
                self.distance_loss = DAN
            elif args.distance_loss == "JMMD":
                # mark: 附加网络
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.distance_loss = JAN
            elif args.distance_loss == "CORAL":
                self.distance_loss = CORAL
            else:
                raise Exception("loss not implement")
        else:
            self.distance_loss = None

        # 定义基于对抗的损失
        if args.adversarial_option:
            if args.adversarial_loss == "DA":
                # 领域对抗方法采用二元交叉熵损失
                self.adversarial_loss = nn.BCELoss()
            elif args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                # 条件领域对抗方法计算原输出与经过softmax输出后并在一起的损失，需引入附加网络
                self.softmax_layer_ad = nn.Softmax(dim=1)
                self.softmax_layer_ad = self.softmax_layer_ad.to(self.device)
                self.adversarial_loss = nn.BCELoss()
            else:
                raise Exception("loss not implement")
        else:
            self.adversarial_loss = None

        self.criterion = nn.CrossEntropyLoss()

        # 画图记录
        self.acc = {"source_train": np.array([]), "source_val": np.array([]), "target_val": np.array([])}

        self.loss = {"source_train": np.array([]), "source_val": np.array([]), "target_val": np.array([])}

    def train(self):
        args = self.args

        batch_acc = 0
        batch_count = 0
        batch_loss = 0.0
        best_acc = 0.0
        current_sample_count = 0

        batch_step = 0
        step_start = time.time()

        # 引入目标域数据训练，初始化计数器
        step_target = 0
        iter_target = iter(self.dataloaders["target_train"])
        len_target_loader = len(self.dataloaders["target_train"])

        # 熵方法的计数器
        iter_num_entropy = 0

        for epoch in range(args.max_epoch):
            # 记录训练轮次与学习率
            logging.info(f"{'-' * 5}Epoch {epoch}/{args.max_epoch - 1}{'-' * 5}")
            if self.lr_scheduler is not None:
                logging.info(f"current lr: {self.lr_scheduler.get_lr()}")
            else:
                logging.info(f"current lr: {args.lr}")

            # 每轮分为三个阶段：源域训练、源域测试、目标域测试
            for phase in ["source_train", "source_val", "target_val"]:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # 设置模型为训练/测试模式
                if phase == "source_train":
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.adversarial_option:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.adversarial_option:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()

                # 遍历每个batch训练
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    # 只有源域训练轮次大于middle_epoch后才在训练过程中引入目标域数据训练
                    if phase != "source_train" or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, _ = next(iter_target)  # 无监督学习，目标域数据无标签
                        step_target += 1
                        # 若目标域训练数据已加载完，重新初始化迭代器
                        if step_target % len_target_loader == 0:
                            iter_target = iter(self.dataloaders["target_train"])

                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    # 只有源域训练时需要计算梯度
                    with torch.set_grad_enabled(phase == "source_train"):
                        # 前向传播
                        features = self.model(inputs)
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)

                        if phase != "source_train" or epoch < args.middle_epoch:
                            # 未引入目标域数据训练，直接计算损失
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else:
                            # 若引入了目标域数据训练，要先提取出源域的前向传播结果，再计算源域损失
                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)

                            # 计算基于映射的损失
                            if self.distance_loss is not None:
                                if args.distance_loss == "MK-MMD":
                                    distance_loss = self.distance_loss(features.narrow(0, 0, labels.size(0)), features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)))
                                elif args.distance_loss == "JMMD":
                                    # JMMD实现多个特征层上的同时对齐，这里采用原输出与经过softmax后输出的两个特征层
                                    softmax_out = self.softmax_layer(outputs)
                                    distance_loss = self.distance_loss(
                                        [features.narrow(0, 0, labels.size(0)), softmax_out.narrow(0, 0, labels.size(0))],
                                        [
                                            features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)),
                                            softmax_out.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)),
                                        ],
                                    )
                                elif args.distance_loss == "CORAL":
                                    distance_loss = self.distance_loss(outputs.narrow(0, 0, labels.size(0)), outputs.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)))
                                else:
                                    raise Exception("loss not implement")
                            else:
                                distance_loss = 0

                            # 计算基于领域对抗的损失
                            if self.adversarial_loss is not None:
                                if args.adversarial_loss == "DA":
                                    # 领域对抗 (Domain Adversarial) 网络损失
                                    # 设数据来自源域 -> 标签为0; 来自目标域 -> 标签为1
                                    domain_label_source = torch.zeros(labels.size(0)).float()
                                    domain_label_target = torch.ones(inputs.size(0) - labels.size(0)).float()

                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_out = self.AdversarialNet(features)
                                    adversarial_loss = self.adversarial_loss(adversarial_out.squeeze(), adversarial_label)

                                elif args.adversarial_loss == "CDA":
                                    # 条件领域对抗 (Condition DA) 网络损失
                                    # 分离分类器特征，防止梯度传播到分类器
                                    softmax_out = self.softmax_layer_ad(outputs).detach()

                                    # 取张量积后展开计算
                                    # 概率转化为行向量，特征转化为列向量，乘积维数为(batch_size, num_classes, featue_dim)
                                    # op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    op_out = softmax_out.unsqueeze(2) * features.unsqueeze(1)
                                    adversarial_out = self.AdversarialNet(op_out.view(-1, softmax_out.size(1) * features.size(1)))

                                    domain_label_source = torch.zeros(labels.size(0)).float()
                                    domain_label_target = torch.ones(inputs.size(0) - labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_loss = self.adversarial_loss(adversarial_out.view(-1), adversarial_label)
                                elif args.adversarial_loss == "CDA+E":
                                    # 引入熵方法，对概率重新加权
                                    softmax_out = self.softmax_layer_ad(outputs)

                                    # 梯度反转层的系数与当前迭代轮次有关
                                    coeff = calc_coeff(iter_num_entropy, max_iter=self.max_iter)
                                    iter_num_entropy += 1
                                    entropy = Entropy(softmax_out)

                                    # 在熵中施加梯度反转，以最大化领域分类误差
                                    entropy.register_hook(grl_hook(coeff))

                                    # 计算权重，提取源域和目标域部分
                                    entropy = 1.0 + torch.exp(-entropy)
                                    entropy_source = entropy.narrow(0, 0, labels.size(0))
                                    entropy_target = entropy.narrow(0, labels.size(0), inputs.size(0) - labels.size(0))

                                    # 分离分类器特征，防止对抗训练的梯度影响主分类
                                    softmax_out = softmax_out.detach()
                                    # op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    op_out = softmax_out.unsqueeze(2) * features.unsqueeze(1)
                                    adversarial_out = self.AdversarialNet(op_out.view(-1, softmax_out.size(1) * features.size(1)))

                                    # 标签生成
                                    domain_label_source = torch.zeros(labels.size(0)).float()
                                    domain_label_target = torch.ones(inputs.size(0) - labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    weight = torch.cat(
                                        (entropy_source / torch.sum(entropy_source).detach().item(), entropy_target / torch.sum(entropy_target).detach().item()), dim=0
                                    )

                                    # 展开权重，对损失重新加权
                                    adversarial_loss = torch.sum(weight.view(-1, 1) * self.adversarial_loss(adversarial_out.squeeze(), adversarial_label))
                                    adversarial_loss /= torch.sum(weight).detach().item()

                                else:
                                    raise Exception("loss not implement")
                            else:
                                adversarial_loss = 0

                            # 计算trade_off参数
                            if args.distance_tradeoff == "Cons":
                                distance_lambda = args.distance_lambda
                            elif args.distance_tradeoff == "Step":
                                tmp = -10 * ((epoch - args.middle_epoch) / (args.max_epoch - args.middle_epoch))
                                distance_lambda = 2 / (1 + math.exp(tmp)) - 1
                            else:
                                raise Exception("trade_off_distance not implement")

                            if args.adversarial_tradeoff == "Cons":
                                adversarial_lambda = args.lam_adversarial
                            elif args.adversarial_tradeoff == "Step":
                                tmp = -10 * ((epoch - args.middle_epoch) / (args.max_epoch - args.middle_epoch))
                                adversarial_lambda = 2 / (1 + math.exp(-10 * ((epoch - args.middle_epoch) / (args.max_epoch - args.middle_epoch)))) - 1
                            else:
                                raise Exception("loss not implement")

                            loss = classifier_loss + distance_lambda * distance_loss + adversarial_lambda * adversarial_loss

                        # 计算每个batch的损失、预测正确个数与长度
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # 计算训练信息
                        if phase == "source_train":
                            # 反向传播
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            # 计算批次损失、准确率与长度
                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)

                            # 记录训练信息
                            if batch_step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if batch_step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                current_sample_count = batch_idx * args.batch_size + len(labels)
                                logging.info(
                                    f"Epoch: {epoch} [{current_sample_count}/{len(self.dataloaders[phase].dataset)}], Train Loss: {batch_loss:.4f} Train Acc: {batch_acc:.4f},{sample_per_sec:.1f} examples/sec {batch_time:.2f} sec/batch"
                                )
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            batch_step += 1

                # 记录每个epoch的损失与准确率
                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length
                logging.info(f"Epoch: {epoch} {phase}-Loss: {epoch_loss:.4f} {phase}-Acc: {epoch_acc:.4f}, Cost {time.time() - epoch_start:.1f} sec")

                # 保存模型
                if phase == "target_val":
                    # 保存最优模型参数
                    # 判据：保存middle_epoch后准确率更优或达到最大轮次的epoch中的参数
                    model_state_dic = self.model_all.state_dict()
                    if (epoch_acc > best_acc or epoch == args.max_epoch - 1) and (epoch > args.middle_epoch - 1):
                        best_acc = epoch_acc
                        logging.info(f"save best model epoch {epoch}, acc {epoch_acc:.4f}")
                        torch.save(model_state_dic, os.path.join(self.save_dir, f"{epoch}-{best_acc:.4f}-best_model.bin"))

                self.acc[phase] = np.append(self.acc[phase], epoch_acc)
                self.loss[phase] = np.append(self.loss[phase], epoch_loss)

            # 每个epoch结束后更新学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def plot(self):
        args = self.args

        plt.subplot(1, 2, 1)
        plt.title("Accuracy")
        plt.xlabel("epoches")
        plt.ylabel("accuracy")
        plt.plot(range(args.max_epoch), self.acc["source_train"], label="source_train")
        plt.plot(range(args.max_epoch), self.acc["source_val"], label="source_val")
        plt.plot(range(args.max_epoch), self.acc["target_val"], label="target_val")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title(f"Loss Function: {args.distance_loss}")
        plt.xlabel("epoches")
        plt.ylabel("accuracy")
        plt.plot(range(args.max_epoch), self.loss["source_train"], label="source_train")
        plt.plot(range(args.max_epoch), self.loss["source_val"], label="source_val")
        plt.plot(range(args.max_epoch), self.loss["target_val"], label="target_val")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def generate_fig(self):
        args = self.args

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].set_title("Accuracy")
        axs[0].set_xlabel("epoches")
        axs[0].set_ylabel("accuracy")
        axs[0].plot(range(args.max_epoch), self.acc["source_train"], label="source_train")
        axs[0].plot(range(args.max_epoch), self.acc["source_val"], label="source_val")
        axs[0].plot(range(args.max_epoch), self.acc["target_val"], label="target_val")
        axs[0].legend()

        axs[1].set_title(f"Loss Function: {args.distance_loss}")
        axs[1].set_xlabel("epoches")
        axs[1].set_ylabel("loss")
        axs[1].plot(range(args.max_epoch), self.loss["source_train"], label="source_train")
        axs[1].plot(range(args.max_epoch), self.loss["source_val"], label="source_val")
        axs[1].plot(range(args.max_epoch), self.loss["target_val"], label="target_val")
        axs[1].legend()

        plt.tight_layout()
        return fig
