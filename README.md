# Bearing-Fault-Diagnosis-System

## Requirements
- Python 3.10
- matplotlib 3.10.0
- numpy 2.2.2
- PyWavelets 1.8.0

## Run
- BFDS_train.py

## 问题
- 目标域测试损失与源域损失完全不在一个数量级
- 引入目标域（middle_epoch）后领域对抗损失收敛到了一个非零点
- 准确率甚至低于20%