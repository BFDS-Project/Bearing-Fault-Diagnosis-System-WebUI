'''
此文件存储train.py种迁移学习部分采用的损失函数
'''
import torch

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    生成多核Gauss核, 便于将取内积的数据映射到无穷维
    Args:
        source(_torch.tensor_): 源域数据
        target(_torch.tenser_): 目标域数据
        kernel_mul(_float_): 带宽放大系数
        kernel_num(_int_): Gauss核的个数
        fix_sigma(_float_): 带宽中值, 缺省时动态生成
    Returns:
        (_torch.tensor_): 对于每个batch的数据返回一个矩阵, 存储两两间的Gauss核函数值, 且同时包括源域和目标域, 多重Gauss核直接求和
    '''
    # 源域与目标域数据拼接
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    # 计算源域、目标域样本中两两的L2距离（与标签无关）
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    
    # 计算带宽（等比数列）
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # 带宽不固定时，用样本对之间的L2距离平均值作为带宽中值
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # 计算多核的高斯核值，带宽即为2σ^2
    # k(x_i, x_j) = \exp (−\Vert x_i - x_j \Vert ^2 / 2\sigma^2)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)

def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    DAN (Deep Adaption Network 深度适应网络) 迁移方法的损失函数
    Args:
        source(_torch.tensor_): 源域数据
        target(_torch.tenser_): 目标域数据
        kernel_mul(_float_): 带宽放大系数
        kernel_num(_int_): Gauss核的个数
        fix_sigma(_float_): 带宽中值, 缺省时动态生成
    Returns:
        (_float_): 返回MK-MMD (Multikernel-Maximum Mean Discrepancy 多核最大均值差异) 损失函数
    '''
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, 
                              target,
                              kernel_mul = kernel_mul,
                              kernel_num = kernel_num,
                              fix_sigma = fix_sigma)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    return torch.mean(XX + YY - XY - YX)

def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    '''
    JAN (Joint Adaption Network 联合适应网络) 迁移方法的损失函数
    多个特征层上同时对齐, 因此输入的是一系列的源域和目标域数据
    Args:
        source_list(_list_torch.tensor_): 源域数据
        target_list(_list_torch.tenser_): 目标域数据
        kernel_muls(_list_float_): 每层Gauss核的带宽放大系数
        kernel_nums(_list_int_): 每层Gauss核的个数
        fix_sigma_list(_list_float_): 每层Gauss核的带宽中值, 缺省时动态生成
    Returns:
        (_float_): 返回多层联合的MK-MMD (Multikernel-Maximum Mean Discrepancy 多核最大均值差异) 损失函数
    '''
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = gaussian_kernel(source, 
                                  target,
                                  kernel_mul=kernel_mul,
                                  kernel_num=kernel_num,
                                  fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            # 多层数据算得的Gauss核函数值对应项相乘，实现联合分布
            joint_kernels = kernels

    XX = joint_kernels[:batch_size, :batch_size]
    YY = joint_kernels[batch_size:, batch_size:]
    XY = joint_kernels[:batch_size, batch_size:]
    YX = joint_kernels[batch_size:, :batch_size]

    return torch.mean(XX + YY - XY - YX)

def CORAL(source, target):
    '''
    CORAL (Correlation Aline 相关性对齐) 迁移方法的损失函数
    Args:
        source(_torch.tensor_): 源域数据
        target(_torch.tenser_): 目标域数据
    Returns:
        loss(_float_): CORAL损失项 L_CORAL
    '''
    d = source.data.shape[1]
    xc = torch.cov(source.T)
    xt = torch.cov(target.T)
 
    return torch.sum(torch.mul(xc-xt, xc-xt))/(4*d**2)