# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Tuple
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class MaxPool2d(Module):
    """
    2D最大池化层
    通过取池化窗口内的最大值来下采样特征图
    """
    
    def __init__(self: MaxPool2d,
                 pool_size: Tuple[int, int],
                 stride: int = 2) -> None:
        """
        初始化最大池化层
        Args:
            pool_size: 池化窗口大小 (height, width)
            stride: 步长，默认为2
        """
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self: MaxPool2d,
                X: np.ndarray) -> np.ndarray:
        """
        前向传播：对输入进行最大池化
        Args:
            X: 输入数据，形状为 (num_examples, height, width, num_channels)
        Returns:
            池化后的数据，形状为 (num_examples, out_height, out_width, num_channels)
        """
        num_examples, h, w, num_channels = X.shape
        pool_h, pool_w = self.pool_size
        
        # 计算输出尺寸
        out_h = 1 + (h - pool_h) // self.stride
        out_w = 1 + (w - pool_w) // self.stride
        
        # 预分配输出数组
        output = np.zeros((num_examples, out_h, out_w, num_channels))
        
        # 对每个输出像素进行池化
        for out_h_idx in range(out_h):
            for out_w_idx in range(out_w):
                # 计算输入窗口的起始和结束位置
                h_start = out_h_idx * self.stride
                h_end = h_start + pool_h
                w_start = out_w_idx * self.stride
                w_end = w_start + pool_w
                
                # 提取池化窗口
                pool_window = X[:, h_start:h_end, w_start:w_end, :]
                
                # 对每个通道分别计算最大值
                # 使用np.max(axis=(1,2))来对高度和宽度维度求最大值
                output[:, out_h_idx, out_w_idx, :] = np.max(pool_window, axis=(1, 2))
        
        return output
    
    def backward(self: MaxPool2d,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        """
        反向传播：计算梯度
        Args:
            X: 原始输入数据
            dLoss_dModule: 来自下游层的梯度
        Returns:
            对输入数据的梯度
        """
        num_examples, h, w, num_channels = X.shape
        pool_h, pool_w = self.pool_size
        
        # 计算输出尺寸
        out_h = 1 + (h - pool_h) // self.stride
        out_w = 1 + (w - pool_w) // self.stride
        
        # 初始化梯度数组
        dLoss_dX = np.zeros_like(X)
        
        # 对每个输出像素计算梯度
        for out_h_idx in range(out_h):
            for out_w_idx in range(out_w):
                # 计算输入窗口的起始和结束位置
                h_start = out_h_idx * self.stride
                h_end = h_start + pool_h
                w_start = out_w_idx * self.stride
                w_end = w_start + pool_w
                
                # 提取池化窗口
                pool_window = X[:, h_start:h_end, w_start:w_end, :]
                
                # 对每个通道分别找到最大值的掩码
                for c in range(num_channels):
                    # 找到每个样本中该通道的最大值位置
                    max_vals = np.max(pool_window[:, :, :, c], axis=(1, 2), keepdims=True)
                    
                    # 创建掩码：最大值位置为1，其他位置为0
                    mask = (pool_window[:, :, :, c] == max_vals)
                    
                    # 将梯度分配到最大值位置
                    # 使用广播将dLoss_dModule的值分配到对应的最大值位置
                    dLoss_dX[:, h_start:h_end, w_start:w_end, c] += \
                        mask * dLoss_dModule[:, out_h_idx:out_h_idx+1, out_w_idx:out_w_idx+1, c]
        
        return dLoss_dX
    
    def parameters(self: MaxPool2d) -> List[Parameter]:
        """
        返回参数列表（池化层没有参数）
        """
        return []
