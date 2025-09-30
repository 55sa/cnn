# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Flatten(Module):
    """
    Flatten层将4D批次数据展平为2D矩阵数据，供Dense层使用。
    没有可学习参数，只是简单的形状变换。
    """
    
    def __init__(self: Flatten) -> None:
        # Flatten层没有参数，不需要构造函数
        pass
    
    def forward(self: Flatten,
                X: np.ndarray) -> np.ndarray:
        """
        将输入批次展平为2D矩阵
        Args:
            X: 输入数据，形状为 (num_examples, height, width, channels)
        Returns:
            展平后的数据，形状为 (num_examples, height*width*channels)
        """
        # 保存原始形状用于backward
        self.original_shape = X.shape
        
        # 将除了第一个维度（批次维度）外的所有维度展平
        return X.reshape(X.shape[0], -1)
    
    def backward(self: Flatten,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        """
        将2D梯度重新整形为原始4D形状
        Args:
            X: 原始输入数据
            dLoss_dModule: 来自下游层的2D梯度
        Returns:
            重新整形为原始形状的梯度
        """
        # 将2D梯度重新整形为原始4D形状
        return dLoss_dModule.reshape(self.original_shape)
    
    def parameters(self: Flatten) -> List[Parameter]:
        """
        返回参数列表（Flatten层没有参数）
        """
        return []
