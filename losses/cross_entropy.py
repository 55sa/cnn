# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..param import Parameter
from ..lf import LossFunction


# 数值稳定性常数
DELTA: float = 1e-12


class CategoricalCrossEntropy(LossFunction):
    """
    分类交叉熵损失函数
    用于多分类问题，输入是预测的概率分布和真实标签的one-hot编码
    """
    
    def __init__(self: CategoricalCrossEntropy) -> None:
        # 损失函数没有可学习参数
        pass
    
    def forward(self: CategoricalCrossEntropy,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        """
        计算分类交叉熵损失
        Args:
            Y_hat: 预测的概率分布，形状为 (n, k)
            Y_gt: 真实标签的one-hot编码，形状为 (n, k)
        Returns:
            平均损失值
        """
        n = Y_hat.shape[0]
        
        # 添加数值稳定性常数
        Y_hat_stable = Y_hat + DELTA
        
        # 计算交叉熵损失: -1/n * sum(sum(Y_gt * log(Y_hat)))
        # 使用np.sum进行向量化计算
        loss = -np.sum(Y_gt * np.log(Y_hat_stable)) / n
        
        return loss
    
    def backward(self: CategoricalCrossEntropy,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        """
        计算损失函数对预测值的梯度
        Args:
            Y_hat: 预测的概率分布，形状为 (n, k)
            Y_gt: 真实标签的one-hot编码，形状为 (n, k)
        Returns:
            梯度，形状与Y_hat相同
        """
        n = Y_hat.shape[0]
        
        # 添加数值稳定性常数
        Y_hat_stable = Y_hat + DELTA
        
        # 梯度公式: dL/dY_hat = -Y_gt / (n * Y_hat)
        # 这里Y_gt是one-hot，所以只有对应类别的位置有值
        gradient = -Y_gt / (n * Y_hat_stable)
        
        return gradient
    
    def parameters(self: CategoricalCrossEntropy) -> List[Parameter]:
        """
        返回参数列表（损失函数没有参数）
        """
        return []
