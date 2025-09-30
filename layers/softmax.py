# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Softmax(Module):

    # technically we can introduce a stretch coefficient to our softmax
    # function. This is a coefficient we multiply our input vector by.
    # This value controls how "skewed" of a distribution softmax produces.
    def __init__(self: Softmax,
                 stretch: float = 1.0) -> None:
        self.stretch: float = stretch

    # convert a vector into its "soft" probability distribution
    # however we are going to do a computational trick here that will
    # make our softmax computation more stable.

    # basically if numbers get too big (or too small), we risk
    # overflow or underflow. So, we want to avoid the values
    # from becoming too extreme.

    # unfortunatley, we need to exponentiate values which runs the risk
    # of overflow. So, here is our trick:

    # the original definition of softmax is:
    #       e^(x_i) / sum_{forall j} (e^(x_j))

    # we're going to compute:
    #       e^(x_i - max_{forall j} (x_j)) / sum_{forall j} e^(x_j - max_{forall k} (x_k))

    # this keeps the power of e from being too big.
    # also, this term factors out from both the numerator and denominator
    # aka it doesn't change the value of the output.
    def forward(self: Softmax,
                X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        
        # 数值稳定性：减去每行的最大值
        X_max = np.max(X, axis=1, keepdims=True)
        X_shifted = X - X_max
        
        # 计算softmax
        exp_X = np.exp(self.stretch * X_shifted)
        exp_sum = np.sum(exp_X, axis=1, keepdims=True)
        
        return exp_X / exp_sum

    # TODO:
    # this method computes the jacobean matrix for a single example.
    # since this is an element-wise DEPENDENT activation function,
    # this jacobean is NOT diagonal, so we must compute all the values.

    # the jacobean should take the form of:
    # dyhat / dx =
    #              yhat_1        yhat_2      ...      yhat_n
    #     x_1 |  dyhat_1/dx_1  dyhat_2/dx_1  ...   dyhat_n/dx_1 |
    #     x_2 |  dyhat_1/dx_2  dyhat_2/dx_2  ...   dyhat_n/dx_2 |
    #     ... |     ...           ...                   ...     |
    #     x_n |  dyhat_1/dx_n  dyhat_2/dx_n  ...   dyhat_n/dx_n |
    def jacobian_single_example(self: Softmax,
                                x: np.ndarray, 
                                y_hat: np.ndarray) -> np.ndarray:
        """
        计算单个样本的雅可比矩阵
        Args:
            x: 输入向量
            y_hat: softmax输出向量
        Returns:
            雅可比矩阵，形状为 (n, n)
        """
        n = len(x)
        jacobian = np.zeros((n, n))
        
        # 对于softmax，雅可比矩阵的公式为：
        # ∂y_i/∂x_j = y_i * (δ_ij - y_j)
        # 其中δ_ij是Kronecker delta
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # 当i=j时：∂y_i/∂x_i = y_i * (1 - y_i)
                    jacobian[i, j] = y_hat[i] * (1 - y_hat[i])
                else:
                    # 当i≠j时：∂y_i/∂x_j = -y_i * y_j
                    jacobian[i, j] = -y_hat[i] * y_hat[j]
        
        return jacobian


    # this is done for you! This is because combining the
    # jacobean matrices with dLoss_dModule is actually pretty complicated
    # so I provided that code for you. Don't worry, I already tested this
    # works with a correct self.jacobean_single_example
    # so please don't change this!
    def backward(self: Softmax,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        Y_hat: np.ndarray = self.forward(X)

        # get the jacobean for each example as a massive 3d matrix
        # the first axis is the batch_dim.
        dYhat_dX: np.ndarray = np.zeros(Y_hat.shape + (Y_hat.shape[-1],))
        for example_idx, (x, y_hat) in enumerate(zip(X, Y_hat)):
            dYhat_dX[example_idx] = self.jacobian_single_example(x, y_hat)

        # this convoluted code is how we distribute dLoss_dModule
        # to the corresponding jacobeans, and then sum up the jacobeans.
        # This took me a while to figure out lol please don't change it.
        return np.einsum("...jk, ...kl", dYhat_dX,
            np.expand_dims(dLoss_dModule, axis=-1)).reshape(X.shape)

    def parameters(self: Softmax) -> List[Parameter]:
        return list()

