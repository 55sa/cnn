# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Dense(Module):

    def __init__(self: Dense,
                 in_dim: int,
                 out_dim: int) -> None:
        self.W = Parameter(np.random.randn(in_dim, out_dim))
        self.b = Parameter(np.random.randn(1, out_dim))

    def forward(self: Dense,
                X: np.ndarray) -> np.ndarray:
        return X @ self.W.val + self.b.val

    def backward(self: Dense,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        dLoss_dX = dLoss_dModule @ self.W.val.T
        
        if not self.W.frozen:
            self.W.grad = X.T @ dLoss_dModule
        
        if not self.b.frozen:
            self.b.grad = np.sum(dLoss_dModule, axis=0, keepdims=True)
        
        return dLoss_dX

    def parameters(self: Dense) -> List[Parameter]:
        return [self.W, self.b]
