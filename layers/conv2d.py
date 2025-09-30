# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Conv2d(Module):
    """
    2D卷积层
    实现卷积操作，支持valid和same填充
    """
    
    def __init__(self: Conv2d,
                 num_kernels: int,
                 num_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: str = "valid") -> None:
        """
        初始化卷积层
        Args:
            num_kernels: 卷积核数量
            num_channels: 输入通道数
            kernel_size: 卷积核大小，可以是int或(int, int)
            stride: 步长，默认为1
            padding: 填充类型，"valid"或"same"
        """
        self.num_kernels = num_kernels
        self.num_channels = num_channels
        self.stride = stride
        self.padding = padding
        
        # 处理kernel_size参数
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        # 创建权重和偏置参数
        # 权重形状: (kernel_height, kernel_width, num_channels, num_kernels)
        self.W = Parameter(np.random.randn(
            self.kernel_size[0], 
            self.kernel_size[1], 
            num_channels, 
            num_kernels
        ))
        
        # 偏置形状: (num_kernels,)
        self.b = Parameter(np.random.randn(num_kernels))
        
        # 计算填充量
        self.pad_amounts = self.get_pad_amounts()
    
    def get_pad_amounts(self: Conv2d) -> Tuple[int, int]:
        """
        计算填充量
        Returns:
            (height_pad, width_pad): 高度和宽度的填充量
        """
        if self.padding == "valid":
            return (0, 0)
        elif self.padding == "same":
            # 计算需要的填充量，使输出尺寸与输入相同
            height_pad = self.kernel_size[0] // 2
            width_pad = self.kernel_size[1] // 2
            return (height_pad, width_pad)
        else:
            raise ValueError(f"不支持的填充类型: {self.padding}")
    
    def pad_imgs(self: Conv2d,
                 X: np.ndarray) -> np.ndarray:
        """
        对输入图像进行填充
        Args:
            X: 输入批次，形状为 (num_examples, height, width, num_channels)
        Returns:
            填充后的图像
        """
        if self.pad_amounts == (0, 0):
            return X
        
        height_pad, width_pad = self.pad_amounts
        
        # 使用np.pad进行填充
        # 填充维度: ((0,0), (height_pad, height_pad), (width_pad, width_pad), (0,0))
        # 分别对应: 批次维度, 高度维度, 宽度维度, 通道维度
        padded_X = np.pad(X, 
                         ((0, 0), (height_pad, height_pad), (width_pad, width_pad), (0, 0)),
                         mode='constant', 
                         constant_values=0.0)
        
        return padded_X
    
    def get_out_shape(self: Conv2d,
                      X: np.ndarray) -> Tuple[int, int, int, int]:
        """
        计算输出形状
        Args:
            X: 输入批次（已填充）
        Returns:
            输出形状 (num_examples, out_height, out_width, num_kernels)
        """
        num_examples, h, w, num_channels = X.shape
        kernel_h, kernel_w = self.kernel_size
        
        # 使用公式: d_out = (d_in - kernel + 2*pad) / stride + 1
        out_h = (h - kernel_h) // self.stride + 1
        out_w = (w - kernel_w) // self.stride + 1
        
        return (num_examples, out_h, out_w, self.num_kernels)
    
    def forward(self: Conv2d,
                X: np.ndarray) -> np.ndarray:
        """
        前向传播：执行卷积操作
        Args:
            X: 输入批次，形状为 (num_examples, height, width, num_channels)
        Returns:
            卷积输出，形状为 (num_examples, out_height, out_width, num_kernels)
        """
        # 填充输入
        X_padded = self.pad_imgs(X)
        
        # 计算输出形状
        out_shape = self.get_out_shape(X_padded)
        output = np.zeros(out_shape)
        
        num_examples, out_h, out_w, num_kernels = out_shape
        kernel_h, kernel_w = self.kernel_size
        
        # 对每个输出像素进行卷积
        for out_h_idx in range(out_h):
            for out_w_idx in range(out_w):
                # 计算输入窗口的起始和结束位置
                h_start = out_h_idx * self.stride
                h_end = h_start + kernel_h
                w_start = out_w_idx * self.stride
                w_end = w_start + kernel_w
                
                # 提取输入窗口
                input_window = X_padded[:, h_start:h_end, w_start:w_end, :]
                
                # 对每个卷积核进行卷积
                for k in range(num_kernels):
                    # 获取当前卷积核
                    kernel = self.W.val[:, :, :, k]  # 形状: (kernel_h, kernel_w, num_channels)
                    
                    # 执行卷积：对每个样本的输入窗口与卷积核进行元素级乘法后求和
                    # 使用np.sum(axis=(1,2,3))对空间和通道维度求和
                    conv_result = np.sum(input_window * kernel, axis=(1, 2, 3))
                    
                    # 添加偏置
                    output[:, out_h_idx, out_w_idx, k] = conv_result + self.b.val[k]
        
        return output
    
    def backward(self: Conv2d,
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
        # 填充输入
        X_padded = self.pad_imgs(X)
        
        num_examples, out_h, out_w, num_kernels = dLoss_dModule.shape
        kernel_h, kernel_w = self.kernel_size
        
        # 初始化梯度
        dLoss_dX = np.zeros_like(X_padded)
        dLoss_dW = np.zeros_like(self.W.val)
        dLoss_db = np.zeros_like(self.b.val)
        
        # 计算权重和偏置的梯度
        for out_h_idx in range(out_h):
            for out_w_idx in range(out_w):
                # 计算输入窗口的起始和结束位置
                h_start = out_h_idx * self.stride
                h_end = h_start + kernel_h
                w_start = out_w_idx * self.stride
                w_end = w_start + kernel_w
                
                # 提取输入窗口
                input_window = X_padded[:, h_start:h_end, w_start:w_end, :]
                
                # 对每个卷积核计算梯度
                for k in range(num_kernels):
                    # 获取当前卷积核
                    kernel = self.W.val[:, :, :, k]
                    
                    # 计算权重梯度
                    # dLoss_dW[:, :, :, k] += sum over batch of (input_window * dLoss_dModule[:, out_h_idx, out_w_idx, k])
                    grad_k = dLoss_dModule[:, out_h_idx, out_w_idx, k]
                    dLoss_dW[:, :, :, k] += np.sum(input_window * grad_k[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    
                    # 计算偏置梯度
                    dLoss_db[k] += np.sum(grad_k)
                    
                    # 计算输入梯度
                    # dLoss_dX += kernel * dLoss_dModule (broadcasted)
                    dLoss_dX[:, h_start:h_end, w_start:w_end, :] += \
                        kernel[np.newaxis, :, :, :] * grad_k[:, np.newaxis, np.newaxis, np.newaxis]
        
        # 更新参数梯度
        self.W.grad = dLoss_dW
        self.b.grad = dLoss_db
        
        # 移除填充，返回原始形状的梯度
        if self.pad_amounts != (0, 0):
            height_pad, width_pad = self.pad_amounts
            return dLoss_dX[:, height_pad:-height_pad, width_pad:-width_pad, :]
        else:
            return dLoss_dX
    
    def parameters(self: Conv2d) -> List[Parameter]:
        """
        返回参数列表
        """
        return [self.W, self.b]
