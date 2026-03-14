import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, init_method, l2_reg=0.0):
        # 根据传入参数选择初始化策略
        if init_method == 'normal':
            # 原始基础方案
            self.W = np.random.randn(input_dim, output_dim) * 0.01
        elif init_method == 'xavier':
            # Xavier (Glorot) 初始化：方差缩放为 2 / (fan_in + fan_out)
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        elif init_method == 'he':
            # He (Kaiming) 初始化：方差缩放为 2 / fan_in
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        else:
            raise ValueError(f"[Error] 不支持的初始化方法: {init_method}。请使用 'normal', 'xavier' 或 'he'。")
        self.b = np.zeros((1, output_dim))
        self.l2_reg = l2_reg
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        # Z=XW+b
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        m = self.X.shape[0]
        # 核心矩阵求导，加入L2正则化梯度项
        # 权重W求导 dW=(1/m)*X^T*dZ+(lambda/m)*W
        self.dW = np.dot(self.X.T, dZ) / m + (self.l2_reg / m) * self.W
        # 偏置b求导，db=(1/m)*sum(dZ,axis=0)，使用keepdims=True保证形状，而不是退化为一维向量
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        # 反向传播
        # 对输出X求导，dX=dZ*W^T
        dX = np.dot(dZ, self.W.T)
        return dX


class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        # ReLU 导数：Z > 0 时为 1，否则为 0
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0
        return dZ


class LeakyReLU:
    def __init__(self, alpha=0.01):
        # 负半轴泄露系数
        self.alpha = alpha
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        # Z > 0 时保持原样，Z <= 0 时乘以 alpha
        return np.where(Z > 0, Z, self.alpha * Z)

    def backward(self, dA):
        dZ = dA.copy()
        # Z<=0的位置，梯度乘alpha；Z>0的位置，梯度保持1
        dZ[self.Z <= 0] *= self.alpha
        return dZ


class Sigmoid:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        # 当 Z 极度偏小时，np.exp(-Z) 会超出 float64 上限导致溢出报错，需要强行截断
        Z_clipped = np.clip(Z, -500, 500)
        self.A = 1.0 / (1.0 + np.exp(-Z_clipped))
        return self.A

    def backward(self, dA):
        # Sigmoid 导数的极简矩阵形态：A * (1 - A)
        return dA * self.A * (1.0 - self.A)


class Tanh:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        # Tanh 导数：1 - A^2
        return dA * (1 - self.A ** 2)


class SoftmaxCrossEntropy:
    """
    将 Softmax 与 交叉熵 合并，避免 Jacobian 矩阵计算导致的性能断崖与数值溢出
    """
    def __init__(self):
        self.Y_pred = None
        self.Y_true = None

    def forward(self, Z, Y_true):
        # 减去最大值防止 exp(Z) 溢出为 NaN
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        self.Y_pred = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self.Y_true = Y_true

        m = Y_true.shape[0]
        # 加 1e-9 防止 log(0) 报错
        loss = -np.sum(Y_true * np.log(self.Y_pred + 1e-9)) / m
        return loss

    def backward(self):
        # 梯度简化为预测概率 P - 真实标签 Y
        return self.Y_pred - self.Y_true
