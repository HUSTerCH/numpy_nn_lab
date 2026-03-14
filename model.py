import numpy as np

from components import FullyConnectedLayer, ReLU, Tanh, Sigmoid, SoftmaxCrossEntropy, LeakyReLU


class Model:
    def __init__(self, layer_dims, init_method, activation='relu', l2_reg=0.0):
        self.layers = []
        self.l2_reg = l2_reg

        # 按照 input -> FC -> Act -> FC -> Act -> output 串联
        for i in range(len(layer_dims) - 1):
            self.layers.append(FullyConnectedLayer(layer_dims[i], layer_dims[i + 1], init_method, l2_reg))
            # 最后一层全连接后不加常规激活函数
            if i < len(layer_dims) - 2:
                if activation == 'relu':
                    self.layers.append(ReLU())
                elif activation == 'leaky_relu':
                    self.layers.append(LeakyReLU())
                elif activation == 'tanh':
                    self.layers.append(Tanh())
                elif activation == 'sigmoid':
                    self.layers.append(Sigmoid())

        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X, Y):
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        # 计算基础交叉熵损失
        data_loss = self.loss_fn.forward(out, Y)

        # 加入 L2 正则化惩罚项
        l2_loss = 0.0
        if self.l2_reg > 0:
            for layer in self.layers:
                if isinstance(layer, FullyConnectedLayer):
                    l2_loss += 0.5 * self.l2_reg * np.sum(layer.W ** 2)

        m = X.shape[0]
        total_loss = data_loss + (l2_loss / m)
        return total_loss, self.loss_fn.Y_pred

    def backward(self):
        # 从损失函数开始，逆序传播梯度
        dout = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_params(self, lr):
        # Mini-batch 梯度下降参数更新
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
