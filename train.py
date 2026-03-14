import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 解决mac绘图崩溃/不显示问题

from utils import load_mnist_data, get_batches
from model import Model


def compute_accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=1)
    labels = np.argmax(Y_true, axis=1)
    return np.mean(predictions == labels)


def train_model(layer_dims: list[int], init_method: str, epochs: int, activation: str = 'relu', lr: float = 0.1, batch_size: int = 64, l2_reg: float = 1e-4) -> tuple[Model, list, list, list, list]:
    # 加载数据
    X_train, y_train, X_val, y_val = load_mnist_data()
    # 初始化模型
    model = Model(layer_dims, init_method, activation, l2_reg)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    print(f"开始训练: 架构={layer_dims}, 激活={activation}, LR={lr}, Batch={batch_size}, L2={l2_reg}")

    # 训练
    for epoch in range(epochs):
        # 打乱训练数据
        indices = np.random.permutation(X_train.shape[0])
        X_train_sh = X_train[indices]
        y_train_sh = y_train[indices]

        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0

        for X_batch, y_batch in get_batches(X_train_sh, y_train_sh, batch_size):
            # 前向传播 -> 反向传播 -> 更新权重
            loss, y_pred = model.forward(X_batch, y_batch)
            model.backward()
            model.update_params(lr)

            epoch_loss += loss
            epoch_acc += compute_accuracy(y_pred, y_batch)
            batches += 1

        train_losses.append(epoch_loss / batches)
        train_accuracies.append(epoch_acc / batches)

        # 验证集评估
        val_loss, val_pred = model.forward(X_val, y_val)
        val_acc = compute_accuracy(val_pred, y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs} | "
                  f"Train Loss: {train_losses[-1]:.4f} Acc: {train_accuracies[-1] * 100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}%")
    return model, train_losses, train_accuracies, val_losses, val_accuracies
