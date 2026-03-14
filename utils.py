import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml


def load_mnist_data(cache_dir='./data'):
    cache_path = os.path.join(cache_dir, 'mnist_cache.pkl')
    if os.path.exists(cache_path):
        print('Loading cached MNIST data...')
        with open(cache_path, 'rb') as f:
            X, y = pickle.load(f)
    else:
        print('Loading MNIST data from Web...')
        os.makedirs(cache_dir, exist_ok=True)
        mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
        X, y = mnist['data'], mnist['target'].astype(np.int8)
        with open(cache_path, 'wb') as f:
            pickle.dump((X, y), f)
    X = X / 255.0
    y_one_hot = np.eye(10)[y]

    np.random.seed(56)
    indices = np.random.permutation(X.shape[0])
    X_shuffled, y_shuffled = X[indices], y_one_hot[indices]
    split_idx = int(X.shape[0] * 0.8)
    X_train, y_train = X_shuffled[:split_idx], y_shuffled[:split_idx]
    X_test, y_test = X_shuffled[split_idx:], y_shuffled[split_idx:]
    print(f"Dataset loaded, Train X shape:{X_train.shape}, Train y shape:{y_train.shape}")
    return X_train, y_train, X_test, y_test


def get_batches(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


def plot_curve(train_losses, train_accuracies, val_losses, val_accuracies, group):
    plt.figure(figsize=(12, 3), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{group} Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title(f'{group} Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{group}_loss_accuracy_curve.png')
    plt.close()
    # plt.show()


def visualize_predictions(model, group, num_samples=15):
    X_train, y_train, X_val, y_val = load_mnist_data()
    # 随机打乱并采样
    m = X_val.shape[0]
    indices = np.random.choice(m, min(num_samples, m), replace=False)
    X_sample = X_val[indices]
    y_sample = y_val[indices]  # 注意：此时是 One-hot 编码

    # 前向传播
    _, Y_pred = model.forward(X_sample, y_sample)

    # 将概率矩阵和 One-hot 标签转回具体的类别索引 (0-9)
    preds = np.argmax(Y_pred, axis=1)
    trues = np.argmax(y_sample, axis=1)

    # 绘图
    figure = plt.figure(figsize=(15, 9))
    cols, rows = 5, 3

    print(f"生成可视化结果图 (随机选取 {len(indices)} 张)")
    os.makedirs('plots', exist_ok=True)

    for i in range(len(indices)):
        # 将 784 维的一维向量重新 Reshape 为 28x28 的二维图像
        img = X_sample[i].reshape(28, 28)

        figure.add_subplot(rows, cols, i + 1)

        # 预测正确标绿，预测错误标红
        color = 'green' if preds[i] == trues[i] else 'red'
        plt.title(f"{group} Pred: {preds[i]} (True: {trues[i]})", color=color, fontsize=14, fontweight='bold')

        plt.axis("off")
        plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.savefig(f'plots/{group}_predictions_vis.png')
    plt.close(figure)
    # plt.show()

def visualize_errors(model, group, num_samples=15):
    # 找出所有错误
    X_train, y_train, X_val, y_val = load_mnist_data()
    _, Y_pred = model.forward(X_val, y_val)

    # 将概率矩阵和 One-hot 标签转回具体的类别索引 (0-9)
    preds = np.argmax(Y_pred, axis=1)
    trues = np.argmax(y_val, axis=1)

    # 提取所有预测错误样本的索引
    error_indices = np.where(preds != trues)[0]
    total_errors = len(error_indices)

    if total_errors == 0:
        print(f">>> [INFO] {group} 模型验证集准确率 100%，无错误样本可供可视化！")
        return

    # 从错误样本库中随机采样
    sample_size = min(num_samples, total_errors)
    selected_indices = np.random.choice(error_indices, sample_size, replace=False)

    X_errors = X_val[selected_indices]
    preds_errors = preds[selected_indices]
    trues_errors = trues[selected_indices]

    figure = plt.figure(figsize=(15, 9))
    cols, rows = 5, 3

    print(f"生成【分类错误】分析图 - {group} (总错误数: {total_errors}, 随机抽取 {sample_size} 张)")
    # 确保保存目录存在
    os.makedirs('plots', exist_ok=True)

    for i in range(sample_size):
        # 降维打击：恢复 28x28 空间结构
        img = X_errors[i].reshape(28, 28)

        figure.add_subplot(rows, cols, i + 1)

        # 因为筛选出来的全都是错的，直接硬编码标红即可
        plt.title(f"{group} Pred: {preds_errors[i]} (True: {trues_errors[i]})",
                  color='red', fontsize=14, fontweight='bold')

        plt.axis("off")
        plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.savefig(f'plots/{group}_errors_vis.png')
    plt.close(figure)


def plot_confusion_matrix(model,  group_name="Baseline"):
    """
    绘制混淆矩阵。
    """
    X_train, y_train, X_val, y_val = load_mnist_data()
    _, Y_pred = model.forward(X_val, y_val)
    preds = np.argmax(Y_pred, axis=1)
    trues = np.argmax(y_val, axis=1)


    num_classes = 10
    cm = np.bincount(num_classes * trues + preds,
                     minlength=num_classes ** 2).reshape(num_classes, num_classes)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用蓝色色带，数据越大颜色越深
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # 设置刻度标记
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(np.arange(num_classes), fontsize=12)
    ax.set_yticklabels(np.arange(num_classes), fontsize=12)

    # 设置轴标签与标题
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {group_name}', fontsize=16, fontweight='bold')

    # 根据底色动态调整字体颜色
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{group_name}_confusion_matrix.png')
    plt.close(fig)
    print(f">>> [INFO] {group_name} 混淆矩阵已保存至 plots/{group_name}_confusion_matrix.png")

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
