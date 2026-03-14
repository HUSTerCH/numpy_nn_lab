import argparse
from train import train_model
from utils import plot_curve, visualize_predictions, visualize_errors, plot_confusion_matrix


def main():
    ap = argparse.ArgumentParser(description="NumPy 实现的全连接神经网络训练脚本")

    # 核心超参数暴露为命令行参数
    ap.add_argument('--layer_dims', type=int, nargs='+', default=[784, 256, 128, 10],
                    help="网络层维度列表，如: 784 256 128 10")
    ap.add_argument("--init_method", type=str, default="normal", choices=['normal', 'he', 'xavier'],
                    help="初始化方式，可选：normal he xavier")
    ap.add_argument('--activation', type=str, default='relu',
                    choices=['relu', 'leaky_relu', 'tanh', 'sigmoid'],
                    help="隐藏层激活函数类型")
    ap.add_argument('--lr', type=float, default=0.01,
                    help="学习率")
    ap.add_argument('--batch_size', type=int, default=64,
                    help="Mini-batch 大小")
    ap.add_argument('--l2_reg', type=float, default=1e-4,
                    help="L2 正则化系数")
    ap.add_argument('--epochs', type=int, default=25,
                    help="训练轮数")
    ap.add_argument('--group', type=str, help="组名")

    args = ap.parse_args()

    # 超参数设置
    layer_dims = args.layer_dims
    init_method = args.init_method
    activation = args.activation
    lr = args.lr  # 适当调大以适配He初始化
    batch_size = args.batch_size
    l2_reg = args.l2_reg
    epochs = args.epochs
    group = args.group
    if (not layer_dims) or layer_dims[-1] != 10 or layer_dims[0] != 784:
        print("网络层维度列表传参错误，要求：第一层必须为784，最后一层必须为10")
        return

    model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(layer_dims, init_method, epochs, activation, lr, batch_size, l2_reg)
    plot_curve(train_losses, train_accuracies, val_losses, val_accuracies, group)
    visualize_predictions(model, group)
    visualize_errors(model, group)
    plot_confusion_matrix(model, group)


if __name__ == '__main__':
    main()
