#!/bin/bash
set -e

LOG_DIR="./logs"
mkdir -p $LOG_DIR

echo "启动批处理流水线"

echo ">>> 正在执行 Baseline 组实验..."
/Users/luochang/PycharmProjects/numpy_nn_lab/.venv/bin/python main.py \
    --layer_dims 784 256 128 10 \
    --activation relu \
    --lr 0.01 \
    --batch_size 64 \
    --l2_reg 1e-4 \
    --group baseline \
    > $LOG_DIR/baseline.log 2>&1
echo "    Baseline 执行完毕，日志已保存至 $LOG_DIR/baseline.log"

echo ">>> 正在执行 组1 实验..."
/Users/luochang/PycharmProjects/numpy_nn_lab/.venv/bin/python main.py \
    --layer_dims 784 512 256 10 \
    --activation leaky_relu \
    --lr 0.005 \
    --batch_size 128 \
    --l2_reg 1e-3 \
    --group group1 \
    > $LOG_DIR/group1.log 2>&1
echo "    组1 执行完毕，日志已保存至 $LOG_DIR/group1.log"


echo ">>> 正在执行 组2 实验..."
/Users/luochang/PycharmProjects/numpy_nn_lab/.venv/bin/python main.py \
    --layer_dims 784 128 10 \
    --activation tanh \
    --lr 0.02 \
    --batch_size 32 \
    --l2_reg 0 \
    --group group2 \
    > $LOG_DIR/group2.log 2>&1
echo "    组2 执行完毕，日志已保存至 $LOG_DIR/group2.log"

echo "全部实验运行完成！"