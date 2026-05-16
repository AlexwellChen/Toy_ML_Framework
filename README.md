# Toy ML Framework

C++ 实现的轻量级机器学习框架，支持**自动微分**、**静态计算图**和 **Keras 风格的高层 API**，用于训练 MNIST 手写数字分类器。

## 快速开始

### 依赖

- Eigen（`brew install eigen`）
- C++14

### 构建 & 运行

```bash
# 训练模型（编译 + 训练 + 生成 loss 曲线）
./scripts/run_train.sh

# 运行所有单元测试（13 个测试，8 个测试套件）
./scripts/run_tests.sh

# 清理编译产物
./scripts/clean.sh
```

### 训练结果

| 指标 | 值 |
|------|------|
| 测试准确率 | **97.5%** |
| 优化器 | AdamW（lr=0.005, β1=0.9, β2=0.999） |
| 网络结构 | Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Output(10) |
| 迭代轮数 | 5000 |

产物输出在 `output/` 目录：
- `mnist_weights.bin` — 可加载的模型权重
- `mnist_graph.txt` — 计算图结构
- `mnist_metrics.csv` — 训练指标（loss/准确率/AUC）
- `mnist_loss_curve.png` — Loss 曲线可视化

## 架构

框架分为三层抽象：

### 1. 计算图（MatAutoDiff.h/cpp, MatOp.h）

核心自动微分引擎，基于**反向模式**构建静态计算图。

- **Node** — 图节点，持有输入指针、Op、缓存结果（MatrixXd）和空间属性
- **Op** — 算子抽象基类，定义 `compute()`、`gradient()`、`getNewNode()`
- **gradients(output, inputs)** — 反向拓扑遍历输出节点的计算图，为每个输入构建梯度表达式
- **Executor** — 合并前向和反向图后进行拓扑排序，根据 feed_dict 依次执行各节点

支持的算子：Placeholder、MatMul、MatAdd、ReLU、Tanh、Softmax、Conv、MaxPool、Reshape、BatchNorm、Dropout 及其对应的梯度算子。

### 2. Model API（Model.h/cpp）

Keras 风格的顺序模型 API。

```cpp
Model model(/*batch_size*/ 64, /*iters*/ 5000, /*lr*/ 0.005f);
model.use_adam = true;  // 启用 AdamW

model.Input(784, "input");
model.Dense(128, "hidden1", "relu");
model.Dense(64, "hidden2", "relu");
model.Output(10, "output");

model.compile();
model.loadData("train.txt");
model.run();

model.save_weights("output/mnist_weights.bin");
model.save_graph("output/mnist_graph.txt");
```

支持的层类型：Input、Dense、Conv2D、MaxPooling2D、Flatten、Dropout、BatchNorm、Output。

### 3. 卷积原语（Conv.h/cpp）

- `img2col2D_multi()` — 多通道 im2col
- `col2im()` — im2col 的逆操作，用于梯度反传
- `MaxPooling2D()` / `MaxPoolingPrime()` — 支持 batch 的最大池化

## 单元测试

`unittest/test_ops.cpp` 覆盖所有算子的前向和反向测试，使用有限差分数值梯度检查。全部 23 个测试通过。

```bash
./unittest/test_ops
```

## 项目结构

```
├── main.cpp              # 训练入口
├── src/
│   ├── MatAutoDiff.cpp   # 计算图、算子实现、梯度构建
│   ├── Model.cpp          # 高层 Model/Layer API
│   └── Conv.cpp           # 卷积原语
├── include/
│   ├── MatAutoDiff.h
│   ├── MatOp.h
│   ├── Model.h
│   └── Conv.h
├── unittest/
│   └── test_ops.cpp       # 单元测试
├── example.cpp            # 底层 API 使用示例
├── Makefile.simple        # 构建脚本
├── train.txt              # MNIST 数据集
└── output/                # 训练产物
```

## 训练稳定性说明

- He 初始化 + AdamW 优化器，推荐 lr 0.001–0.01
- Softmax 使用稳定计算（减去最大值防溢出）
- Early stopping：自动保存验证准确率最佳时刻的权重
- Dropout 在推理时自动关闭（由 `g_dropout_training` 全局标志控制）
