# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Build (requires Eigen, installed via Homebrew)
make -f Makefile.simple

# Build and run
make -f Makefile.simple run

# Build and run unit tests
make -f Makefile.simple test
```

**Dependencies:** Eigen (`brew install eigen`). Binary is `toy_ml`, test binary is `unittest/test_ops`.

The Makefile hardcodes a specific Eigen version path - update `INCLUDES` in `Makefile.simple` to match your installed version if the build fails.

## Architecture

This is a C++ educational ML framework implementing **reverse-mode automatic differentiation** over a static computation graph, designed for training MNIST classifiers.

### Per-operator file structure

Each operator has its own header, implementation, and unit test:

| Module | include/ | src/ | unittest/ |
|--------|----------|------|-----------|
| Graph engine (Op, Node, Executor, Placeholder, gradients) | `Node.h` | `Node.cpp` | — |
| MatMul | `MatMul.h` | `MatMul.cpp` | `test_matmul.cpp` |
| MatAdd | `MatAdd.h` | `MatAdd.cpp` | `test_matadd.cpp` |
| ReLU | `Relu.h` | `Relu.cpp` | `test_relu.cpp` |
| Softmax | `Softmax.h` | `Softmax.cpp` | `test_softmax.cpp` |
| OnesLike + ZerosLike | `Constant.h` | `Constant.cpp` | `test_constant.cpp` |
| Model + Layer + Data | `Model.h` | `Model.cpp` | `test_mlp.cpp` |

### Core engine

- `Op` (Op.h) — abstract base: `compute()`, `gradient()`, `getNewNode()`. Each operator is a global singleton (e.g. `extern MatMulOp matmul_op`).
- `Node` (Node.h) — graph vertex holding `vector<Node> input`, `Op* op`, cached `MatrixXd res`, and `hash_code` for topological sort. `operator*` and `operator+` sugar-build MatMul/MatAdd sub-graphs.
- `Variable(name)` — creates a Placeholder node. Used for weights, biases, inputs, and labels.
- `gradients(output_node, input_nodes)` — reverse-mode AD engine. Traverses graph in reverse topological order, building gradient expression nodes for each input.
- `Executor(eval_node_list)` — topologically sorts the merged forward+backward graph, runs `node.op->compute()` for each non-placeholder node given a `feed_dict`.

### Model API (Layer.h + Model.h)

Keras-like sequential API. Three layer types: `LAYER_INPUT`, `LAYER_DENSE`, `LAYER_OUTPUT`.

- Dense layers support `"relu"` or `"linear"` activation with He initialization
- Output layer: Dense + Softmax + cross-entropy gradient shortcut
- **AdamW** — enabled via `model.use_adam = true` (beta1/2, eps, weight decay configurable)
- **LR decay** — step-based: `lr *= lr_decay` every `lr_decay_step` iterations
- **Early stopping** — tracks `best_val_acc`, restores best weights at end
- `inference_executor` — forward-only executor for fast evaluation

### Data flow

`Model` → layers built → `compile()` calls `gradients()` → `Executor` + `inference_executor` → `loadData()` reads MNIST, splits 70/15/15 → `run()` loops: batch → `executor.run(feed_dict)` → AdamW/SGD update → periodic eval + CSV logging. Best weights restored.

MNIST data (`train.txt`): one sample per line, 784 pixels (0-255) + 10 one-hot labels. Data is used as flat (batch, 784) matrices for MLP.

## Unit Tests

8 separate test binaries in `unittest/`, each testing one operator with numerical gradient checking (finite differences). Run all with `make -f Makefile.simple test`.

## Training stability notes

- SGD update normalizes gradient by `batch_size`: `W -= (lr/batch_size) * dW`
- AdamW provides per-parameter adaptive learning rates with decoupled weight decay
- With He initialization, lr 0.001–0.01 works well for MLP
- Early stopping captures peak model before potential divergence
- `SoftmaxOp::compute()` uses stable softmax (subtract max before exp)
