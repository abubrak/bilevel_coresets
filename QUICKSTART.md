# bilevel_coresets — JAX 原生重写快速上手

## 目录结构

```
./
├── run_pipeline.py              ← 主运行脚本（本文件描述的入口）
├── ntk_jax/                     ← JAX 原生 NTK 替换实现
│   ├── ntk_core.py              ← 通用 NTK 计算引擎
│   ├── models_jax.py            ← FNN/CNN/ResNet/WideResNet
│   ├── ntk_generator_jax.py     ← 替换 cl_streaming/ntk_generator.py
│   ├── generate_cntk_jax.py     ← 替换 data_summarization/generate_cntk.py
│   └── test_ntk_jax.py          ← 单元测试（22/22 通过）
└── bilevel_coresets/            ← 原始项目（保持不变，仅替换 NTK 相关导入）
```

---

## 环境安装

```bash
# JAX（必须）
pip install "jax[cpu]==0.7.2"

# 科学计算（必须）
pip install numpy scipy scikit-learn

# 实验 2/3/4 需要（可选）
pip install torch torchvision
```

---

## 运行方式

### 实验 1：玩具回归（无需 torch，最快验证）

```bash
python run_pipeline.py --exp regression
```

演示多项式核 Coreset vs 均匀采样的 MSE 对比，以及 JAX NTK 接口的兼容性。

---

### 实验 2：MNIST 数据摘要

```bash
# Coreset 方法（使用 CNN NTK 作为代理模型）
python run_pipeline.py --exp mnist --method coreset --coreset_size 100 --seed 0

# 对照：均匀采样
python run_pipeline.py --exp mnist --method uniform --coreset_size 100 --seed 0
```

对应原始文件 `data_summarization/cnn_mnist.py`。

**关键替换**：
```python
# 原版
from cl_streaming import ntk_generator
kernel_fn = lambda x, y: ntk_generator.generate_cnn_ntk(...)

# 本版
from ntk_generator_jax import generate_cnn_ntk
kernel_fn = lambda x, y: generate_cnn_ntk(...)
```

结果保存至 `results/mnist/{method}_{size}_{seed}.json`。

---

### 实验 3：持续学习

```bash
# Split-MNIST，Coreset 方法
python run_pipeline.py --exp cl --dataset splitmnist --method coreset \
    --buffer_size 100 --beta 1.0 --seed 0 --nr_epochs 10

# Permuted-MNIST，均匀采样（对照）
python run_pipeline.py --exp cl --dataset permmnist --method uniform \
    --buffer_size 100 --seed 0
```

对应原始文件 `cl_streaming/cl.py`。

**关键替换**：
```python
# 原版（已失效）
from jax.api import jit             # jax.api 在 JAX 0.3+ 已移除
from neural_tangents import stax    # 已停止维护

if dataset == 'permmnist':
    return lambda x, y: ntk_generator.generate_fnn_ntk(...)
else:
    return lambda x, y: ntk_generator.generate_cnn_ntk(...)

# 本版（JAX 0.7.2 原生）
from jax import jit                 # ✓
from ntk_generator_jax import generate_fnn_ntk, generate_cnn_ntk  # ✓
```

结果保存至 `results/cl/{dataset}_{method}_{buffer}_{seed}.json`。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--dataset` | `splitmnist` / `permmnist` | `splitmnist` |
| `--buffer_size` | 记忆 buffer 容量 | `100` |
| `--beta` | 历史任务损失权重 | `1.0` |
| `--nr_epochs` | 每任务训练轮数 | `10`（原论文更大）|
| `--samples_per_task` | 每任务样本数 | `200`（原论文更大）|

---

### 实验 4：批量主动学习（Nystrom 代理）

```bash
python run_pipeline.py --exp active --coreset_size 20 --nystrom_dim 500 --seed 0
```

对应原始文件 `batch_active_learning/nystrom_example.py`。

**关键替换**：
```python
# 原版（已失效）
from jax.api import jit
from neural_tangents import stax
_, _, kernel_fn = WideResnet(block_size=4, k=1, num_classes=10)
kernel_fn = jit(kernel_fn, static_argnums=(2,))
def kernel_fn_ntk(x, y, step=64):
    return np.array(kernel_fn(x_nhwc, y_nhwc, 'ntk'))

# 本版（JAX 0.7.2 原生）
from generate_cntk_jax import generate_kernel  # WRN JAX 原生实现
```

结果保存至 `results/active/nystrom_{size}_{seed}.json`。

---

### 一键运行所有实验

```bash
python run_pipeline.py --exp all
```

---

## 核心替换速查表

| 原始代码 | 本版替换 | 原因 |
|---|---|---|
| `from jax.api import jit` | `from jax import jit` | `jax.api` 在 JAX 0.3+ 已移除 |
| `from neural_tangents import stax` | `from models_jax import *` | neural_tangents 已归档 |
| `stax.Dense(100,1.,0.05)` | `fnn_init(key, ...)` | JAX 原生等价实现 |
| `stax.Conv(32,(5,5),...)` | `cnn_init(key, ...)` | JAX 原生等价实现 |
| `kernel_fn(X, Y, 'ntk')` | `ntk_matrix(apply_fn, params, X, Y)` | 经验 NTK，双重 vmap |
| `generate_fnn_ntk(X, Y)` | `generate_fnn_ntk(X, Y)` | **接口不变**，底层替换 |
| `generate_cnn_ntk(X, Y)` | `generate_cnn_ntk(X, Y)` | **接口不变**，底层替换 |
| `generate_resnet_ntk(X, Y, skip)` | `generate_resnet_ntk(X, Y, skip)` | **接口不变**，底层替换 |
| `generate_kernel(X)` | `generate_kernel(X)` | **接口不变**，底层替换 |

---

## 验证单元测试

```bash
cd ntk_jax && python test_ntk_jax.py
# Result: 22/22 passed
```
