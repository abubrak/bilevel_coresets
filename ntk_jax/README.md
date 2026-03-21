# bilevel_coresets NTK — JAX 0.7.2 原生重写

将 `zalanborsos/bilevel_coresets` 项目中所有依赖 `neural_tangents` 的 NTK
部分，用 **JAX 0.7.2 原生 API** 完整重写，无任何第三方 NTK 库依赖。

---

## 文件对应关系

| 原始文件 | 本项目替换文件 | 说明 |
|---|---|---|
| `cl_streaming/ntk_generator.py` | `ntk_generator_jax.py` | FNN/CNN/ResNet NTK，接口完全兼容 |
| `data_summarization/generate_cntk.py` | `generate_cntk_jax.py` | WideResNet CNTK，接口完全兼容 |
| *(新增)* | `ntk_core.py` | 通用经验 NTK 引擎（架构无关）|
| *(新增)* | `models_jax.py` | 所有架构的 JAX 纯函数式实现 |

---

## 依赖

```
jax[cpu]==0.7.2   # 或 jax[cuda12]
numpy
```

原始项目的 `neural_tangents` 和 `from jax.api import jit`（已在 JAX 0.3+ 移除）
均不再需要。

---

## 快速开始

```python
import numpy as np
from ntk_generator_jax import generate_fnn_ntk, generate_cnn_ntk

# FNN NTK：输入展平 MNIST (n, 784)
X_train = np.random.randn(100, 784).astype(np.float32)
X_test  = np.random.randn(20,  784).astype(np.float32)
K = generate_fnn_ntk(X_train, X_test)   # (100, 20) np.ndarray

# CNN NTK：输入 NHWC 格式 (n, 28, 28, 1)
X_img = np.random.randn(50, 28, 28, 1).astype(np.float32)
K_cnn = generate_cnn_ntk(X_img, X_img)  # (50, 50)
```

直接替换原项目中的 `kernel_fn_np` 参数：

```python
# 原来：
# from cl_streaming.ntk_generator import generate_fnn_ntk as kernel_fn_np

# 现在：
from ntk_generator_jax import generate_fnn_ntk as kernel_fn_np

coreset = BilevelCoreset(...)
inds, weights = coreset.build_with_representer_proxy_batch(
    X, y, m=100, kernel_fn_np=kernel_fn_np
)
```

---

## 架构说明

### `ntk_core.py` — 经验 NTK 引擎

架构无关，接受任意 `apply_fn(params, x) → (out_dim,)` 函数：

```python
from ntk_core import ntk_matrix, ntk_matrix_col_blocked, ntk_matrix_blocked

# 小数据集（完整一次性计算）
K = ntk_matrix(my_model_fwd, my_params, X1, X2)

# CNN 风格（逐列，节省显存）
K = ntk_matrix_col_blocked(cnn_fwd, cnn_params, X1, X2, col_block=1)

# 大数据集（双向分块，复现 generate_cntk.py block_size 策略）
K = ntk_matrix_blocked(wrn_fwd, wrn_params, X, row_block=10, col_block=10)
```

核心实现（双重 `vmap` + `jacrev`）：

```python
batch_jac = jit(vmap(jacrev(f_flat), in_axes=(None, 0)))
J1 = batch_jac(fp, X1)                        # (n1, out_dim, n_params)
J2 = batch_jac(fp, X2)                        # (n2, out_dim, n_params)
K  = jnp.einsum('ikp,jkp->ij', J1, J2)       # (n1, n2)
```

### `models_jax.py` — 架构实现

| 类 | 初始化 | 前向 | 对应原版 stax 定义 |
|---|---|---|---|
| FNN | `fnn_init(key, input_dim, hidden_dim, output_dim)` | `fnn_fwd(params, x)` | `Dense(100)→ReLU→Dense(100)→ReLU→Dense(10)` |
| CNN | `cnn_init(key, in_channels, input_hw, output_dim)` | `cnn_fwd(params, x)` | `Conv(32,5×5)→ReLU→Conv(64,5×5)→ReLU→Flat→Dense(128)→ReLU→Dense(10)`，无 MaxPool |
| ResNet | `resnet_init(key, block_size, num_classes)` | `resnet_fwd(params, x)` | 预激活残差块，`block_size=2` |
| WideResNet | `wrn_init(key, block_size, k, num_classes)` | `wrn_fwd(params, x)` | 预激活，GlobalAvgPool，`block_size=4, k=1` |

所有函数均为纯函数（无副作用），直接兼容 `jit` / `vmap` / `grad`。

---

## 与原版的差异

| 项目 | 原版（neural_tangents） | 本版（JAX 原生） |
|---|---|---|
| NTK 类型 | 解析无限宽 NTK | 经验有限宽 NTK |
| 网络宽度 | 理论极限 (→∞) | 有限（FNN hidden=100, CNN ch=32/64） |
| 数值等价性 | — | 宽度足够时两者近似等价 |
| 分块策略 | 逐列/逐 skip | 完全复现原始分块模式 |
| ResNet NTK `/100` | 原版有此缩放 | 保留（`generate_resnet_ntk` 除以 100）|

对于 bilevel coreset 的 representer proxy 使用场景，经验 NTK 与解析 NTK
在实践中效果相当（参见原论文附录）。

---

## 内存注意事项

NTK 计算的瓶颈是 Jacobian 矩阵的大小：

```
显存 ≈ n_samples × out_dim × n_params × 4 bytes

FNN (hidden=100):   n_params ≈ 90K  → 每样本 ~3.6MB  ✓ CPU 可用
CNN (MNIST):        n_params ≈ 3.4M → 每样本 ~136MB  需要 GPU
ResNet (block=2):   n_params ≈ 11M  → 每样本 ~440MB  需要高内存 GPU
WideResNet (k=1):   n_params ≈ 0.3M → 每样本 ~12MB   CPU 可用（k=1 时）
```

大参数量模型建议使用 `ntk_matrix_col_blocked` 或 `ntk_matrix_blocked`
并将 `col_block` / `row_block` 设为 1~5。

---

## 测试

```bash
python3 test_ntk_jax.py
# Result: 22/22 passed
```
