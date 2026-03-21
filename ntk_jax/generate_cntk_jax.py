"""
generate_cntk_jax.py
====================
data_summarization/generate_cntk.py 的 JAX 0.7.2 原生重写。

原始文件依赖：
    from neural_tangents import stax
    from jax.api import jit

本文件用 JAX 原生代码替换所有 NTK 计算部分，数据加载部分（torchvision）保持不变。

架构对应
--------
原版：WideResnet(block_size=4, k=1, num_classes=10) via neural_tangents.stax
本版：wrn_init / wrn_fwd，完全等价的预激活 WideResNet，JAX 纯手写

generate_kernel(X) 接口保持不变：
  输入 X: (N, H, W, C)  NHWC 格式
  输出 K: (N, N) NTK 核矩阵，已保存为 data/kernel.npy

分块策略
--------
原版：for i,j in range(n//block_size): K[bi:ei, bj:ej] = kernel_fn(..., 'ntk')
本版：ntk_matrix_blocked(..., row_block=block_size, col_block=block_size)，完全等价

注意：原版 block_size=10 是为了适配 neural_tangents 的批量接口；
      JAX 原生版本可以根据 GPU 显存调整 block_size 以提升效率。
"""

import os
import numpy as np
import jax
from jax import random, jit
import jax.numpy as jnp

from models_jax import wrn_init, wrn_fwd
from ntk_core import ntk_matrix_blocked, ntk_matrix_blocked_rect

# ─────────────────────────────────────────────────────────────────────────────
# 模块级参数初始化（对应原版 init_fn, apply_fn, kernel_fn = WideResnet(...) 模块级定义）
# ─────────────────────────────────────────────────────────────────────────────

_KEY = random.PRNGKey(0)
_wrn_params = wrn_init(_KEY, block_size=4, k=1, num_classes=10)
_wrn_apply  = jit(wrn_fwd)


# ─────────────────────────────────────────────────────────────────────────────
# 核心接口（与原 generate_cntk.py 完全兼容）
# ─────────────────────────────────────────────────────────────────────────────

def generate_kernel(X: np.ndarray, block_size: int = 10) -> np.ndarray:
    """
    计算 WideResNet 的 NTK 矩阵 K[i,j] = K_NTK(X[i], X[j])。

    使用 ntk_matrix_blocked_rect 双向分块，峰值显存约：
        2 × block_size × out_dim × n_params × 4B
        WRN(k=1): n_params≈0.3M → block=10 → ~240MB，非常安全

    Parameters
    ----------
    X : np.ndarray  (N, H, W, C)，NHWC 格式
    block_size : int  行/列分块大小，默认 10
    """
    return ntk_matrix_blocked_rect(
        _wrn_apply, _wrn_params,
        X, X,
        row_block=block_size,
        col_block=block_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主脚本（复现原 generate_cntk.py 的完整流程）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 数据加载：与原版完全相同（保留 torchvision 依赖，这部分无需改动）
    import torch
    import torchvision.transforms as transforms
    import torchvision.datasets

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),
        download=True
    )

    n_train = train_dataset.data.shape[0]
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_train, shuffle=False)
    X_train, y_train = next(iter(loader))
    X_train, y_train = X_train.numpy(), y_train.numpy()

    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.data.shape[0], shuffle=False)
    X_test, y_test = next(iter(loader))
    X_test, y_test = X_test.numpy(), y_test.numpy()

    # 拼接并转换为 NHWC（与原版 .transpose(0,2,3,1) 一致）
    X = np.vstack([X_train, X_test]).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
    print(f"X shape: {X.shape}, dtype: {X.dtype}")

    # 生成核矩阵（与原版 generate_kernel(X) 接口一致）
    K = generate_kernel(X, block_size=10)
    print(f"K shape: {K.shape}, min: {K.min():.4f}, max: {K.max():.4f}")

    # 保存（与原版路径一致）
    os.makedirs('data', exist_ok=True)
    np.save('data/kernel.npy', K)
    print("Saved to data/kernel.npy")
