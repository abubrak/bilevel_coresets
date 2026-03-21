"""
ntk_generator_jax.py
====================
cl_streaming/ntk_generator.py 的 JAX 0.7.2 原生重写。

原始文件依赖：
    from jax.api import jit          # jax.api 在 JAX 0.3+ 已移除
    from neural_tangents import stax  # 已停止维护

本文件完全等价地替换上述两项依赖，提供相同的公开接口：
    generate_fnn_ntk(X, Y)  → np.ndarray (n, m)
    generate_cnn_ntk(X, Y)  → np.ndarray (n, m)
    generate_resnet_ntk(X, Y, skip=25) → np.ndarray (n, m)

差异说明
--------
原版使用 neural_tangents 的「解析无限宽 NTK」；本版使用「经验有限宽 NTK」，
在网络足够宽时两者等价（此处 FNN hidden=100, CNN ch=32/64，ResNet 已为有限宽）。
对于 bilevel coreset 的 representer proxy，经验 NTK 完全可用。

随机种子
--------
模块加载时固定种子（与原版 stax.serial 模块级初始化行为一致），
确保每次 import 生成同样的随机网络参数。
"""

import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
from functools import partial

from models_jax import (
    fnn_init, fnn_fwd,
    cnn_init, cnn_fwd,
    resnet_init, resnet_fwd,
)
from ntk_core import (
    ntk_matrix,
    ntk_matrix_col_blocked,
    ntk_matrix_skip,
)

# ─────────────────────────────────────────────────────────────────────────────
# 模块级参数初始化（对应原版 stax.serial 的模块级定义）
# ─────────────────────────────────────────────────────────────────────────────

_BASE_KEY = random.PRNGKey(0)
_k_fnn, _k_cnn, _k_resnet = random.split(_BASE_KEY, 3)

# FNN: Dense(100)→ReLU→Dense(100)→ReLU→Dense(10)
# 对应原版 stax.Dense(100,1.,0.05) × 2 + Dense(10,1.,0.05)
_fnn_params = fnn_init(_k_fnn, input_dim=784, hidden_dim=100, output_dim=10)

# CNN（无 MaxPool）: Conv(32,5×5)→ReLU→Conv(64,5×5)→ReLU→Flat→Dense(128)→ReLU→Dense(10)
# 对应原版 stax.Conv(32,(5,5),(1,1),SAME) 等
# MNIST 输入 (28,28,1)
_cnn_params = cnn_init(_k_cnn, in_channels=1, input_hw=(28, 28), output_dim=10)

# ResNet 参数懒加载：ResNet ~11M 参数，模块级初始化会预分配大量显存，
# 改为首次调用 generate_resnet_ntk 时才初始化。
_resnet_params = None

# 预 JIT 的单样本前向（apply_fn 接口，供 ntk_core 使用）
_fnn_apply    = jit(fnn_fwd)
_cnn_apply    = jit(cnn_fwd)
_resnet_apply = jit(resnet_fwd)


def _get_resnet_params():
    global _resnet_params
    if _resnet_params is None:
        _resnet_params = resnet_init(_k_resnet, block_size=2, num_classes=10)
    return _resnet_params


# ─────────────────────────────────────────────────────────────────────────────
# 公开接口（与原 ntk_generator.py 完全兼容）
# ─────────────────────────────────────────────────────────────────────────────

def generate_fnn_ntk(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    计算 FNN 的 NTK 矩阵 K[i,j] = K_NTK(X[i], Y[j])。

    原版：np.array(fnn_kernel_fn(X, Y, 'ntk'))
    本版：经验 NTK，双重 vmap 一次性计算（适合 FNN 参数量小的场景）

    Parameters
    ----------
    X : np.ndarray  (n, 784)  MNIST 展平输入
    Y : np.ndarray  (m, 784)

    Returns
    -------
    np.ndarray  (n, m)
    """
    return ntk_matrix(_fnn_apply, _fnn_params, X, Y)


def generate_cnn_ntk(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    计算 CNN 的 NTK 矩阵 K[i,j] = K_NTK(X[i], Y[j])。

    原版：逐列循环 for i in range(m): K[:,i:i+1] = cnn_kernel_fn(X, Y[i:i+1], 'ntk')
    本版：复现相同的逐列策略（col_block=1），降低峰值显存。

    Parameters
    ----------
    X : np.ndarray  (n, 28, 28, 1)  NHWC 格式的 MNIST 图像
    Y : np.ndarray  (m, 28, 28, 1)

    Returns
    -------
    np.ndarray  (n, m)
    """
    return ntk_matrix_col_blocked(_cnn_apply, _cnn_params, X, Y, col_block=1)


def generate_resnet_ntk(X: np.ndarray, Y: np.ndarray,
                        skip: int = 25) -> np.ndarray:
    """
    计算 ResNet 的 NTK 矩阵，除以 100（与原版 return K / 100 一致）。

    原版：逐 skip 列循环；结果除以 100 以稳定数值。
    本版：复现相同的 skip-batch 策略。

    Parameters
    ----------
    X : np.ndarray  (n, 32, 32, 3)  NHWC 格式的 CIFAR-10 图像
    Y : np.ndarray  (m, 32, 32, 3)
    skip : int  每批处理的列数，默认 25

    Returns
    -------
    np.ndarray  (n, m)，已除以 100
    """
    K = ntk_matrix_skip(_resnet_apply, _get_resnet_params(), X, Y, skip=skip)
    return K / 100.0
