"""
ntk_core.py
===========
JAX 0.7.2 原生实现的经验 NTK 计算核心。
无任何 neural_tangents 依赖。

设计原则
--------
- apply_fn(params, x) : pytree × 单样本 → (out_dim,) 标量输出
- 使用 jacrev（反向模式 Jacobian），适合 out_dim << n_params 的常见场景
- scalar (trace) NTK：K(x1,x2) = tr(J(x1) J(x2)^T) = ΣΣ ∂f_k/∂θ_p(x1) · ∂f_k/∂θ_p(x2)
  等价于把所有输出类的参数梯度拼接成一个长向量后取内积，与原 neural_tangents 行为一致

内存说明
--------
对于 n_params 个参数、out_dim 个输出类：
  - 单对样本的 Jacobian：2 × out_dim × n_params floats
  - ResNet18 (~11M 参数, 10 类) ≈ 880MB/对，需分块处理
  - 建议用 ntk_matrix_blocked() 而非 ntk_matrix()
"""

from functools import partial
from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax import jit, vmap, jacrev
from jax.flatten_util import ravel_pytree
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

def _make_flat_fn(apply_fn, params):
    """
    把 apply_fn(params_pytree, x) 改写成 apply_fn(flat_params_vector, x)，
    返回 (f_flat, flat_params, unravel_fn)。
    """
    flat_params, unravel = ravel_pytree(params)

    def f_flat(fp, x):
        return apply_fn(unravel(fp), x)   # (out_dim,)

    return f_flat, flat_params, unravel


# ─────────────────────────────────────────────────────────────────────────────
# 单对样本的 NTK（标量）
# ─────────────────────────────────────────────────────────────────────────────

def ntk_pair(apply_fn: Callable, params: Any, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """
    计算两个单样本之间的标量 NTK 值。

    K(x1, x2) = Σ_k <∇_θ f_k(x1), ∇_θ f_k(x2)>
              = vdot( J(x1), J(x2) )   其中 J ∈ R^{out_dim × n_params}

    Parameters
    ----------
    apply_fn : callable
        apply_fn(params, x) -> (out_dim,)，单样本前向传播。
    params : pytree
        网络参数（任意 JAX pytree 结构）。
    x1, x2 : jnp.ndarray
        两个单样本输入（不含 batch 维）。

    Returns
    -------
    jnp.ndarray
        标量 NTK 值 K(x1, x2)。
    """
    f_flat, fp, _ = _make_flat_fn(apply_fn, params)
    J1 = jacrev(f_flat)(fp, x1)   # (out_dim, n_params)
    J2 = jacrev(f_flat)(fp, x2)   # (out_dim, n_params)
    return jnp.vdot(J1, J2)       # 标量


# ─────────────────────────────────────────────────────────────────────────────
# 小规模：双重 vmap，适合能完整放入显存的数据集
# ─────────────────────────────────────────────────────────────────────────────

def ntk_matrix(apply_fn: Callable, params: Any,
               X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    计算完整 NTK 矩阵 K[i,j] = K(X1[i], X2[j])，shape (n1, n2)。

    内部一次性计算所有 Jacobian，适合 n × n_params 可放入显存的情形
    （例如 FNN on MNIST with n≤5000）。

    Parameters
    ----------
    apply_fn : callable
    params   : pytree
    X1       : (n1, ...) 数组
    X2       : (n2, ...) 数组

    Returns
    -------
    np.ndarray  shape (n1, n2)
    """
    f_flat, fp, _ = _make_flat_fn(apply_fn, params)

    # 批量 Jacobian：(n, out_dim, n_params)
    batch_jac = jit(vmap(jacrev(f_flat), in_axes=(None, 0)))

    J1 = batch_jac(fp, X1)   # (n1, out_dim, n_params)
    J2 = batch_jac(fp, X2)   # (n2, out_dim, n_params)

    # reshape 成 2D 再做 matmul，避免 einsum 创建 (n1,n2,out_dim,n_params) 中间张量
    n1 = J1.shape[0]
    n2 = J2.shape[0]
    K = J1.reshape(n1, -1) @ J2.reshape(n2, -1).T   # (n1, n2)
    return np.array(K)


# ─────────────────────────────────────────────────────────────────────────────
# 大规模：分块计算，复现原始项目的 block/column 策略
# ─────────────────────────────────────────────────────────────────────────────

def ntk_matrix_col_blocked(apply_fn: Callable, params: Any,
                            X1: np.ndarray, X2: np.ndarray,
                            col_block: int = 1) -> np.ndarray:
    """
    按列分块计算 NTK 矩阵，复现 ntk_generator.generate_cnn_ntk 的逐列策略。

    先一次性算出 X1 的全部 Jacobian（需要 n1 × out_dim × n_params 显存），
    再逐块处理 X2 列，降低瞬时峰值显存。

    col_block=1 时完全等价于原项目的 `for i in range(m): K[:,i:i+1]` 模式。
    """
    f_flat, fp, _ = _make_flat_fn(apply_fn, params)
    batch_jac = jit(vmap(jacrev(f_flat), in_axes=(None, 0)))

    n1, n2 = X1.shape[0], X2.shape[0]
    J1 = batch_jac(fp, X1)   # (n1, out_dim, n_params)，固定住

    K = np.zeros((n1, n2), dtype=np.float32)
    for j in range(0, n2, col_block):
        end = min(j + col_block, n2)
        J2_block = batch_jac(fp, X2[j:end])            # (bs, out_dim, n_params)
        K_block = jnp.einsum('ikp,jkp->ij', J1, J2_block)  # (n1, bs)
        K[:, j:end] = np.array(K_block)

    return K


def ntk_matrix_blocked(apply_fn: Callable, params: Any,
                        X: np.ndarray,
                        row_block: int = 10,
                        col_block: int = 10) -> np.ndarray:
    """
    双向分块计算自核矩阵 K[i,j] = K(X[i], X[j])，复现
    generate_cntk.generate_kernel() 的 block_size×block_size 策略。

    同时分块行和列，适合 n 很大（如 CIFAR-10 全集 60000 条）。

    Parameters
    ----------
    row_block / col_block : int
        每次处理的行/列数，调小以节省显存。
        显存估算：block × out_dim × n_params × 4 bytes
        CNN (3.4M params, out=10)：block=10 → ~1.36GB，block=50 → ~6.8GB
    """
    return ntk_matrix_blocked_rect(apply_fn, params, X, X, row_block, col_block)


def ntk_matrix_blocked_rect(apply_fn: Callable, params: Any,
                             X1: np.ndarray, X2: np.ndarray,
                             row_block: int = 10,
                             col_block: int = 10) -> np.ndarray:
    """
    双向分块计算矩形 NTK 矩阵 K[i,j] = K(X1[i], X2[j])，shape (n1, n2)。

    行和列都分块，峰值显存仅为单个块的 Jacobian：
        row_block × out_dim × n_params × 4 bytes（行块）
      + col_block × out_dim × n_params × 4 bytes（列块）

    这是处理大规模 CNN/ResNet NTK 的正确方法。

    Parameters
    ----------
    row_block : int  每次计算的 X1 行数
    col_block : int  每次计算的 X2 列数

    显存预算参考（CNN，3.4M 参数，out_dim=10，float32）
    -------------------------------------------------------
    block=1  →  ~136MB/块   最慢但最省显存
    block=5  →  ~680MB/块   T4 15GB 上约 10 块并行
    block=10 →  ~1.36GB/块  T4 推荐默认值
    block=50 →  ~6.8GB/块   T4 较激进，留余量给 PyTorch
    """
    f_flat, fp, _ = _make_flat_fn(apply_fn, params)
    batch_jac = jit(vmap(jacrev(f_flat), in_axes=(None, 0)))

    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2), dtype=np.float32)

    for i in range(0, n1, row_block):
        end_i = min(i + row_block, n1)
        Ji = batch_jac(fp, X1[i:end_i])              # (rb, out_dim, n_params)
        rb = end_i - i
        Ji_2d = Ji.reshape(rb, -1)                   # (rb, out_dim*n_params)
        for j in range(0, n2, col_block):
            end_j = min(j + col_block, n2)
            Jj = batch_jac(fp, X2[j:end_j])          # (cb, out_dim, n_params)
            cb = end_j - j
            Jj_2d = Jj.reshape(cb, -1)               # (cb, out_dim*n_params)
            # matmul 比 einsum 更省显存：不创建 (rb,cb,out,params) 中间张量
            K_block = Ji_2d @ Jj_2d.T                # (rb, cb)
            K[i:end_i, j:end_j] = np.array(K_block)

    return K


# ─────────────────────────────────────────────────────────────────────────────
# ResNet 专用：复现 generate_resnet_ntk 的 skip-batch 策略
# ─────────────────────────────────────────────────────────────────────────────

def ntk_matrix_skip(apply_fn: Callable, params: Any,
                    X1: np.ndarray, X2: np.ndarray,
                    skip: int = 25) -> np.ndarray:
    """
    按 skip 列批量计算，复现 generate_resnet_ntk(X, Y, skip=25) 的模式。
    使用双向分块：行块=skip，列块=skip。
    """
    return ntk_matrix_blocked_rect(apply_fn, params, X1, X2,
                                   row_block=skip, col_block=skip)
