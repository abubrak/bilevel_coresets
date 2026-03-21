"""
test_ntk_jax.py
===============
验证 JAX 0.7.2 原生 NTK 重写的正确性。

测试内容
--------
1. FNN / CNN / ResNet / WideResNet 前向传播形状正确性
2. NTK pair：对称性与 PSD 对角元素（FNN）
3. NTK 矩阵：形状、对称性、PSD（FNN，n=8）
4. ntk_matrix / col_blocked / blocked 三种方式结果一致性（FNN）
5. generate_fnn_ntk / generate_cnn_ntk 接口可调用
   （ResNet NTK 跳过：~11M 参数，每样本 Jacobian ≈440MB，需要高内存 GPU）
6. bilevel_coreset kernel_fn_np 接口兼容性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

print(f"JAX version: {jax.__version__}")

from models_jax import (
    fnn_init, fnn_fwd,
    cnn_init, cnn_fwd,
    resnet_init, resnet_fwd,
    wrn_init, wrn_fwd,
)
from ntk_core import (
    ntk_pair,
    ntk_matrix,
    ntk_matrix_col_blocked,
    ntk_matrix_blocked,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m~\033[0m"
_results = []

def check(name, cond, detail=""):
    _results.append((name, cond))
    print(f"  {PASS if cond else FAIL} {name}" + (f"  [{detail}]" if detail else ""))
    return cond

def skip(name, reason=""):
    print(f"  {SKIP} SKIP {name}  [{reason}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 前向传播形状
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Forward pass shapes")
key = random.PRNGKey(42)

fnn_p = fnn_init(key, input_dim=784, hidden_dim=100, output_dim=10)
check("FNN (784,)→(10,)", fnn_fwd(fnn_p, jnp.ones((784,))).shape == (10,))

cnn_p = cnn_init(key, in_channels=1, input_hw=(28, 28), output_dim=10)
check("CNN (28,28,1)→(10,)", cnn_fwd(cnn_p, jnp.ones((28, 28, 1))).shape == (10,))

resnet_p = resnet_init(key, block_size=2, num_classes=10)
check("ResNet (32,32,3)→(10,)", resnet_fwd(resnet_p, jnp.ones((32, 32, 3))).shape == (10,))
del resnet_p  # 释放大参数

wrn_p = wrn_init(key, block_size=4, k=1, num_classes=10)
check("WRN (32,32,3)→(10,)", wrn_fwd(wrn_p, jnp.ones((32, 32, 3))).shape == (10,))
del wrn_p


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NTK pair 对称性 & PSD 对角
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] NTK pair (FNN)")
k1, k2 = random.split(key)
x1 = random.normal(k1, (784,))
x2 = random.normal(k2, (784,))

k12 = ntk_pair(fnn_fwd, fnn_p, x1, x2)
k21 = ntk_pair(fnn_fwd, fnn_p, x2, x1)
check("K(x1,x2)==K(x2,x1)", abs(float(k12)-float(k21)) < 1e-4,
      f"|diff|={abs(float(k12)-float(k21)):.2e}")
check("K(x,x)>=0", float(ntk_pair(fnn_fwd, fnn_p, x1, x1)) >= 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NTK 矩阵性质（FNN，n=8）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] NTK matrix properties (FNN, n=8)")
rng = np.random.default_rng(0)
X8 = rng.standard_normal((8, 784)).astype(np.float32)
K  = ntk_matrix(fnn_fwd, fnn_p, X8, X8)

check("shape (8,8)",   K.shape == (8, 8), str(K.shape))
check("all finite",    np.isfinite(K).all())
sym_err = np.max(np.abs(K - K.T))
check("symmetric",     sym_err < 1e-3, f"{sym_err:.2e}")
min_eig = np.linalg.eigvalsh(K).min()
check("PSD min_eig>=-1e-4", min_eig >= -1e-4, f"{min_eig:.4e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 三种计算方式一致性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Consistency across compute strategies (FNN)")
X6 = rng.standard_normal((6, 784)).astype(np.float32)
X4 = rng.standard_normal((4, 784)).astype(np.float32)

K_full = ntk_matrix(fnn_fwd, fnn_p, X6, X4)
K_c1   = ntk_matrix_col_blocked(fnn_fwd, fnn_p, X6, X4, col_block=1)
K_c2   = ntk_matrix_col_blocked(fnn_fwd, fnn_p, X6, X4, col_block=2)
check("col_blocked(1) matches", np.max(np.abs(K_full-K_c1)) < 1e-3,
      f"{np.max(np.abs(K_full-K_c1)):.2e}")
check("col_blocked(2) matches", np.max(np.abs(K_full-K_c2)) < 1e-3,
      f"{np.max(np.abs(K_full-K_c2)):.2e}")

X8b   = rng.standard_normal((8, 784)).astype(np.float32)
K_ref = ntk_matrix(fnn_fwd, fnn_p, X8b, X8b)
K_blk = ntk_matrix_blocked(fnn_fwd, fnn_p, X8b, row_block=3, col_block=4)
check("blocked(3,4) matches", np.max(np.abs(K_ref-K_blk)) < 1e-3,
      f"{np.max(np.abs(K_ref-K_blk)):.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. generate_* 接口
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] generate_* interface")
from ntk_generator_jax import generate_fnn_ntk, generate_cnn_ntk

Xf = rng.standard_normal((5, 784)).astype(np.float32)
Yf = rng.standard_normal((3, 784)).astype(np.float32)
Kf = generate_fnn_ntk(Xf, Yf)
check("generate_fnn_ntk shape (5,3)", Kf.shape == (5,3), str(Kf.shape))
check("generate_fnn_ntk finite",      np.isfinite(Kf).all())
check("generate_fnn_ntk is ndarray",  isinstance(Kf, np.ndarray))

Xc = rng.standard_normal((2, 28, 28, 1)).astype(np.float32)
Yc = rng.standard_normal((2, 28, 28, 1)).astype(np.float32)
Kc = generate_cnn_ntk(Xc, Yc)
check("generate_cnn_ntk shape (2,2)", Kc.shape == (2,2), str(Kc.shape))
check("generate_cnn_ntk finite",      np.isfinite(Kc).all())

skip("generate_resnet_ntk", "Jacobian ~440MB/sample, needs high-mem GPU")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. bilevel_coreset kernel_fn_np 接口兼容性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6] bilevel_coreset kernel_fn_np compatibility")
Xdata = rng.standard_normal((12, 784)).astype(np.float32)
selected = np.array([0, 3, 7])

K_XS = generate_fnn_ntk(Xdata, Xdata[selected])   # (12, 3)
check("K_X_S shape (12,3)",  K_XS.shape == (12,3), str(K_XS.shape))
check("K_X_S is ndarray",    isinstance(K_XS, np.ndarray))

K_SS = K_XS[selected]                              # (3, 3)
check("K_S_S shape (3,3)",   K_SS.shape == (3,3))
check("K_S_S symmetric",     np.max(np.abs(K_SS - K_SS.T)) < 1e-3,
      f"{np.max(np.abs(K_SS-K_SS.T)):.2e}")

try:
    import torch
    K_t = torch.from_numpy(K_XS).float()
    check("torch.from_numpy OK", K_t.shape == (12, 3))
except ImportError:
    skip("torch.from_numpy", "torch not installed")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
passed = sum(1 for _, r in _results if r)
total  = len(_results)
print(f"Result: {passed}/{total} passed")
if passed == total:
    print("\033[32mAll tests passed!\033[0m")
else:
    print(f"\033[31mFailed: {[n for n,r in _results if not r]}\033[0m")
    sys.exit(1)
