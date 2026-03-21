"""
models_jax.py
=============
JAX 0.7.2 原生实现的网络架构，对应 bilevel_coresets 项目中所有 NTK 代理模型。

架构说明（严格对齐原 neural_tangents stax 定义）
------------------------------------------------
FNN      : Dense(100)→ReLU→Dense(100)→ReLU→Dense(10)
CNN      : Conv(32,5×5,SAME)→ReLU→Conv(64,5×5,SAME)→ReLU→Flatten→Dense(128)→ReLU→Dense(10)
           无 MaxPool（与 torch ConvNet 不同，与 stax 版一致）
ResNet   : Conv(64,3×3)→Group(2,64)→Group(2,128,↓2)→Group(2,256,↓2)→Group(2,512,↓2)→Flat→Dense(10)
           预激活残差块 (ReLU→Conv→ReLU→Conv)，shortcut 为 Conv 或 Identity
WideResNet: Conv(16,3×3)→WGroup(4,16)→WGroup(4,32,↓2)→WGroup(4,64,↓2)→GlobalAvgPool→Dense(10)
            同为预激活结构，k=1（宽度倍率）

初始化方案（与 neural_tangents 保持一致）
-----------------------------------------
Dense  : W ~ N(0, W_std²/fan_in)，b ~ N(0, b_std²)
Conv   : W ~ N(0, W_std²/fan_in) 其中 fan_in = kH×kW×C_in，b ~ N(0, b_std²)
最终 Dense 层某些情况下 b_std=0（generate_cntk.py 中的 WideResNet）

数据格式
--------
所有卷积使用 JAX 默认的 NHWC（channels last），与 neural_tangents 一致。
单样本输入形状：FNN (D,)，CNN/ResNet/WRN (H,W,C)。
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Optional, List, Dict, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
# 基础层：初始化 & 前向
# ─────────────────────────────────────────────────────────────────────────────

def dense_init(key: Any, in_dim: int, out_dim: int,
               W_std: float = 1.0, b_std: float = 0.05) -> Tuple:
    """Dense 层参数初始化，NTK 参数化：W~N(0, W_std²/in_dim)。"""
    k1, k2 = random.split(key)
    W = random.normal(k1, (in_dim, out_dim)) * (W_std / jnp.sqrt(in_dim))
    b = random.normal(k2, (out_dim,)) * b_std
    return (W, b)


def conv_init(key: Any, kH: int, kW: int, C_in: int, C_out: int,
              W_std: float = 1.0, b_std: float = 0.05) -> Tuple:
    """Conv 层参数初始化，W shape: (kH, kW, C_in, C_out)（HWIO）。"""
    k1, k2 = random.split(key)
    fan_in = kH * kW * C_in
    W = random.normal(k1, (kH, kW, C_in, C_out)) * (W_std / jnp.sqrt(fan_in))
    b = random.normal(k2, (C_out,)) * b_std
    return (W, b)


def dense_fwd(params: Tuple, x: jnp.ndarray) -> jnp.ndarray:
    """Dense 前向：x @ W + b。x: (..., in_dim) → (..., out_dim)。"""
    W, b = params
    return x @ W + b


def conv_fwd(params: Tuple, x: jnp.ndarray,
             strides: Tuple[int, int] = (1, 1),
             padding: str = 'SAME') -> jnp.ndarray:
    """
    单样本 Conv 前向：x (H,W,C_in) → (H',W',C_out)。
    内部临时加/去 batch 维以符合 jax.lax.conv_general_dilated 接口。
    """
    W, b = params  # W: (kH, kW, C_in, C_out)
    y = jax.lax.conv_general_dilated(
        x[None],             # (1, H, W, C_in)
        W,                   # (kH, kW, C_in, C_out)
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )[0]                     # (H', W', C_out)
    return y + b             # 广播 bias (C_out,)


# ─────────────────────────────────────────────────────────────────────────────
# FNN
# Dense(100)→ReLU→Dense(100)→ReLU→Dense(10)
# 对应 ntk_generator.py: stax.Dense(100)→Relu→Dense(100)→Relu→Dense(10)
# ─────────────────────────────────────────────────────────────────────────────

def fnn_init(key: Any, input_dim: int = 784,
             hidden_dim: int = 100, output_dim: int = 10) -> List:
    """
    FNN 参数初始化。返回 list of (W, b)，从输入到输出共 3 层。
    W_std=1.0, b_std=0.05 与 ntk_generator.py 的 stax.Dense(100,1.,0.05) 一致。
    """
    k1, k2, k3 = random.split(key, 3)
    return [
        dense_init(k1, input_dim,  hidden_dim, W_std=1., b_std=0.05),
        dense_init(k2, hidden_dim, hidden_dim, W_std=1., b_std=0.05),
        dense_init(k3, hidden_dim, output_dim, W_std=1., b_std=0.05),
    ]


def fnn_fwd(params: List, x: jnp.ndarray) -> jnp.ndarray:
    """FNN 单样本前向。x: (input_dim,) → (output_dim,)。"""
    x = jax.nn.relu(dense_fwd(params[0], x))
    x = jax.nn.relu(dense_fwd(params[1], x))
    x = dense_fwd(params[2], x)
    return x                 # (output_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# CNN
# Conv(32,5×5,SAME)→ReLU→Conv(64,5×5,SAME)→ReLU→Flatten→Dense(128)→ReLU→Dense(10)
# 无 MaxPool，与 ntk_generator.py stax 定义严格一致
# ─────────────────────────────────────────────────────────────────────────────

def _cnn_flat_dim(H: int, W: int, C: int = 1) -> int:
    """SAME padding 不改变空间尺寸，conv2 输出 64 通道。"""
    return H * W * 64


def cnn_init(key: Any, in_channels: int = 1,
             input_hw: Tuple[int, int] = (28, 28),
             output_dim: int = 10) -> Dict:
    """
    CNN 参数初始化。
    input_hw: (H, W) 用于计算 flatten 后的维度。
    """
    k1, k2, k3, k4 = random.split(key, 4)
    flat_dim = _cnn_flat_dim(input_hw[0], input_hw[1])
    return {
        'conv1':  conv_init(k1, 5, 5, in_channels, 32,  W_std=1., b_std=0.05),
        'conv2':  conv_init(k2, 5, 5, 32,           64,  W_std=1., b_std=0.05),
        'dense1': dense_init(k3, flat_dim, 128,          W_std=1., b_std=0.05),
        'dense2': dense_init(k4, 128,      output_dim,   W_std=1., b_std=0.05),
    }


def cnn_fwd(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    """CNN 单样本前向。x: (H, W, C) → (output_dim,)。"""
    x = jax.nn.relu(conv_fwd(params['conv1'], x))   # (H, W, 32)
    x = jax.nn.relu(conv_fwd(params['conv2'], x))   # (H, W, 64)
    x = x.ravel()                                    # (H*W*64,)
    x = jax.nn.relu(dense_fwd(params['dense1'], x)) # (128,)
    x = dense_fwd(params['dense2'], x)               # (output_dim,)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# ResNet（用于 cl_streaming/ntk_generator.py）
# 预激活残差块，block_size=2，5 个卷积阶段
# ─────────────────────────────────────────────────────────────────────────────

def _resnet_block_init(key: Any, in_ch: int, out_ch: int,
                       channel_mismatch: bool = False) -> Dict:
    """
    单个预激活残差块参数。
    Main: ReLU→Conv(out_ch,3×3,strides)→ReLU→Conv(out_ch,3×3,1×1)
    Shortcut: Conv(out_ch,3×3,strides) 若 channel_mismatch，否则 None
    strides 在 forward 时传入，init 只存权重形状。
    """
    k1, k2, k3 = random.split(key, 3)
    return {
        'conv1':    conv_init(k1, 3, 3, in_ch,  out_ch, W_std=1., b_std=0.05),
        'conv2':    conv_init(k2, 3, 3, out_ch, out_ch, W_std=1., b_std=0.05),
        'shortcut': conv_init(k3, 3, 3, in_ch,  out_ch, W_std=1., b_std=0.05)
                    if channel_mismatch else None,
    }


def _resnet_group_init(key: Any, block_size: int,
                       in_ch: int, out_ch: int) -> List:
    """一个 ResNet group = block_size 个 block，首块含 shortcut conv。"""
    keys = random.split(key, block_size)
    blocks = [_resnet_block_init(keys[0], in_ch, out_ch, channel_mismatch=True)]
    for i in range(1, block_size):
        blocks.append(_resnet_block_init(keys[i], out_ch, out_ch, channel_mismatch=False))
    return blocks


def resnet_init(key: Any, block_size: int = 2,
                num_classes: int = 10) -> Dict:
    """
    完整 ResNet 参数初始化。
    对应 ntk_generator.py: Resnet(block_size=2, num_classes=10)。
    输入预期 CIFAR-10 风格 (32,32,3)。
    """
    k0, k1, k2, k3, k4, k5 = random.split(key, 6)
    # 初始 conv 后 spatial 仍为 32×32，经 4 组 stride-2 后→4×4×512=8192
    flat_dim = 4 * 4 * 512
    return {
        'conv0':   conv_init(k0, 3, 3, 3, 64, W_std=1., b_std=0.05),
        'group1':  _resnet_group_init(k1, block_size, 64,  64),
        'group2':  _resnet_group_init(k2, block_size, 64,  128),
        'group3':  _resnet_group_init(k3, block_size, 128, 256),
        'group4':  _resnet_group_init(k4, block_size, 256, 512),
        'dense':   dense_init(k5, flat_dim, num_classes, W_std=1., b_std=0.05),
    }


def _resnet_block_fwd(params: Dict, x: jnp.ndarray,
                      strides: Tuple[int, int] = (1, 1)) -> jnp.ndarray:
    """单个预激活残差块前向。"""
    # Main: ReLU → conv1(strides) → ReLU → conv2(1,1)
    h = jax.nn.relu(x)
    h = conv_fwd(params['conv1'], h, strides=strides)
    h = jax.nn.relu(h)
    h = conv_fwd(params['conv2'], h, strides=(1, 1))
    # Shortcut
    sc = conv_fwd(params['shortcut'], x, strides=strides) \
         if params['shortcut'] is not None else x
    return h + sc


def _resnet_group_fwd(block_params: List, x: jnp.ndarray,
                      strides: Tuple[int, int] = (1, 1)) -> jnp.ndarray:
    """Group 前向：首块用给定 strides，后续块用 (1,1)。"""
    x = _resnet_block_fwd(block_params[0], x, strides=strides)
    for bp in block_params[1:]:
        x = _resnet_block_fwd(bp, x, strides=(1, 1))
    return x


def resnet_fwd(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    ResNet 单样本前向。x: (H,W,C) → (num_classes,)。
    对应 ntk_generator.py Resnet(block_size=2) 的前向逻辑。
    """
    x = conv_fwd(params['conv0'], x)                           # (32,32,64)
    x = _resnet_group_fwd(params['group1'], x, strides=(1, 1)) # (32,32,64)
    x = _resnet_group_fwd(params['group2'], x, strides=(2, 2)) # (16,16,128)
    x = _resnet_group_fwd(params['group3'], x, strides=(2, 2)) # (8,8,256)
    x = _resnet_group_fwd(params['group4'], x, strides=(2, 2)) # (4,4,512)
    x = x.ravel()                                               # (8192,)
    x = dense_fwd(params['dense'], x)                          # (num_classes,)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# WideResNet（用于 data_summarization/generate_cntk.py）
# block_size=4, k=1, 预激活，GlobalAvgPool，最终 Dense 无 bias
# ─────────────────────────────────────────────────────────────────────────────

def _wrn_block_init(key: Any, in_ch: int, out_ch: int,
                    channel_mismatch: bool = False) -> Dict:
    """
    WideResNet 预激活块，与 ResNet 块结构相同。
    对应 generate_cntk.py WideResnetBlock。
    """
    return _resnet_block_init(key, in_ch, out_ch, channel_mismatch)


def _wrn_group_init(key: Any, block_size: int,
                    in_ch: int, out_ch: int) -> List:
    """WideResNet Group，与 ResNet group 结构相同。"""
    return _resnet_group_init(key, block_size, in_ch, out_ch)


def wrn_init(key: Any, block_size: int = 4, k: int = 1,
             num_classes: int = 10) -> Dict:
    """
    WideResNet 参数初始化（b_std=0 in final Dense，与原代码 stax.Dense(10,1.,0.) 一致）。
    对应 generate_cntk.py: WideResnet(block_size=4, k=1, num_classes=10)。
    输入 CIFAR-10 (32,32,3)，经 GlobalAvgPool 后 flat_dim = 64*k。
    """
    k0, k1, k2, k3, k4 = random.split(key, 5)
    ch1, ch2, ch3 = int(16 * k), int(32 * k), int(64 * k)
    flat_dim = ch3   # after GlobalAvgPool: (ch3,)
    return {
        'conv0':   conv_init(k0, 3, 3, 3,   16,       W_std=1., b_std=0.05),
        'group1':  _wrn_group_init(k1, block_size, 16,  ch1),
        'group2':  _wrn_group_init(k2, block_size, ch1, ch2),
        'group3':  _wrn_group_init(k3, block_size, ch2, ch3),
        # b_std=0. 对应原 stax.Dense(num_classes, 1., 0.)
        'dense':   dense_init(k4, flat_dim, num_classes, W_std=1., b_std=0.),
    }


def wrn_fwd(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    WideResNet 单样本前向。x: (H,W,C) → (num_classes,)。
    GlobalAvgPool = mean over spatial (H,W) dims。
    """
    x = conv_fwd(params['conv0'], x)                           # (32,32,16)
    x = _resnet_group_fwd(params['group1'], x, strides=(1, 1)) # (32,32,ch1)
    x = _resnet_group_fwd(params['group2'], x, strides=(2, 2)) # (16,16,ch2)
    x = _resnet_group_fwd(params['group3'], x, strides=(2, 2)) # (8,8,ch3)
    x = jnp.mean(x, axis=(0, 1))                              # GlobalAvgPool (ch3,)
    x = dense_fwd(params['dense'], x)                          # (num_classes,)
    return x
