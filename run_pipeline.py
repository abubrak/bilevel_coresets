#!/usr/bin/env python3
"""
run_pipeline.py
===============
bilevel_coresets 项目的完整运行流程脚本（NTK 部分已替换为 JAX 0.7.2 原生实现）。

使用方法
--------
    # 实验 1：玩具回归（纯 CPU，无额外依赖）
    python run_pipeline.py --exp regression

    # 实验 2：MNIST 数据摘要（需要 torch + torchvision）
    python run_pipeline.py --exp mnist --method coreset --coreset_size 100 --seed 0

    # 实验 3：MNIST 持续学习（需要 torch + torchvision）
    python run_pipeline.py --exp cl --dataset splitmnist --method coreset --buffer_size 100

    # 实验 4：批量主动学习 Nystrom 代理（需要 torch + torchvision）
    python run_pipeline.py --exp active --coreset_size 20

    # 顺序运行全部可用实验
    python run_pipeline.py --exp all

依赖
----
    必须：jax[cpu]==0.7.2, numpy, scipy, scikit-learn
    可选：torch, torchvision（用于 mnist / cl / active 实验）

原始文件 → 本版替换对应关系
-----------------------------
    cl_streaming/ntk_generator.py        → ntk_generator_jax.py
    data_summarization/generate_cntk.py  → generate_cntk_jax.py
    from jax.api import jit              → from jax import jit
    from neural_tangents import stax     → models_jax.py + ntk_core.py
"""

import argparse
import sys
import os
import json
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 路径：把 ntk_jax/ 和原始 bilevel_coresets/ 都加入 sys.path
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
NTK_JAX_DIR   = os.path.join(SCRIPT_DIR, 'ntk_jax')
BILEVEL_DIR   = os.path.join(SCRIPT_DIR, 'bilevel_coresets')
CL_DIR        = os.path.join(BILEVEL_DIR, 'cl_streaming')
for p in [NTK_JAX_DIR, BILEVEL_DIR, CL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 屏蔽 JAX 的 GPU 警告（CPU 运行时）
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
print(f"[环境] JAX {jax.__version__}  |  设备: {jax.devices()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)

def step(msg: str):
    print(f"\n  ▶ {msg}")

def result(msg: str):
    print(f"  ✓ {msg}")


def try_import_torch():
    """返回 torch 模块，若未安装则提示并退出。"""
    try:
        import torch
        import torchvision
        return torch, torchvision
    except ImportError:
        print("\n[错误] 此实验需要 torch 和 torchvision。")
        print("  安装命令：pip install torch torchvision")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 实验 1：玩具回归（与 demo.ipynb 中 Regression 部分等价）
# NTK 不涉及：使用多项式核，作为基础流程验证
# ═══════════════════════════════════════════════════════════════════════════════

def run_regression():
    section("实验 1：玩具 1D 回归（多项式核 Coreset vs 均匀采样）")

    from sklearn.preprocessing import PolynomialFeatures
    # ── 在这里导入 JAX NTK 核（虽然本实验用多项式核，但演示接口兼容性）
    from ntk_generator_jax import generate_fnn_ntk

    # 导入原始 bilevel_coreset（PyTorch 实现，无需修改）
    try:
        import torch
        import bilevel_coreset as bc_module
        import loss_utils
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        print("  [注意] torch 未安装，将跳过 BilevelCoreset 优化，仅演示 NTK 接口")

    np.random.seed(0)
    lim = 1
    x = np.random.randn(85) * 0.2
    x = x[np.logical_and(x <= lim, x >= -lim)]
    x_linspace = np.linspace(-lim, lim, num=15)
    x = np.concatenate((x, x_linspace)).reshape(-1, 1)
    true_fn = lambda x: 5 * np.sin(x * 7)
    y = true_fn(x.reshape(-1)) + 0.25 * np.random.randn(x.shape[0])
    reg = 1e-6

    poly = PolynomialFeatures(7)
    X_feat = poly.fit_transform(x)
    step(f"数据集：{x.shape[0]} 个样本，多项式特征维度 {X_feat.shape[1]}")

    # 全量最优解
    theta_full = np.linalg.pinv(X_feat.T @ X_feat + reg * np.eye(X_feat.shape[1])) @ X_feat.T @ y
    mse_full = np.mean((X_feat @ theta_full - y) ** 2)
    result(f"全量解 MSE：{mse_full:.6f}")

    # 均匀采样子集（size=10）
    chosen = np.random.choice(x.shape[0], 10, replace=False)
    theta_unif = np.linalg.pinv(X_feat[chosen].T @ X_feat[chosen] + reg * np.eye(X_feat.shape[1])) \
                 @ X_feat[chosen].T @ y[chosen]
    mse_unif = np.mean((X_feat @ theta_unif - y) ** 2)
    result(f"均匀采样（n=10）MSE：{mse_unif:.6f}")

    # 多项式核 coreset（需要 torch）
    if HAS_TORCH:
        import bilevel_coreset
        import loss_utils
        poly_kernel = lambda x1, x2: np.dot(
            poly.transform(x1.reshape(-1, 1)),
            poly.transform(x2.reshape(-1, 1)).T
        )
        bc = bilevel_coreset.BilevelCoreset(
            outer_loss_fn=loss_utils.weighted_mse,
            inner_loss_fn=loss_utils.weighted_mse,
            out_dim=1, max_outer_it=1,
            inner_lr=0.25, max_inner_it=500,
            logging_period=9999   # 静默
        )
        coreset_inds, _ = bc.build_with_representer_proxy_batch(
            x, y.reshape(-1, 1), 10,
            kernel_fn_np=poly_kernel,
            cache_kernel=True, start_size=3, inner_reg=reg
        )
        theta_cs = np.linalg.pinv(X_feat[coreset_inds].T @ X_feat[coreset_inds] + reg * np.eye(X_feat.shape[1])) \
                   @ X_feat[coreset_inds].T @ y[coreset_inds]
        mse_cs = np.mean((X_feat @ theta_cs - y) ** 2)
        result(f"Bilevel Coreset（n=10）MSE：{mse_cs:.6f}  （相比均匀 {'+' if mse_cs>=mse_unif else '-'}{abs(mse_cs-mse_unif):.4f}）")

    # 演示 JAX NTK 接口
    step("演示 generate_fnn_ntk 作为 kernel_fn_np（维度适配）")
    x_flat = np.tile(x, (1, 784 // x.shape[1] + 1))[:, :784].astype(np.float32)
    K_demo = generate_fnn_ntk(x_flat[:5], x_flat[:3])
    result(f"FNN NTK kernel 矩阵形状：{K_demo.shape}，数值范围 [{K_demo.min():.2f}, {K_demo.max():.2f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 实验 2：MNIST 数据摘要（对应 data_summarization/cnn_mnist.py）
# 关键替换：ntk_generator.generate_cnn_ntk → ntk_generator_jax.generate_cnn_ntk
# ═══════════════════════════════════════════════════════════════════════════════

def run_mnist(method: str = 'coreset', coreset_size: int = 100, seed: int = 0):
    section(f"实验 2：MNIST 数据摘要  method={method}  size={coreset_size}  seed={seed}")
    torch, torchvision = try_import_torch()
    import torch.nn.functional as F

    # ── 关键替换：原版 from cl_streaming import ntk_generator
    #              本版 from ntk_generator_jax import generate_cnn_ntk
    from ntk_generator_jax import generate_cnn_ntk
    import bilevel_coreset
    import loss_utils
    import models
    from cl_streaming.summary import UniformSummarizer

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── 数据加载（与原版 get_data() 完全一致）
    step("加载 MNIST 数据集")
    mnist_tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=mnist_tf, download=True)
    test_dataset  = torchvision.datasets.MNIST(
        root='./data', train=False, transform=mnist_tf)

    n_full = len(train_dataset)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_full, shuffle=False)
    X_all, y_all = next(iter(loader))
    X_all, y_all = X_all.numpy(), y_all.numpy()

    # 限制候选集大小（与原版 lim=10000 一致）
    lim = 10000
    X, y = X_all[:lim], y_all[:lim]
    result(f"候选集：{X.shape[0]} 个样本，形状 {X.shape}")

    # ── 核心：NTK 核函数定义
    # 原版：kernel_fn = lambda x, y: ntk_generator.generate_cnn_ntk(
    #                       x.reshape(-1,28,28,1), y.reshape(-1,28,28,1))
    # 本版：完全等价，仅替换导入源
    kernel_fn = lambda x1, x2: generate_cnn_ntk(
        x1.reshape(-1, 28, 28, 1),
        x2.reshape(-1, 28, 28, 1)
    )

    # ── 子集选择
    step(f"使用 '{method}' 方法选择 {coreset_size} 个训练样本")
    t0 = time.time()

    if method == 'uniform':
        summarizer = UniformSummarizer(np.random.RandomState(seed))
        inds    = summarizer.build_summary(X, y, coreset_size)
        weights = np.ones(coreset_size)

    elif method == 'coreset':
        bc = bilevel_coreset.BilevelCoreset(
            outer_loss_fn=loss_utils.cross_entropy,
            inner_loss_fn=loss_utils.cross_entropy,
            out_dim=10,
            max_outer_it=10,
            outer_lr=0.05,
            max_inner_it=200,
            logging_period=1000  # 每 1000 步打印一次
        )
        # ── 原版调用（接口完全一致，只是 kernel_fn 来源不同）
        inds, weights = bc.build_with_representer_proxy_batch(
            X, y, coreset_size,
            kernel_fn_np=kernel_fn,    # ← 这里使用 JAX 原生 NTK
            cache_kernel=True,
            start_size=10,
            inner_reg=1e-7
        )
    else:
        raise ValueError(f"未知方法：{method}")

    elapsed = time.time() - t0
    result(f"子集选择完成，耗时 {elapsed:.1f}s")
    result(f"选出样本数：{len(inds)}，类别分布：{np.bincount(y_all[inds])}")

    # ── 在子集上训练 CNN
    step("在选出的子集上训练 ConvNet（4000 epoch）")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result(f"训练设备：{device}")

    # 构造 DataLoader（与原版 get_mnist_loaders 等价）
    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=False, transform=mnist_tf)
    train_data.data    = train_data.data[inds]
    train_data.targets = train_data.targets[inds]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    model     = models.ConvNet(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    w_tensor  = torch.from_numpy(np.array(weights)).float().to(device)
    nr_epochs = 4000
    test_accs = []

    for epoch in range(1, nr_epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = torch.mean(
                F.cross_entropy(model(data), target, reduction='none') * w_tensor
            )
            loss.backward()
            optimizer.step()

        if epoch % 500 == 0 or epoch > nr_epochs - 5:
            model.eval()
            correct = sum(
                model(d.to(device)).argmax(1).eq(t.to(device)).sum().item()
                for d, t in test_loader
            )
            acc = correct / len(test_loader.dataset)
            if epoch % 500 == 0:
                print(f"    Epoch {epoch:4d}  test_acc={acc:.4f}")
            if epoch > nr_epochs - 5:
                test_accs.append(acc)

    final_acc = np.mean(test_accs)
    result(f"最终测试准确率（最后 5 epoch 平均）：{final_acc:.4f}")

    # 保存结果
    os.makedirs('./results/mnist', exist_ok=True)
    fname = f'./results/mnist/{method}_{coreset_size}_{seed}.json'
    with open(fname, 'w') as f:
        json.dump({'method': method, 'coreset_size': coreset_size,
                   'seed': seed, 'test_acc': final_acc}, f, indent=2)
    result(f"结果已保存至 {fname}")
    return final_acc


# ═══════════════════════════════════════════════════════════════════════════════
# 实验 3：持续学习（对应 cl_streaming/cl.py）
# 关键替换：ntk_generator.generate_fnn/cnn_ntk → ntk_generator_jax 对应函数
# ═══════════════════════════════════════════════════════════════════════════════

def run_continual_learning(dataset: str = 'splitmnist',
                           method: str = 'coreset',
                           buffer_size: int = 100,
                           beta: float = 1.0,
                           seed: int = 0,
                           nr_epochs: int = 10,
                           samples_per_task: int = 200):
    section(f"实验 3：持续学习  dataset={dataset}  method={method}  buffer={buffer_size}")
    torch, torchvision = try_import_torch()

    # ── 关键替换：原版 from cl_streaming import ntk_generator
    #              本版 from ntk_generator_jax import ...
    from ntk_generator_jax import generate_fnn_ntk, generate_cnn_ntk
    import bilevel_coreset
    import loss_utils
    import models
    from cl_streaming.datagen import PermutedMnistGenerator, SplitMnistGenerator, NumpyDataset
    from cl_streaming.training import Training
    from cl_streaming import summary

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── NTK 核函数（与原版 get_kernel_fn 等价，仅替换导入）
    # 原版：return lambda x, y: ntk_generator.generate_fnn_ntk(x.reshape(-1, 784), ...)
    # 本版：
    if dataset == 'permmnist':
        kernel_fn = lambda x, y: generate_fnn_ntk(
            x.reshape(-1, 784), y.reshape(-1, 784))
    else:
        kernel_fn = lambda x, y: generate_cnn_ntk(
            x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))

    # ── 数据生成器
    step(f"初始化数据集：{dataset}，每任务 {samples_per_task} 样本")
    if dataset == 'permmnist':
        generator = PermutedMnistGenerator(samples_per_task)
        model_cls = lambda: models.FNNet(784, 100, 10)
    elif dataset == 'splitmnist':
        generator = SplitMnistGenerator(samples_per_task)
        model_cls = lambda: models.ConvNet(10)
    else:
        raise ValueError(f"未知数据集：{dataset}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result(f"任务数：{generator.max_iter}，训练设备：{device}")

    # 预生成所有任务数据
    tasks        = []
    train_loaders = []
    test_loaders  = []
    for _ in range(generator.max_iter):
        X_tr, y_tr, X_te, y_te = generator.next_task()
        tasks.append((X_tr, y_tr, X_te, y_te))
        tr_ds = NumpyDataset(X_tr, y_tr)
        te_ds = NumpyDataset(X_te, y_te)
        train_loaders.append(torch.utils.data.DataLoader(tr_ds, batch_size=64, shuffle=True))
        test_loaders.append(torch.utils.data.DataLoader(te_ds, batch_size=256))

    inner_reg = 1e-3
    model     = model_cls().to(device)
    trainer   = Training(model, device, nr_epochs, beta=beta)

    bc = bilevel_coreset.BilevelCoreset(
        outer_loss_fn=loss_utils.cross_entropy,
        inner_loss_fn=loss_utils.cross_entropy,
        out_dim=10, max_outer_it=1,
        max_inner_it=200, logging_period=9999
    )
    rs = np.random.RandomState(seed)

    # ── 主循环（与原版 continual_learning() 等价）
    for i in range(generator.max_iter):
        step(f"任务 {i+1}/{generator.max_iter}")

        # 训练当前任务
        trainer.train(train_loaders[i])

        # 裁剪历史 buffer
        size_per_task = buffer_size // (i + 1)
        for j in range(i):
            (X_buf, y_buf), _ = trainer.buffer[j]
            trainer.buffer[j] = (
                (X_buf[:size_per_task], y_buf[:size_per_task]),
                np.ones(size_per_task)
            )

        # 选择当前任务的 coreset
        X_task, y_task, _, _ = tasks[i]
        if method == 'coreset':
            chosen_inds, _ = bc.build_with_representer_proxy_batch(
                X_task, y_task, size_per_task,
                kernel_fn,              # ← JAX 原生 NTK
                cache_kernel=True,
                start_size=1,
                inner_reg=inner_reg
            )
        else:
            summarizer = summary.Summarizer.factory(method, rs)
            chosen_inds = summarizer.build_summary(
                X_task, y_task, size_per_task,
                method=method, model=model, device=device
            )

        trainer.buffer.append((
            (X_task[chosen_inds], y_task[chosen_inds]),
            np.ones(size_per_task)
        ))

    # ── 评估所有任务
    step("评估全部任务测试准确率")
    accs = []
    for k in range(generator.max_iter):
        acc = trainer.test(test_loaders[k])
        accs.append(acc)
        print(f"    任务 {k+1} 准确率：{acc:.2f}%")

    mean_acc = np.mean(accs)
    result(f"平均准确率：{mean_acc:.2f}%")

    os.makedirs('./results/cl', exist_ok=True)
    fname = f'./results/cl/{dataset}_{method}_{buffer_size}_{seed}.json'
    with open(fname, 'w') as f:
        json.dump({'dataset': dataset, 'method': method,
                   'buffer_size': buffer_size, 'seed': seed,
                   'mean_acc': mean_acc, 'per_task': accs}, f, indent=2)
    result(f"结果已保存至 {fname}")
    return mean_acc


# ═══════════════════════════════════════════════════════════════════════════════
# 实验 4：批量主动学习（对应 batch_active_learning/nystrom_example.py）
# 关键替换：neural_tangents WideResnet → generate_cntk_jax.generate_kernel
# ═══════════════════════════════════════════════════════════════════════════════

def run_active_learning(coreset_size: int = 20, base_inds_size: int = 10,
                        nystrom_dim: int = 500, seed: int = 0):
    section(f"实验 4：批量主动学习（Nystrom 代理）  coreset_size={coreset_size}")
    torch, torchvision = try_import_torch()

    # ── 关键替换：原版用 neural_tangents WideResnet + jax.api.jit
    #              本版用 generate_cntk_jax.generate_kernel（WRN JAX 原生）
    #   原版：kernel_fn = jit(kernel_fn, static_argnums=(2,))
    #         def kernel_fn_ntk(x, y, step=64): ...np.array(kernel_fn(x,y,'ntk'))
    #   本版：
    from generate_cntk_jax import generate_kernel as _wrn_kernel_fn
    import bilevel_coreset
    import models

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 数据集（与原版完全一致）
    step("加载 CIFAR-10 数据集")
    tf_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    num_classes = 10

    def target_transform(x):
        res = np.zeros(num_classes)
        res[x] = 1000.0
        return res

    trainset       = torchvision.datasets.CIFAR10(
        './data', train=True, download=True,
        transform=tf_train, target_transform=target_transform)
    trainset_no_aug = torchvision.datasets.CIFAR10(
        './data', train=True, download=False,
        transform=tf_test, target_transform=target_transform)

    batch_size = 64

    def loader_creator_fn(dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=0, shuffle=shuffle)

    def loss_fn(pred, true, weights=None):
        import torch.nn.functional as F
        kl = -torch.mean(
            torch.sum(F.log_softmax(pred, dim=1) * F.softmax(true.float(), dim=1), dim=1))
        if weights is not None:
            kl = torch.mean(
                torch.sum(F.log_softmax(pred, dim=1) * F.softmax(true.float(), dim=1), dim=1)
                * weights * -1)
        return kl

    # ── kernel_fn_ntk（与原版接口一致，改用 JAX 原生 WRN）
    # 原版：
    #   def kernel_fn_ntk(x, y, step=64):
    #       K = np.zeros(...)
    #       x = x.transpose(0,2,3,1)   # NCHW → NHWC
    #       for i in range(...): K[...] = kernel_fn(x[...], y, 'ntk')
    #       return K
    # 本版：
    def kernel_fn_ntk(x: np.ndarray, y: np.ndarray, step: int = 64) -> np.ndarray:
        """
        WideResNet NTK 核函数。
        x, y: NCHW numpy 数组（torchvision 默认格式）
        返回: (n, m) NTK 矩阵
        """
        x_nhwc = x.transpose(0, 2, 3, 1).astype(np.float32)  # → NHWC
        y_nhwc = y.transpose(0, 2, 3, 1).astype(np.float32)
        n, m = x_nhwc.shape[0], y_nhwc.shape[0]
        K = np.zeros((n, m), dtype=np.float32)
        from ntk_core import ntk_matrix_col_blocked
        from models_jax import wrn_init, wrn_fwd
        from jax import random, jit

        # 懒加载 WRN 参数（避免每次重新初始化）
        if not hasattr(kernel_fn_ntk, '_params'):
            kernel_fn_ntk._params = wrn_init(random.PRNGKey(0), block_size=4, k=1, num_classes=10)
        from ntk_core import ntk_matrix_col_blocked
        K = ntk_matrix_col_blocked(jit(wrn_fwd), kernel_fn_ntk._params,
                                   x_nhwc, y_nhwc, col_block=step)
        return K

    step(f"初始化 BilevelCoreset（Nystrom 代理，nystrom_dim={nystrom_dim}）")
    model = models.LogisticRegression(nystrom_dim, num_classes)
    bc    = bilevel_coreset.BilevelCoreset(
        loss_fn, loss_fn,
        max_inner_it=500,           # 原版 7500，这里缩减以便快速演示
        max_conj_grad_it=50
    )

    base_inds = np.random.choice(len(trainset.targets), base_inds_size, replace=False)
    step(f"已选基础索引：{base_inds_size} 个，开始贪婪选择 {coreset_size} 个样本")

    inds = bc.build_with_nystrom_proxy(
        trainset, trainset_no_aug,
        base_inds, coreset_size,
        kernel_fn_ntk,
        loader_creator_fn, model,
        nystrom_features_dim=nystrom_dim,
        val_size=1000,             # 原版 30000，缩减以便演示
        inner_reg=1e-4,
        nr_presampled_transforms=10   # 原版 100，缩减
    )

    new_inds = inds[-coreset_size:]
    result(f"新选出的 {coreset_size} 个索引：{new_inds}")

    os.makedirs('./results/active', exist_ok=True)
    fname = f'./results/active/nystrom_{coreset_size}_{seed}.json'
    with open(fname, 'w') as f:
        json.dump({'coreset_size': coreset_size, 'seed': seed,
                   'selected_inds': new_inds.tolist()}, f, indent=2)
    result(f"结果已保存至 {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='bilevel_coresets 项目运行脚本（JAX 0.7.2 原生 NTK）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--exp', default='regression',
        choices=['regression', 'mnist', 'cl', 'active', 'all'],
        help='运行的实验类型')

    # MNIST / CL 共用
    parser.add_argument('--method', default='coreset',
        choices=['uniform', 'coreset'],
        help='子集选择方法')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--coreset_size', type=int, default=100)

    # CL 专用
    parser.add_argument('--dataset', default='splitmnist',
        choices=['permmnist', 'splitmnist'],
        help='持续学习数据集')
    parser.add_argument('--buffer_size', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1.0,
        help='持续学习中历史 buffer 损失的权重')
    parser.add_argument('--nr_epochs', type=int, default=10,
        help='每个任务的训练 epoch 数（原版为更大值，此处缩减以便快速运行）')
    parser.add_argument('--samples_per_task', type=int, default=200,
        help='每个 CL 任务的样本数（原版为更大值）')

    # Active learning 专用
    parser.add_argument('--nystrom_dim', type=int, default=500,
        help='Nystrom 特征维度（原版 2000，缩减以便快速运行）')

    args = parser.parse_args()

    t_start = time.time()

    if args.exp == 'regression' or args.exp == 'all':
        run_regression()

    if args.exp == 'mnist' or args.exp == 'all':
        run_mnist(method=args.method,
                  coreset_size=args.coreset_size,
                  seed=args.seed)

    if args.exp == 'cl' or args.exp == 'all':
        run_continual_learning(
            dataset=args.dataset,
            method=args.method,
            buffer_size=args.buffer_size,
            beta=args.beta,
            seed=args.seed,
            nr_epochs=args.nr_epochs,
            samples_per_task=args.samples_per_task
        )

    if args.exp == 'active' or args.exp == 'all':
        run_active_learning(
            coreset_size=args.coreset_size,
            nystrom_dim=args.nystrom_dim,
            seed=args.seed
        )

    section(f"全部完成，总耗时 {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
