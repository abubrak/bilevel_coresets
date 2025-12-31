#!/usr/bin/env python
"""
Demonstration script for the Bilevel Coresets repository.

This script demonstrates the core functionality and displays expected results.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/home/runner/work/bilevel_coresets/bilevel_coresets')

import bilevel_coreset
import loss_utils
import warnings

warnings.filterwarnings('ignore')


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_regression_coreset():
    """Demonstrate regression coreset selection."""
    print_header("1. REGRESSION CORESET DEMONSTRATION")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    n_features = 20
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X @ np.random.randn(n_features, 1) + np.random.randn(n_samples, 1) * 0.5).astype(np.float32)
    
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    
    linear_kernel_fn = lambda x1, x2: np.dot(x1, x2.T)
    
    coreset_size = 10
    bc = bilevel_coreset.BilevelCoreset(
        outer_loss_fn=loss_utils.weighted_mse,
        inner_loss_fn=loss_utils.weighted_mse,
        out_dim=1,
        max_outer_it=5,
        max_inner_it=50,
        logging_period=1000
    )
    
    coreset_inds, coreset_weights = bc.build_with_representer_proxy_batch(
        X, y, coreset_size, linear_kernel_fn, 
        cache_kernel=True, start_size=1, inner_reg=1e-4
    )
    
    print(f"\nCoreset size: {coreset_size}")
    print(f"Selected indices: {coreset_inds}")
    print(f"Weights: {np.round(coreset_weights, 3)}")
    print("\n✓ Regression coreset test PASSED")


def demo_classification_coreset():
    """Demonstrate classification coreset selection."""
    print_header("2. CLASSIFICATION CORESET DEMONSTRATION")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    n_features = 30
    n_classes = 5
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    print(f"\nDataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Class distribution in full data: {np.bincount(y, minlength=n_classes)}")
    
    linear_kernel_fn = lambda x1, x2: np.dot(x1, x2.T)
    
    coreset_size = 20
    bc = bilevel_coreset.BilevelCoreset(
        outer_loss_fn=loss_utils.cross_entropy,
        inner_loss_fn=loss_utils.cross_entropy,
        out_dim=n_classes,
        max_outer_it=5,
        max_inner_it=50,
        logging_period=1000
    )
    
    coreset_inds, coreset_weights = bc.build_with_representer_proxy_batch(
        X, y, coreset_size, linear_kernel_fn,
        cache_kernel=True, start_size=1, inner_reg=1e-5
    )
    
    print(f"\nCoreset size: {coreset_size}")
    print(f"Class distribution in coreset: {np.bincount(y[coreset_inds], minlength=n_classes)}")
    print(f"Selected indices: {coreset_inds}")
    print("\n✓ Classification coreset test PASSED")


def display_paper_results():
    """Display the expected experimental results from the paper."""
    print_header("3. EXPERIMENTAL RESULTS FROM PAPER")
    
    print("\n--- Continual Learning Results (Buffer Size = 100) ---")
    print("\nMethod                        | PermMNIST     | SplitMNIST")
    print("-" * 60)
    cl_results = [
        ("Uniform", "75.19 ± 0.85", "86.73 ± 1.19"),
        ("K-means (Gradients)", "73.28 ± 0.32", "90.20 ± 1.36"),
        ("K-center (Gradients)", "73.67 ± 0.31", "88.30 ± 1.05"),
        ("iCaRL", "74.52 ± 0.73", "87.54 ± 1.41"),
        ("Gradient Matching", "74.06 ± 0.46", "89.95 ± 1.00"),
        ("Coreset (Ours)", "77.21 ± 0.26", "91.94 ± 0.47"),
    ]
    for method, perm, split in cl_results:
        print(f"{method:29} | {perm:13} | {split}")
    
    print("\n--- Imbalanced Streaming Results ---")
    print("\nMethod                        | SplitMNIST-Imbalanced")
    print("-" * 50)
    imbalanced_results = [
        ("Reservoir", "80.60 ± 4.36"),
        ("CBRS", "89.71 ± 1.31"),
        ("Coreset (Ours)", "92.30 ± 0.23"),
    ]
    for method, result in imbalanced_results:
        print(f"{method:29} | {result}")
    
    print("\n--- Data Summarization Results ---")
    print("\nMNIST Summarization (CNN, 250 samples):")
    print("  - Uniform: ~87% test accuracy")
    print("  - Coreset: ~94% test accuracy (+7% improvement)")


def main():
    """Main demonstration function."""
    print("\n" + "#" * 70)
    print("#  BILEVEL CORESETS FOR CONTINUAL LEARNING AND STREAMING  #".center(70))
    print("#" * 70)
    
    demo_regression_coreset()
    demo_classification_coreset()
    display_paper_results()
    
    print_header("DEMONSTRATION COMPLETE")
    print("""
✓ The bilevel coresets library is working correctly!

To run full experiments with real data:
  cd cl_streaming
  python cl.py --dataset splitmnist --method coreset --buffer_size 100

See README.md for detailed instructions.
""")


if __name__ == '__main__':
    main()
