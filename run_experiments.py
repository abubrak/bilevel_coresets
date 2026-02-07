#!/usr/bin/env python
"""
Run real MNIST experiments with bilevel coresets.

This script runs actual experiments on MNIST data to demonstrate the
effectiveness of bilevel coresets compared to uniform sampling.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.transforms import transforms

# Add the repository root to the path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bilevel_coreset
import loss_utils
import models
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='keras')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# MNIST transforms
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def linear_kernel_fn(x1, x2):
    """Linear kernel function: K(x1, x2) = x1 @ x2.T"""
    return np.dot(x1, x2.T)


def get_mnist_data():
    """Load and preprocess MNIST data."""
    print("\nDownloading/Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=mnist_transform, download=True)
    
    # Load all training data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X_train, y_train = next(iter(train_loader))
    X_train, y_train = X_train.numpy(), y_train.numpy()
    
    # Load all test data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test, y_test = next(iter(test_loader))
    X_test, y_test = X_test.numpy(), y_test.numpy()
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_test, y_test


def train_model(model, device, train_loader, optimizer, weights=None, epochs=100):
    """Train the model."""
    model.train()
    if weights is not None:
        weights = torch.from_numpy(np.array(weights)).float().to(device)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if weights is not None and len(weights) == len(target):
                loss = F.cross_entropy(output, target, reduction='none')
                loss = torch.mean(loss * weights[:len(target)])
            else:
                loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


def test_model(model, device, test_loader):
    """Evaluate the model."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)


def run_mnist_summarization_experiment(X_train, y_train, X_test, y_test, coreset_size=100, epochs=500):
    """Run MNIST data summarization experiment comparing coreset vs uniform sampling."""
    
    print_header(f"MNIST SUMMARIZATION EXPERIMENT (Coreset Size = {coreset_size})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Use a subset for coreset selection (to speed up)
    subset_size = min(5000, len(X_train))
    X_subset = X_train[:subset_size].reshape(subset_size, -1)  # Flatten for linear kernel
    y_subset = y_train[:subset_size]
    
    results = {}
    
    # --- Uniform Sampling ---
    print("\n1. Uniform Sampling...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    uniform_inds = np.random.choice(subset_size, coreset_size, replace=False)
    
    # Create data loaders for uniform
    train_data = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=False)
    train_data.data = train_data.data[uniform_inds]
    train_data.targets = train_data.targets[uniform_inds]
    uniform_train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    test_data = datasets.MNIST(root='./data', train=False, transform=mnist_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256)
    
    # Train with uniform
    model_uniform = models.ConvNet(10).to(device)
    optimizer = torch.optim.Adam(model_uniform.parameters(), lr=5e-4)
    train_model(model_uniform, device, uniform_train_loader, optimizer, epochs=epochs)
    uniform_acc = test_model(model_uniform, device, test_loader)
    results['uniform'] = uniform_acc
    print(f"   Uniform test accuracy: {uniform_acc:.2f}%")
    
    # --- Coreset Selection ---
    print("\n2. Bilevel Coreset Selection...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    bc = bilevel_coreset.BilevelCoreset(
        outer_loss_fn=loss_utils.cross_entropy,
        inner_loss_fn=loss_utils.cross_entropy,
        out_dim=10,
        max_outer_it=10,
        max_inner_it=100,
        logging_period=1000
    )
    
    coreset_inds, coreset_weights = bc.build_with_representer_proxy_batch(
        X_subset, y_subset, coreset_size, linear_kernel_fn,
        cache_kernel=True, start_size=5, inner_reg=1e-5
    )
    
    # Map back to original indices
    print(f"   Selected {len(coreset_inds)} coreset points")
    print(f"   Class distribution: {np.bincount(y_subset[coreset_inds], minlength=10)}")
    
    # Create data loaders for coreset
    train_data = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=False)
    train_data.data = train_data.data[coreset_inds]
    train_data.targets = train_data.targets[coreset_inds]
    coreset_train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Train with coreset
    model_coreset = models.ConvNet(10).to(device)
    optimizer = torch.optim.Adam(model_coreset.parameters(), lr=5e-4)
    train_model(model_coreset, device, coreset_train_loader, optimizer, epochs=epochs)
    coreset_acc = test_model(model_coreset, device, test_loader)
    results['coreset'] = coreset_acc
    print(f"   Coreset test accuracy: {coreset_acc:.2f}%")
    
    # --- Summary ---
    print("\n" + "-" * 50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("-" * 50)
    print(f"Coreset size: {coreset_size} samples (from {subset_size} training samples)")
    print(f"Training epochs: {epochs}")
    print(f"\nMethod          | Test Accuracy")
    print("-" * 35)
    print(f"Uniform         | {uniform_acc:.2f}%")
    print(f"Coreset (Ours)  | {coreset_acc:.2f}%")
    print(f"Improvement     | +{coreset_acc - uniform_acc:.2f}%")
    
    return results


def run_continual_learning_simulation(X_train, y_train, X_test, y_test, buffer_size=100, epochs=100):
    """Run a simplified continual learning experiment on split MNIST."""
    
    print_header(f"CONTINUAL LEARNING EXPERIMENT (Buffer Size = {buffer_size})")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Split MNIST into 5 tasks (digits 0-1, 2-3, 4-5, 6-7, 8-9)
    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    results = {'uniform': [], 'coreset': []}
    
    for method in ['uniform', 'coreset']:
        print(f"\n--- Running {method.upper()} method ---")
        np.random.seed(42)
        torch.manual_seed(42)
        
        model = models.ConvNet(10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        
        buffer_X, buffer_y = None, None
        
        for task_id, (class_a, class_b) in enumerate(tasks):
            print(f"\nTask {task_id + 1}: Classes {class_a}, {class_b}")
            
            # Get task data
            task_mask = (y_train == class_a) | (y_train == class_b)
            X_task = X_train[task_mask][:1000]  # Limit samples per task
            y_task = y_train[task_mask][:1000]
            
            # Create combined training data (current task + buffer)
            if buffer_X is not None:
                X_combined = np.concatenate([X_task, buffer_X])
                y_combined = np.concatenate([y_task, buffer_y])
            else:
                X_combined = X_task
                y_combined = y_task
            
            # Train on combined data
            train_data = torch.utils.data.TensorDataset(
                torch.from_numpy(X_combined).float(),
                torch.from_numpy(y_combined).long()
            )
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
            train_model(model, device, train_loader, optimizer, epochs=epochs)
            
            # Update buffer
            size_per_task = buffer_size // (task_id + 1)
            
            if method == 'uniform':
                inds = np.random.choice(len(X_task), min(size_per_task, len(X_task)), replace=False)
            else:  # coreset
                X_flat = X_task.reshape(len(X_task), -1)
                bc = bilevel_coreset.BilevelCoreset(
                    outer_loss_fn=loss_utils.cross_entropy,
                    inner_loss_fn=loss_utils.cross_entropy,
                    out_dim=10,
                    max_outer_it=5,
                    max_inner_it=50,
                    logging_period=1000
                )
                inds, _ = bc.build_with_representer_proxy_batch(
                    X_flat, y_task, min(size_per_task, len(X_task)), linear_kernel_fn,
                    cache_kernel=True, start_size=1, inner_reg=1e-4
                )
            
            # Add to buffer
            if buffer_X is None:
                buffer_X = X_task[inds]
                buffer_y = y_task[inds]
            else:
                # Trim old buffer entries and add new
                buffer_X = buffer_X[:size_per_task * task_id]
                buffer_y = buffer_y[:size_per_task * task_id]
                buffer_X = np.concatenate([buffer_X, X_task[inds]])
                buffer_y = np.concatenate([buffer_y, y_task[inds]])
        
        # Test on all classes
        test_data = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).long()
        )
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256)
        final_acc = test_model(model, device, test_loader)
        results[method] = final_acc
        print(f"\n{method.upper()} Final Test Accuracy: {final_acc:.2f}%")
    
    # Summary
    print("\n" + "-" * 50)
    print("CONTINUAL LEARNING RESULTS SUMMARY")
    print("-" * 50)
    print(f"Buffer size: {buffer_size}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"\nMethod          | Final Test Accuracy")
    print("-" * 40)
    print(f"Uniform         | {results['uniform']:.2f}%")
    print(f"Coreset (Ours)  | {results['coreset']:.2f}%")
    print(f"Improvement     | +{results['coreset'] - results['uniform']:.2f}%")
    
    return results


def main():
    """Main function to run all experiments."""
    print("\n" + "#" * 70)
    print("#  BILEVEL CORESETS - REAL MNIST EXPERIMENTS  #".center(70))
    print("#" * 70)
    
    # Load MNIST data
    X_train, y_train, X_test, y_test = get_mnist_data()
    
    # Run experiments
    print("\n" + "=" * 70)
    print("Running experiments... (this may take several minutes)")
    print("=" * 70)
    
    # Experiment 1: Data Summarization
    summarization_results = run_mnist_summarization_experiment(
        X_train, y_train, X_test, y_test,
        coreset_size=100,
        epochs=200
    )
    
    # Experiment 2: Continual Learning
    cl_results = run_continual_learning_simulation(
        X_train, y_train, X_test, y_test,
        buffer_size=100,
        epochs=50
    )
    
    # Final Summary
    print_header("FINAL EXPERIMENTAL RESULTS")
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    EXPERIMENTAL RESULTS SUMMARY                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. DATA SUMMARIZATION (MNIST, 100 samples)                          ║
║     ┌────────────────┬─────────────────┐                             ║
║     │ Method         │ Test Accuracy   │                             ║""")
    print(f"║     │ Uniform        │ {summarization_results['uniform']:>13.2f}% │                             ║")
    print(f"║     │ Coreset (Ours) │ {summarization_results['coreset']:>13.2f}% │                             ║")
    print(f"║     │ Improvement    │ {summarization_results['coreset'] - summarization_results['uniform']:>+13.2f}% │                             ║")
    print("""║     └────────────────┴─────────────────┘                             ║
║                                                                       ║
║  2. CONTINUAL LEARNING (Split MNIST, Buffer=100)                     ║
║     ┌────────────────┬─────────────────┐                             ║
║     │ Method         │ Test Accuracy   │                             ║""")
    print(f"║     │ Uniform        │ {cl_results['uniform']:>13.2f}% │                             ║")
    print(f"║     │ Coreset (Ours) │ {cl_results['coreset']:>13.2f}% │                             ║")
    print(f"║     │ Improvement    │ {cl_results['coreset'] - cl_results['uniform']:>+13.2f}% │                             ║")
    print("""║     └────────────────┴─────────────────┘                             ║
║                                                                       ║
║  ✓ Experiments completed successfully!                               ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Save results
    results = {
        'summarization': summarization_results,
        'continual_learning': cl_results
    }
    
    results_file = os.path.join(_REPO_ROOT, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    main()
