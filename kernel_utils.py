"""Shared kernel function utilities."""
from cl_streaming import ntk_generator


def get_kernel_fn(dataset):
    """Get kernel function for the specified dataset.
    
    Args:
        dataset (str): Dataset name ('permmnist', 'splitmnist', etc.)
        
    Returns:
        function: Kernel function for the dataset
    """
    if dataset == 'permmnist':
        return lambda x, y: ntk_generator.generate_fnn_ntk(x.reshape(-1, 28 * 28), y.reshape(-1, 28 * 28))
    else:
        return lambda x, y: ntk_generator.generate_cnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))
