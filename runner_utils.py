"""Shared utilities for experiment runner scripts."""
import os


def setup_environment(gpu_id):
    """Set up environment variables for running experiments.
    
    Args:
        gpu_id (int): GPU device ID to use
        
    Returns:
        dict: Environment dictionary with configured variables
    """
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return env


def get_gpu_id(counter, nr_gpus):
    """Get GPU ID based on counter and number of available GPUs.
    
    Args:
        counter (int): Current experiment counter
        nr_gpus (int): Number of available GPUs
        
    Returns:
        int: GPU device ID to use
    """
    return counter % nr_gpus
