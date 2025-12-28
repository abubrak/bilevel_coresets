import numpy as np
import json
import argparse


def get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=False, path='cl_results',
                   save_path='cl_results/best_betas.txt'):
    best_betas = {}
    for method in methods:
        best_beta_for_method = {}
        for dataset in datasets:
            best_acc, best_beta = -1, -1
            for beta in betas:
                res = []
                for seed in seeds:
                    with open('{}/{}_{}_{}_{}_{}.txt'.format(path, dataset, method, buffer_size, beta, seed),
                              'r') as f:
                        data = json.load(f)
                        res.append(data['test_acc'])
                if len(res) > 0 and np.mean(res) > best_acc:
                    best_acc = np.mean(res)
                    best_beta = beta
            print(method, dataset, best_beta)
            best_beta_for_method[dataset] = best_beta
        best_betas[method] = best_beta_for_method
    if save_best:
        with open(save_path, "w") as f:
            json.dump(best_betas, f, sort_keys=True, indent=4)
    return best_betas


def get_result(method, dataset, beta, seeds, buffer_size, path='cl_results'):
    res = []
    for seed in seeds:
        with open('{}/{}_{}_{}_{}_{}.txt'.format(path, dataset, method, buffer_size, beta, seed),
                  'r') as f:
            data = json.load(f)
            res.append(data)
    return res


def print_results(study_name, methods, datasets, betas, seeds, buffer_size, path, save_path):
    """Generic function to print results for any experiment type.
    
    Args:
        study_name (str): Name of the study for display
        methods (list): List of method names
        datasets (list): List of dataset names
        betas (list): List of beta values to search
        seeds (range): Range of seed values
        buffer_size (int): Size of the buffer
        path (str): Path to results directory
        save_path (str): Path to save best betas
    """
    best_betas = get_best_betas(methods, datasets, betas, seeds, buffer_size, save_best=True, 
                                path=path, save_path=save_path)
    print(f'{study_name} study\n')

    print('Method \\ Dataset'.ljust(45), end='')
    for dataset in datasets:
        print(' ' + dataset.ljust(18), end='')
    print('')
    for method in methods:
        print(method.ljust(43), end='')
        for dataset in datasets:
            beta = best_betas[method][dataset]
            res = get_result(method, dataset, beta, seeds, buffer_size, path)
            res = [r['test_acc'] for r in res]
            print(' {:.2f} +- {:.2f}'.format(np.mean(res), np.std(res)).ljust(20), end='')
        print('')


def continual_learning_results():
    datasets = ['permmnist', 'splitmnist']
    methods = [
        'uniform', 'kmeans_features', 'kmeans_embedding', 'kmeans_grads',
        'kcenter_features', 'kcenter_embedding', 'kcenter_grads',
        'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching',
        'coreset'
    ]
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100
    
    print_results('Continual Learning', methods, datasets, betas, seeds, buffer_size, 
                  'cl_results', 'cl_results/best_betas.txt')


def streaming_results():
    datasets = ['permmnist', 'splitmnist']
    methods = ['reservoir', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100
    
    print_results('Streaming', methods, datasets, betas, seeds, buffer_size,
                  'streaming_results', 'streaming_results/best_betas.txt')


def imbalanced_streaming_results():
    datasets = ['splitmnistimbalanced']
    methods = ['reservoir', 'cbrs', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 100
    
    print_results('Streaming', methods, datasets, betas, seeds, buffer_size,
                  'streaming_results', 'streaming_results/best_betas_imbalanced.txt')


def splitcifar_results():
    datasets = ['splitcifar']
    methods = [
        'uniform', 'kmeans_features', 'kmeans_embedding', 'kmeans_grads',
        'kcenter_features', 'kcenter_embedding', 'kcenter_grads',
        'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching',
        'coreset'
    ]

    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 200
    
    print_results('Streaming', methods, datasets, betas, seeds, buffer_size,
                  'cl_results', 'cl_results/best_betas_splitcifar.txt')


def imbalanced_streaming_cifar_results():
    datasets = ['stream_imbalanced_splitcifar']
    methods = ['reservoir', 'cbrs', 'coreset']
    seeds = range(5)
    betas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    buffer_size = 200
    
    print_results('Streaming', methods, datasets, betas, seeds, buffer_size,
                  'streaming_results', 'streaming_results/best_betas_imbalanced_cifar.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results processor')
    parser.add_argument('--exp', default='cl',
                        choices=['cl', 'streaming', 'imbalanced_streaming', 'splitcifar', 'imbalanced_streaming_cifar'])
    args = parser.parse_args()
    exp = args.exp
    if exp == 'cl':
        continual_learning_results()
    elif exp == 'streaming':
        streaming_results()
    elif exp == 'imbalanced_streaming':
        imbalanced_streaming_results()
    elif exp == 'splitcifar':
        splitcifar_results()
    elif exp == 'imbalanced_streaming_cifar':
        imbalanced_streaming_cifar_results()
    else:
        raise Exception('Unknown experiment')
