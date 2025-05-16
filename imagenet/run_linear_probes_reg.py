#!/usr/bin/env python3
"""
Model evaluation script for Imagenette dataset using pre-computed kernels.

This script evaluates different models on a set of pre-generated tasks
using linear probes. For each kernel file, it:

1. Loads the pre-computed kernel
2. Runs a linear probe on multiple tasks (loaded from a pre-saved file)
3. Records the resulting per-task accuracies
4. Saves the results to a JSON file

Example
-------
$ python imagenette_model_evaluation.py --tasks_file labels_list_imagenette_1.npy \
                                       --kernels_dir kernels_out \
                                       --output kernel_evaluation_results.json
"""

import argparse
import json
import os
import time
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm, trange

# -----------------------------------------------------------------------------#
#                       Multi‑task linear probe                                #
# -----------------------------------------------------------------------------#
def linear_probe_multi(features, labels, batch_size=64, device=torch.device('cuda')):
    """Run linear probe on multiple tasks using pre-computed features."""
    n_samples, feat_dim = features.shape
    n_tasks = labels.shape[0]
    
    # Create dataset and loader from features and labels
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Create linear head
    head = torch.nn.Linear(feat_dim, n_tasks).to(device)

    # Compute the OLS solution to the classification problem by treating it as a regression
    # Convert features to numpy for OLS computation
    features_np = features
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    
    # Prepare labels for regression (+1 for positive class, -1 for negative class)
    labels_np = labels.T  # Transpose to match expected shape
    if isinstance(labels_np, torch.Tensor):
        labels_np = labels_np.cpu().numpy()
    
    # Replace 0s with -1s for regression
    labels_np = 2 * labels_np - 1
    
    # Compute OLS solution: w = (X^T X)^(-1) X^T y
    # Add a small regularization term to ensure invertibility
    reg_term = 1e-8 * np.eye(feat_dim)
    XT_X = features_np.T @ features_np + reg_term
    XT_y = features_np.T @ labels_np
    
    # Solve the linear system
    weights = np.linalg.solve(XT_X, XT_y)
    bias = np.mean(labels_np - features_np @ weights, axis=0)
    
    # Transfer the computed weights to the linear head
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weights.T).float())
        head.bias.copy_(torch.from_numpy(bias).float())


    dataset = TensorDataset(
        torch.from_numpy(features).float().to(device),
        torch.from_numpy(labels.T).float().to(device)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ─ Evaluation per task ─
    correct = torch.zeros(n_tasks, device=device)
    total   = torch.zeros(n_tasks, device=device)

    with torch.no_grad():
        for feats, lbls in loader:
            with torch.cuda.amp.autocast():
                preds = (head(feats) > 0).float()
                
            correct += ((preds == lbls) & (lbls == 1)).sum(dim=0)
            total += (lbls == 1).sum(dim=0)

    acc = (100 * correct / total).cpu().numpy()

    # Compute Mean Squared Error (MSE) between predictions and true labels
    mse = torch.zeros(n_tasks, device=device)
    total_samples = torch.zeros(n_tasks, device=device)
    
    with torch.no_grad():
        for feats, lbls in loader:
            with torch.cuda.amp.autocast():
                outputs = head(feats)
                # Convert binary labels (0/1) to regression targets (-1/+1)
                targets = 2 * lbls - 1
                # Calculate squared error for each task
                squared_error = (outputs - targets) ** 2
                
            # Sum squared errors for each task
            mse += squared_error.sum(dim=0)
            total_samples += feats.size(0)
    
    # Calculate mean squared error for each task
    mse = (mse / total_samples).cpu().numpy()

    return acc, mse

# -----------------------------------------------------------------------------#
#                           Feature extraction                                 #
# -----------------------------------------------------------------------------#
def extract_features_from_kernel(kernel_matrix):
    """Extract features from kernel matrix using eigendecomposition."""
    # Center the kernel
    n = kernel_matrix.shape[0]
    one_n = np.ones((n, n)) / n
    centered_K = kernel_matrix - one_n @ kernel_matrix - kernel_matrix @ one_n + one_n @ kernel_matrix @ one_n
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(centered_K)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Keep only positive eigenvalues
    pos_idx = eigenvalues > 1e-10
    eigenvalues = eigenvalues[pos_idx]
    eigenvectors = eigenvectors[:, pos_idx]
    
    # Create features: V * sqrt(Λ)
    features = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    return features

# -----------------------------------------------------------------------------#
#                                 Main entry                                   #
# -----------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks_file', default='labels_mini_imagenet_efficientnet_b5.npy',
                    help='file containing pre-generated task labels')
    ap.add_argument('--kernels_dir', default='kernels_out',
                    help='directory containing kernel files (K_*.pt)')
    ap.add_argument('--output', default='kernel_evaluation_results_all_models_huge.json',
                    help='where to store the JSON results')
    ap.add_argument('--batch_size', type=int, default=256,
                    help='batch size for evaluation')
    ap.add_argument('--probe_epochs', type=int, default=100,
                    help='number of epochs for linear probe training')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Find kernel files
    kernels_dir = Path(args.kernels_dir)
    kernel_files = list(kernels_dir.glob("K_*.pt"))
    
    if not kernel_files:
        raise FileNotFoundError(
            f"No kernel files found in '{args.kernels_dir}'. "
            "Please generate kernel files first.")
    
    print(f"Found {len(kernel_files)} kernel files:")
    for i, f in enumerate(kernel_files):
        model_name = f.stem.split('_', 1)[1].rsplit('_', 1)[0]
        n_images = int(f.stem.split('_')[-1])
        print(f"[{i}] {model_name} ({n_images} images)")
    
    # Load task labels
    if not os.path.exists(args.tasks_file):
        raise FileNotFoundError(
            f'{args.tasks_file} not found – please place it next to this script or pass the correct path.')
    
    labels_list = np.load(args.tasks_file, allow_pickle=True)
    print(f"Loaded {len(labels_list)} tasks from {args.tasks_file}")
    
    # Convert labels to matrix
    labels_matrix = np.stack(labels_list)
    
    # Evaluate each kernel
    results = []
    
    for kernel_file in kernel_files:
        model_name = kernel_file.stem.split('_', 1)[1].rsplit('_', 1)[0]
        print(f"\nEvaluating {model_name}...")
        start = time.time()
        
        kernel_data = torch.load(kernel_file)

        features = kernel_data['Z'].numpy()
        print(f"Extracted features with shape {features.shape}")
        
        # Run linear probe evaluation
        task_acc, task_mse= linear_probe_multi(
            features, 
            labels_matrix,
            batch_size=args.batch_size,
            device=device
        )
        
        runtime = time.time() - start
        
        # Save results
        results.append({
            'model': model_name,
            'mean_accuracy': float(task_acc.mean()),
            'mean_mse': float(task_mse.mean()),
            'accuracies': task_acc.tolist(),
            'seconds': runtime,
        })
        
        # Print summary
        print(f"{model_name} → mean acc {task_acc.mean():5.2f} (across {len(task_acc)} tasks) "
                f"in {runtime/60:.1f} min")
        
        # Free up memory
        torch.cuda.empty_cache()
        
        # Save intermediate results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nEvaluation complete. Results saved to: {args.output}')

if __name__ == '__main__':
    main()