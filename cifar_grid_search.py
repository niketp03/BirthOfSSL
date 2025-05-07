
#!/usr/bin/env python3
"""
Grid search evaluation script generated from cifar_eval_on_tasks.ipynb.

This script performs a grid‑search over lists of β (beta) and temperature T
values.  For every (beta, T) pair it:

1.  Instantiates a fresh ResNet‑18 backbone (pre‑trained on ImageNet) and a
    frozen reference backbone.
2.  Fine‑tunes the backbone for `--epochs` epochs with the contrastive‑style
    loss used in the original notebook:

        loss = β · Var(M, σ(K/T)) – E(M, σ(K/T))

    where K is the batch–similarity matrix from the frozen model and M is the
    batch–similarity matrix from the trainable model (see
    `compute_expectation_variance`).
3.  Runs a 100‑task linear probe (`linear_probe_multi`) and records the
    resulting per‑task accuracies.

Results are appended to a list of dictionaries and written to disk as JSON so
the file is human‑readable and can be re‑loaded with `json.load`.

Example
-------
$ python cifar_grid_search.py --betas 0.1,0.5,1 --Ts 0.5,1,2 --epochs 10 \
                              --output grid_results.json
"""

import argparse, json, os, time
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from einops import rearrange
from tqdm.auto import trange

# -----------------------------------------------------------------------------#
#                                 Data stuff                                   #
# -----------------------------------------------------------------------------#
def get_loaders(batch_size: int = 512):
    """Return CIFAR‑10 train and test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform)
    test_ds  = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader

# -----------------------------------------------------------------------------#
#                          Expectation / variance                              #
# -----------------------------------------------------------------------------#
def compute_expectation_variance(K: torch.Tensor,
                                 M: torch.Tensor,
                                 T: float = 1.0) -> Tuple[torch.Tensor,
                                                          torch.Tensor]:
    """Return Σ M σ and Σ M² σ(1‑σ) with σ = sigmoid(K/T)."""
    sigma       = torch.sigmoid(K / T)
    expectation = (M * sigma).sum()
    variance    = ((M**2) * sigma * (1 - sigma)).sum()
    return expectation, variance

# -----------------------------------------------------------------------------#
#                       Multi‑task linear probe (100 tasks)                    #
# -----------------------------------------------------------------------------#
class MultiTaskWrapper(torch.utils.data.Dataset):
    """Wrap a dataset to yield (img, label‑vector[100])."""
    def __init__(self, base_ds, all_labels: torch.Tensor):
        self.base_ds  = base_ds               # underlying dataset
        self.all_lbls = all_labels            # (100, N)

    def __len__(self): return len(self.base_ds)

    def __getitem__(self, idx):
        img, _     = self.base_ds[idx]
        lbl_vector = self.all_lbls[:, idx]    # (100,)
        return img, lbl_vector


def linear_probe_multi(model: nn.Module,
                       loader: DataLoader,
                       n_epoch: int = 30,
                       device: torch.device = torch.device('cpu')) -> np.ndarray:
    """Return accuracy (shape = (100,)) across the 100 binary tasks."""
    feat_dim, n_tasks = 512, 100
    model.eval().to(device)

    head = nn.Linear(feat_dim, n_tasks).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(head.parameters(), lr=1e-2)

    for epoch in trange(n_epoch, desc='[probe]', leave=False):
        runloss = 0.
        for imgs, lbls in loader:
            imgs = torch.nn.functional.interpolate(imgs, size=(224, 224),
                                                   mode='bilinear',
                                                   align_corners=False
                                                  ).to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            logits = head(model(imgs))
            loss = criterion(logits, lbls)
            optim.zero_grad()
            loss.backward()
            optim.step()
            runloss += loss.item()

    # ── evaluation ──
    correct = torch.zeros(n_tasks, device=device)
    total   = 0
    head.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = torch.nn.functional.interpolate(imgs, size=(224, 224),
                                                   mode='bilinear',
                                                   align_corners=False
                                                  ).to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            preds = (head(model(imgs)) > 0).float()
            correct += (preds == lbls).sum(dim=0)
            total   += lbls.size(0)

    acc = (100 * correct / total).cpu().numpy()   # (100,)
    return acc

# -----------------------------------------------------------------------------#
#                              Training routine                                #
# -----------------------------------------------------------------------------#
def train_and_evaluate(beta: float,
                       T: float,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       labels_list: np.ndarray,
                       n_epoch: int = 30,
                       device: torch.device = torch.device('cpu')
                      ) -> np.ndarray:
    """Fine‑tune and return per‑task accuracies (length=100)."""
    # Fresh backbones for every run
    model        = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc     = nn.Identity()

    frozen_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    frozen_model.fc = nn.Identity()
    frozen_model.eval()

    model.to(device)
    frozen_model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in trange(n_epoch, desc=f'[train β={beta},T={T}]', leave=False):
        run_loss = run_exp = run_var = 0.0
        n_batches = 0
        for images, _ in train_loader:
            images = torch.nn.functional.interpolate(images, size=(224, 224),
                                                     mode='bilinear',
                                                     align_corners=False
                                                    ).to(device, non_blocking=True)

            # ── similarity matrices K (frozen) and M (trainable) ──
            with torch.no_grad():
                out_frozen = frozen_model(images)
                out_frozen = out_frozen / out_frozen.norm(dim=1, keepdim=True)
                K = out_frozen @ out_frozen.t()

            out_train = model(images)
            out_train = out_train / out_train.norm(dim=1, keepdim=True)
            M = out_train @ out_train.t()

            exp, var = compute_expectation_variance(K, M, T)
            loss = beta * var - exp

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            run_loss += loss.item()
            run_exp  += exp.item()
            run_var  += var.item()
            n_batches += 1

        if n_batches:
            print(f'Epoch {epoch+1:02d}: '
                  f'loss {run_loss/n_batches:8.4f}, '
                  f'E {run_exp/n_batches:8.4f}, '
                  f'Var {run_var/n_batches:8.4f}')

    # ── Linear‑probe evaluation on 100 tasks ─────────────────────────────────
    labels_matrix = torch.from_numpy(np.stack(labels_list)).float()
    multi_ds      = MultiTaskWrapper(test_loader.dataset, labels_matrix)
    multi_loader  = DataLoader(multi_ds, batch_size=test_loader.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

    task_acc = linear_probe_multi(model, multi_loader, n_epoch=30,
                                  device=device)
    return task_acc


# -----------------------------------------------------------------------------#
#                                 Main entry                                   #
# -----------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--betas', default='0.5,1.0,5.0,10.0,100.0',
                    help='comma‑separated list of β values')
    ap.add_argument('--Ts',    default='0.1,0.5,1.0,2.0,5.0',
                    help='comma‑separated list of temperature T values')
    ap.add_argument('--epochs', type=int, default=30,
                    help='training epochs for each (β,T) run')
    ap.add_argument('--output', default='grid_search_results_full.json',
                    help='where to store the JSON results')
    ap.add_argument('--batch_size', type=int, default=512,
                    help='batch size for CIFAR‑10 training and evaluation')
    args = ap.parse_args()

    betas = [float(x) for x in args.betas.split(',') if x]
    Ts    = [float(x) for x in args.Ts.split(',') if x]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # ── data (shared across runs) ────────────────────────────────────────────
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)

    # ── load the 100‑task label list ─────────────────────────────────────────
    if not os.path.exists('labels_list_new_algo_1.npy'):
        raise FileNotFoundError(
            'labels_list_new_algo_1.npy not found – please place it next to '
            'this script or pass the correct path.')
    labels_list = np.load('labels_list_new_algo_1.npy', allow_pickle=True)

    results: List[Dict] = []
    for beta in betas:
        for T in Ts:
            start = time.time()
            acc = train_and_evaluate(beta, T,
                                     train_loader, test_loader,
                                     labels_list,
                                     n_epoch=args.epochs,
                                     device=device)
            runtime = time.time() - start
            results.append({
                'beta': beta,
                'T':    T,
                'mean_accuracy': float(acc.mean()),
                'accuracies':    acc.tolist(),
                'seconds':       runtime,
            })
            # flush intermediate result in case of interruption
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'β={beta:4.2f}, T={T:4.2f} → '
                  f'mean acc {acc.mean():5.2f} (100‑task) '
                  f'in {runtime/60:.1f} min')

    print(f'Grid‑search complete. Results saved to: {args.output}')

if __name__ == '__main__':
    main()
