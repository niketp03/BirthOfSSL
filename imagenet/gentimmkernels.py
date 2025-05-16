#!/usr/bin/env python
"""
mini_imagenet_kernels_per_model.py
----------------------------------
• Uses the open timm/mini‑imagenet dataset (50 k images, 100 classes)
• For each timm encoder in MODEL_NAMES:
      – extracts features on N_IMAGES random samples
      – builds its own cosine‑similarity kernel  K = Z Zᵀ
      – saves to  kernels_out/K_<model>.pt
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

import gc


# ───────────── user‑tweakables ─────────────
N_IMAGES   = 8_192
BATCH_SIZE = 1024
NUM_WORKERS = 0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAMES = ['deit_tiny_patch16_224', 'deit_small_patch16_224', 'pvt_v2_b2',
       'resnet34', 'resnet18', 'beit_base_patch16_224', 'convnext_small',
       'vit_base_patch16_224', 'mixer_b16_224',
       'swin_tiny_patch4_window7_224', 'regnety_016',
       'beit_large_patch16_224', 'gcvit_base', 'convnextv2_base',
       'ese_vovnet39b', 'densenet121', 'ghostnet_100', 'efficientnet_b0',
       'mobilenetv3_large_100', 'regnety_032', 'densenet201', 'resnet152',
       'resnet101', 'wide_resnet50_2', 'resnext50_32x4d',
       'resnext101_32x8d', 'wide_resnet101_2', 'resnet50', 'dm_nfnet_f0',
       'deit_base_patch16_224', 'swin_base_patch4_window7_224',
       'convnext_base', 'efficientnet_b5']


OUT_DIR = Path("kernels_out")
OUT_DIR.mkdir(exist_ok=True)
# ───────────────────────────────────────────


# 1) Dataset -------------------------------------------------------------------
print("📦  loading timm/mini‑imagenet …")
hf_ds = load_dataset("timm/mini-imagenet", split="train")

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),   # ensure 3‑ch
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class HFWrapper(Dataset):
    def __init__(self, ds, tfm):
        self.ds, self.tfm = ds, tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[int(i)]
        return self.tfm(item["image"]), item["label"]

# sample once, reuse for every model
full_ds   = HFWrapper(hf_ds, transform)
indices   = random.sample(range(len(full_ds)), N_IMAGES)
subset_ds = Subset(full_ds, indices)
loader    = DataLoader(subset_ds, batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=NUM_WORKERS,
                       pin_memory=True)
print(f"✓ dataset ready — {len(subset_ds)} images\n")


# 2) Feature → kernel → save  (one loop per model) -----------------------------

@torch.no_grad()
def features(model_name: str) -> torch.Tensor:
    model = timm.create_model(model_name, pretrained=True,
                              num_classes=0).to(DEVICE).eval()
    vecs = []
    for imgs, _ in tqdm(loader, desc=f"{model_name:>24}", leave=False):
        # Process in smaller batches if needed
        batch_output = model(imgs.to(DEVICE, non_blocking=True)).flatten(1)
        # Convert to float32 for better memory efficiency
        batch_output = batch_output.to(torch.float32)
        vecs.append(batch_output)
        # Explicitly free memory
        torch.cuda.empty_cache()
    print('done with looping')

    return torch.cat(vecs)


for m in MODEL_NAMES:
    print(f"🚀  processing {m} …")
    try:
        F_m = features(m)                            # (N, D_m)
        print('features computed')
        Z_m = F.normalize(F_m, p=2, dim=1)           # row‑norm
        K_m = Z_m @ Z_m.T                            # (N, N)

        torch.save(
            {"K": K_m.cpu(),                         # kernel
             "Z": F_m.cpu(),                         # normalised feats
             "dim": F_m.shape[1],                    # feature length of this model
             "indices": indices},
            OUT_DIR / f"K_{m}_{N_IMAGES}.pt"
        )
        print(f"   ↳ saved  {OUT_DIR / f'K_{m}_{N_IMAGES}.pt'}\n")

        # Clean up memory before next model
        del F_m, Z_m, K_m
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
    except Exception as e:
        print(f"Error processing model {m}: {e}")
        torch.cuda.empty_cache()
        gc.collect()  # Still try to clean up memory on error

print("✅  all kernels done.")