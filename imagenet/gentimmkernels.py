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



MODEL_NAMES = [
    # ── Classic CNNs ───────────────────────────────────────────────────────
    #"resnet18",
    #"resnet34",
    #"resnet50",
    #"resnet101",
    #"resnet152",
    #"wide_resnet50_2",
    #"wide_resnet101_2",
    #"resnext50_32x4d",
    #"resnext101_32x8d",
    #"densenet121",
    #"densenet201",
    #"ese_vovnet39b",
    #"regnety_016",
    #"regnety_032",
    # ConvNeXt family
    #"convnext_small",
    # EfficientNet & friends
    #"efficientnet_b0",
    # Mobile / lightweight
    #"mobilenetv3_large_100",
    #"ghostnet_100",
    # NF‑Nets (DeepMind)
    #"dm_nfnet_f0",
    # ── Vision Transformers & hybrids ─────────────────────────────────────
    # ViT
    #"vit_base_patch16_224",
    # DeiT
    #"deit_tiny_patch16_224",
    #"deit_small_patch16_224",
    # BEiT
    #"beit_base_patch16_224",
    #"beit_large_patch16_224",
    # Swin
    #"swin_tiny_patch4_window7_224",
    # PVT‑v2
    #"pvt_v2_b2",
    # CSWin
    #"cswin_tiny_224",
    # CoAtNet
    #"coatnet_0",
    # Mixers / Convmixer
    #"mixer_b16_224",
    # GC ViT
    #"gcvit_base",
    # ConvNeXt‑v2
    #"convnextv2_base",
    # CLIP ViT (image branch only)
    #"clip_vit_base_patch32",
]

MODEL_NAMES = [
    # ── Classic CNNs ───────────────────────────────────────────────────────
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "densenet121",
    "densenet201",
    "ese_vovnet39b",
    "regnety_016",
    "regnety_032"
]

MODEL_NAMES = [
    # ─────────────── ResNet family ───────────────
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnet26d", "resnet50d", "resnet50_gn", "resnet200d",
    "wide_resnet50_2", "wide_resnet101_2",
    "resnext50_32x4d", "resnext101_32x8d", "bat_resnext26ts",
    "resnest50d", "resnest101e", "resnest200e", "resnest269e",
    "seresnet50", "seresnet152", "seresnext50_32x4d",
    "skresnet50", "skresnext50_32x4d",

    # ─────────────── Dense / DPN / HRNet ─────────
    "densenet121", "densenet161", "densenet201",
    "tv_densenet169", "densenet264d_iabn",
    "dpn68", "dpn92", "dpn131",
    "hrnet_w18", "hrnet_w30", "hrnet_w48",
    
    # ─────────────── EfficientNet & friends ──────
    "efficientnet_b0", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnet_lite0", "efficientnet_lite4",
    "efficientnetv2_rw_t", "efficientnetv2_rw_s", "efficientnetv2_rw_b",

    # ─────────────── ConvNeXt & ConvNeXt‑v2 ──────
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "convnextv2_tiny", "convnextv2_base", "convnextv2_large",

    # ─────────────── RegNet / NF‑Net / VovNet ────
    "regnetx_002", "regnetx_004", "regnetx_016", "regnetx_064",
    "regnety_004", "regnety_016", "regnety_032", "regnety_080",
    "dm_nfnet_f0", "dm_nfnet_f3",
    "ese_vovnet19b_dw", "ese_vovnet39b", "vovnet99",

    # ─────────────── Mobile / Edge models ────────
    "mobilenetv2_100", "mobilenetv3_large_100", "mobilenetv3_small_075",
    "mobilenetv3_rw", "hardnet85",
    "ghostnet_100", "ghostnetv2_pico", "fbnetv3_b",
    "mnasnet_100", "spnasnet_100", "efficientnet_es",

    # ─────────────── Inception / Xception ────────
    "inception_v3", "adv_inception_v3",
    "inception_v4", "inception_resnet_v2",
    "inception_next_atto", "inception_next_small",
    "xception", "xception41",
    
    # ─────────────── Misc. specialised CNNs ──────
    "darknet53", "cspdarknet53", "dla46x12", "ecaresnet50d",
    "sele_resnext26_32x4d", "gla_resnet26ts",
    "repvgg_a2", "repvgg_b3g4",
    "squeezenet1_0", "squeezenet1_1",
    "vgg11_bn", "vgg16_bn", "vgg19_bn",
]


OUT_DIR = Path("kernels_out_all_cnns")
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