#!/usr/bin/env python
"""
hf_kernels_per_model.py
-----------------------
• Uses the HF "timm/mini-imagenet" dataset (50k images, 100 classes)
• For each *Hugging Face* vision encoder in HF_MODEL_IDS:
      – extracts features on N_IMAGES random samples
      – builds its own cosine-similarity kernel  K = Z Zᵀ
      – saves to  kernels_out/K_<sanitized_model_id>_<N>.pt

Fixes & features:
- Custom DataLoader collate_fn keeps images as PIL list (no default_collate errors).
- Transformers path: loads via AutoModel/AutoProcessor and **matches input dtype to model weights**.
- OpenCLIP fallback for LAION `laion/CLIP-ViT-*` repos from HF, with version-agnostic
  `create_model_from_pretrained` unpacking and **dtype-matched inputs** (no Half vs Float mismatch).
- Per-model log written to kernels_out/run_log.csv.
"""

import random
from pathlib import Path
import re
import gc
import csv
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

from transformers import AutoProcessor, AutoModel

# ───────────── user-tweakables ─────────────
N_IMAGES    = 8_192
BATCH_SIZE  = 512              # lower if you hit VRAM limits
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your HF-model subset (only entries with '/'; you can add/remove freely)
HF_MODEL_IDS = [
    "google/siglip-so400m-patch14-384",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",     # OpenCLIP fallback
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",     # OpenCLIP fallback
    "google/siglip-large-patch16-384",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",         # OpenCLIP fallback
    "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",         # OpenCLIP fallback
    "google/siglip-large-patch16-256",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",         # OpenCLIP fallback
    "google/siglip-base-patch16-512",
    "google/siglip-base-patch16-384",
    "laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",     # OpenCLIP fallback
    "google/siglip-base-patch16-256",
    "openai/clip-vit-large-patch14",
    "google/siglip-base-patch16-224",
    "google/siglip-base-patch16-256-multilingual",
    "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",     # OpenCLIP fallback
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",         # OpenCLIP fallback
    "nomic-ai/nomic-embed-vision-v1.5",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "jinaai/jina-clip-v1",
    "Salesforce/blip-itm-large-flickr",
    "kakaobrain/align-base",
    "Salesforce/blip-itm-large-coco",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip-itm-base-flickr",
    # Known problematic/invalid or not plain encoders (skip or add custom handling if you need):
    # "BAAI/bge-visualized-base",
    # "BAAI/bge-visualized-m3",
    # "royokong/e5-v",              # multi-modal VLM config (not a plain vision encoder via AutoModel)
    # "TIGER-Lab/VLM2Vec-LoRA",
    # "TIGER-Lab/VLM2Vec-Full",
]

OUT_DIR = Path("kernels_out")
OUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUT_DIR / "run_log.csv"
# ───────────────────────────────────────────

# Optional OpenCLIP fallback for LAION OpenCLIP repositories
TRY_OPENCLIP = True
try:
    import open_clip  # pip install open_clip_torch
except Exception:
    open_clip = None
    TRY_OPENCLIP = False

# ------------------------------ utils ------------------------------

def sanitize_model_id(mid: str) -> str:
    s = mid.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._\-+]+", "_", s)

def is_openclip_repo(mid: str) -> bool:
    # LAION OpenCLIP weights on HF (common pattern)
    return mid.lower().startswith("laion/clip-vit-")

# ------------------------------ dataset ------------------------------

print("📦  loading timm/mini-imagenet …")
hf_ds = load_dataset("timm/mini-imagenet", split="train")

class PILOnlyDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[int(i)]
        return item["image"].convert("RGB"), int(item["label"])

def collate_pils(batch):
    # Keep images as list of PILs; stack labels only
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels)

# sample once, reuse for every model
full_ds   = PILOnlyDataset(hf_ds)
indices   = random.sample(range(len(full_ds)), N_IMAGES)
subset_ds = Subset(full_ds, indices)
loader    = DataLoader(
    subset_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    collate_fn=collate_pils,   # <<< important
)

print(f"✓ dataset ready — {len(subset_ds)} images\n")

# ------------------------------ dtype helpers ------------------------------

def move_to_device_and_match_dtype(inputs: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device, non_blocking=True)
            if v.dtype.is_floating_point:
                v = v.to(dtype)
        out[k] = v
    return out

# ------------------------------ transformers path ------------------------------

@torch.no_grad()
def extract_features_for_batch_transformers(model, processor, pil_imgs):
    """
    Build inputs with processor, then cast float tensors to model dtype and run forward.
    Tries multiple common feature fields in order.
    """
    inputs = processor(images=pil_imgs, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    inputs = move_to_device_and_match_dtype(inputs, DEVICE, model_dtype)

    # 1) CLIP/SigLIP style
    if hasattr(model, "get_image_features"):
        feats = model.get_image_features(**inputs)
        return feats.flatten(1).to(torch.float32)

    # 2) Submodule vision backbone (e.g., BLIP)
    if hasattr(model, "vision_model") and "pixel_values" in inputs:
        vout = model.vision_model(pixel_values=inputs["pixel_values"])
        if hasattr(vout, "pooler_output") and vout.pooler_output is not None:
            feats = vout.pooler_output
        elif hasattr(vout, "last_hidden_state") and vout.last_hidden_state is not None:
            x = vout.last_hidden_state
            feats = x.mean(dim=1) if x.dim() == 3 else x
        else:
            raise RuntimeError("vision_model produced no usable outputs")
        return feats.flatten(1).to(torch.float32)

    # 3) Generic forward: look for common fields
    outputs = model(**inputs)
    if isinstance(outputs, dict):
        for key in ("image_embeds", "pooler_output", "last_hidden_state"):
            if key in outputs and outputs[key] is not None:
                x = outputs[key]
                if key == "last_hidden_state" and x.dim() == 3:
                    x = x.mean(dim=1)
                return x.flatten(1).to(torch.float32)
    else:
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds.flatten(1).to(torch.float32)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.flatten(1).to(torch.float32)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            x = outputs.last_hidden_state
            if x.dim() == 3:
                x = x.mean(dim=1)
            return x.flatten(1).to(torch.float32)

    raise RuntimeError("Could not find image features in model outputs.")

@torch.no_grad()
def features_hf_transformers(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(DEVICE).eval()
    vecs = []
    for pil_imgs, _ in tqdm(loader, desc=f"{model_id:>40}", leave=False):
        feats = extract_features_for_batch_transformers(model, processor, pil_imgs)
        vecs.append(feats.cpu())
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(vecs, dim=0)

# ------------------------------ openclip path ------------------------------

@torch.no_grad()
def extract_features_for_batch_openclip(oclip_model, oclip_preprocess, pil_imgs):
    # Match the batch dtype to the model's weight dtype (usually float32)
    weight_dtype = next(oclip_model.parameters()).dtype
    batch = torch.stack([oclip_preprocess(img) for img in pil_imgs], dim=0)
    batch = batch.to(device=DEVICE, dtype=weight_dtype, non_blocking=True)
    feats = oclip_model.encode_image(batch)
    return feats.float().flatten(1)  # keep outputs in float32 for stability

@torch.no_grad()
def features_hf_openclip(model_id: str):
    if open_clip is None:
        raise RuntimeError("open_clip not installed (pip install open_clip_torch)")

    # Version-agnostic unpacking
    res = open_clip.create_model_from_pretrained(f"hf-hub:{model_id}")
    if isinstance(res, tuple):
        model = res[0]
        preprocess = res[2] if len(res) >= 3 else res[1]
    else:
        model = res
        # Fallback preprocess if needed
        _, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32", pretrained=None
        )

    model = model.to(DEVICE).eval()

    vecs = []
    for pil_imgs, _ in tqdm(loader, desc=f"[open_clip] {model_id:>30}", leave=False):
        feats = extract_features_for_batch_openclip(model, preprocess, pil_imgs)
        vecs.append(feats.cpu())
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(vecs, dim=0)

# ------------------------------ main loop ------------------------------

def main():
    log_rows = []
    for mid in HF_MODEL_IDS:
        tag = sanitize_model_id(mid)
        print(f"\n🚀  processing {mid} …")
        try:
            if is_openclip_repo(mid):
                if TRY_OPENCLIP:
                    F_m = features_hf_openclip(mid)
                else:
                    raise RuntimeError("Needs OpenCLIP fallback (pip install open_clip_torch)")
            else:
                F_m = features_hf_transformers(mid)

            print("   ✓ features computed")
            Z_m = F.normalize(F_m, p=2, dim=1)
            K_m = Z_m @ Z_m.T

            out_path = OUT_DIR / f"K_{tag}_{N_IMAGES}.pt"
            torch.save(
                {"K": K_m, "Z": F_m, "dim": F_m.shape[1], "indices": indices, "model_id": mid},
                out_path,
            )
            print(f"   ↳ saved  {out_path}\n")
            log_rows.append({"model_id": mid, "status": "ok", "path": str(out_path), "dim": F_m.shape[1]})

            del F_m, Z_m, K_m
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"   ✗ skipped {mid} — {e}")
            log_rows.append({"model_id": mid, "status": f"error: {e}", "path": "", "dim": ""})
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_id", "status", "path", "dim"])
        w.writeheader()
        for r in log_rows:
            w.writerow(r)

    print("\n✅  done. See:", OUT_DIR, "and", LOG_PATH)

if __name__ == "__main__":
    main()
