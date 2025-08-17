#!/usr/bin/env python3
"""
Example script showing how to customize the MIEB evaluation.
"""

# You can modify this list to test different timm models
CUSTOM_MODELS = [
    "resnet18",
    "resnet50", 
    "resnet101",
    "efficientnet_b0",
    "efficientnet_b1",
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "convnext_tiny",
    "maxvit_tiny_tf_512"
]

# To use these models, modify the evaluate_timm_models.py file:
# 1. Open evaluate_timm_models.py
# 2. Find the model_names list around line 80
# 3. Replace it with: model_names = CUSTOM_MODELS

print("Available models to test:")
for i, model in enumerate(CUSTOM_MODELS, 1):
    print(f"  {i:2d}. {model}")

print(f"\nTotal: {len(CUSTOM_MODELS)} models")
print("\nTo use these models:")
print("1. Copy the CUSTOM_MODELS list above")
print("2. Replace the model_names list in evaluate_timm_models.py")
print("3. Run: python evaluate_timm_models.py") 