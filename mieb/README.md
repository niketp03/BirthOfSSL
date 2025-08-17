# MIEB Timm Model Evaluation

This script evaluates timm models on MIEB (Multilingual) benchmarks, focusing on image classification and image clustering tasks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the evaluation script:
```bash
python evaluate_timm_models.py
```

## What it does

- Loads MIEB benchmarks and filters for image-related tasks
- Evaluates specified timm models (currently resnet18 and resnet50)
- Saves results for each model individually and combined
- Results are saved in the `results/` folder

## Customization

To test different models, modify the `model_names` list in the script:
```python
model_names = [
    "resnet18",
    "resnet50",
    "vit_base_patch16_224",
    "efficientnet_b0"
    # Add any timm model name
]
```

## Note

This is a simplified implementation. The actual model evaluation requires proper wrapping of timm models to be compatible with the mteb framework. The current version provides a structure that you can build upon. 