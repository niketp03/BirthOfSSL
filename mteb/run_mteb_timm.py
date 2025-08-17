# mieb_image_only_timm.py
import os
from typing import List, Union, Iterable

import torch
import torch.nn as nn
import timm
from PIL import Image

import mteb
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class TimmImageEncoder(nn.Module):
    """
    Minimal image->vector wrapper for MIEB/MTEB image-only tasks.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", device: str = None, half: bool = False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        self.model.eval().to(self.device)
        if half and self.device.startswith("cuda"):
            self.model.half()

        # Build an eval transform from TIMM's data config
        cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**cfg)

        # Cache embedding dimension
        test = torch.zeros(1, 3, cfg["input_size"][1], cfg["input_size"][2])
        with torch.no_grad():
            emb = self.model(test.to(self.device))
        self.embedding_dimension = emb.shape[-1]
        
        # MTEB model metadata - these are the key attributes MTEB looks for
        self.modalities = ["image"]
        self.max_length = None  # Not applicable for images
        self.max_seq_length = None  # Not applicable for images
        
        # Set attributes directly on the model object for MTEB compatibility
        self.model_name = model_name
        self.revision = "main"
        self.release_date = None
        self.languages = None
        self.n_parameters = None
        self.memory_usage_mb = None
        self.max_tokens = None
        self.embed_dim = self.embedding_dimension
        self.license = None
        self.open_weights = True
        self.public_training_code = None
        self.public_training_data = None
        self.framework = ["pytorch"]
        self.reference = None
        self.similarity_fn_name = "cosine"
        self.use_instructions = False
        self.training_datasets = None
        self.adapted_from = None
        self.superseded_by = None
        self.is_cross_encoder = False
        self.loader = None
        
        # Add model_card_data attribute that MTEB expects
        from types import SimpleNamespace
        self.model_card_data = SimpleNamespace(
            name=model_name,
            revision="main",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=self.embedding_dimension,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["pytorch"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
            is_cross_encoder=False,
            loader=None
        )

    @torch.inference_mode()
    def _encode_tensor_batch(self, pixel_batch: torch.Tensor) -> torch.Tensor:
        feats = self.model(pixel_batch.to(self.device))
        feats = nn.functional.normalize(feats, dim=-1)
        return feats

    def _preprocess(self, ims: Iterable[Image.Image]) -> torch.Tensor:
        tensors = []
        for im in ims:
            if not isinstance(im, Image.Image):
                im = Image.open(im).convert("RGB")
            tensors.append(self.transform(im))
        return torch.stack(tensors, dim=0)

    # MTEB expects encode() to return a numpy array for image models
    def encode(self, sentences: List[Union[Image.Image, str]], *, task_name: str, prompt_type=None, batch_size: int = 32, **kwargs):
        # For image tasks, sentences are actually images
        images = sentences
        embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pixel_batch = self._preprocess(batch)
            out = self._encode_tensor_batch(pixel_batch)
            embs.append(out.detach().cpu())
        embs = torch.cat(embs, dim=0)
        return embs.numpy()  # Return numpy array instead of list


if __name__ == "__main__":
    # List of timm models to evaluate
    # Easy to expand by adding more model names to this list
    TIMM_MODELS = [
        "vit_base_patch16_224"
        # Add more models here as needed:
        # "resnet50",
        # "efficientnet_b0",
        # "swin_base_patch16_224",
        # etc.
    ]
    
    print(f"Evaluating {len(TIMM_MODELS)} timm models: {TIMM_MODELS}")
    
    # Filter to image-only tasks
    tasks = mteb.get_tasks(
        task_types=["ImageClassification", "ImageClustering", "RetrievalImageToImage"]
    )

    print(f"Loaded {len(tasks)} image-only tasks from MIEB.")
    
    # Print task details for debugging
    for i, task in enumerate(tasks[:3]):  # Just show first 3 tasks
        print(f"Task {i}: {task}")
        print(f"  Task type: {type(task)}")
        print(f"  Task name: {getattr(task, 'name', 'unknown')}")
        print(f"  Task description: {getattr(task, 'description', 'unknown')}")
        print(f"  Task type: {getattr(task, 'type', 'unknown')}")
        print()

    # Evaluate each model
    for model_name in TIMM_MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Wrap the timm model as an encoder
            model = TimmImageEncoder(model_name=model_name, half=False)
            
            # Print model info
            print(f"Model: {model}")
            print(f"Model modalities: {getattr(model, 'modalities', 'unknown')}")
            print(f"Model embedding dimension: {getattr(model, 'embedding_dimension', 'unknown')}")
            
            # Actually run MTEB evaluation
            print(f"\nRunning MTEB evaluation for {model_name}...")
            
            # Create MTEB evaluation pipeline
            from mteb.evaluation import MTEB
            evaluator = MTEB(tasks=tasks)
            
            # Run evaluation
            results = evaluator.run(
                model, 
                verbosity=2, 
                output_folder=f"results_{model_name}",
                overwrite_results=True
            )
            
            # Print results summary
            print(f"\nResults for {model_name}:")
            for result in results:
                print(f"  {result.task_name}:")
                if hasattr(result, 'scores') and result.scores:
                    for metric, value in result.scores.items():
                        if isinstance(value, (int, float)):
                            print(f"    {metric}: {value:.4f}")
                        else:
                            print(f"    {metric}: {value}")
                else:
                    print(f"    No scores available")
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            continue

    print("\nMTEB evaluation complete for all models.")
    print("Results have been saved to individual output folders.") 