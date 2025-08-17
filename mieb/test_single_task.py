#!/usr/bin/env python3
"""
Test script to evaluate a single MIEB task and see what's happening.
"""

import mteb
import timm
import torch
import numpy as np

def test_single_task():
    """Test evaluation on a single task."""
    
    # Get a single task
    benchmark = mteb.get_benchmarks(['MIEB(Multilingual)'])[0]
    single_task = [benchmark.tasks[0]]  # Just the first task
    
    print(f"Testing single task: {single_task[0]}")
    
    # Load a simple model
    model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    model.eval()
    
    # Create wrapper
    class TimmModelWrapper:
        def __init__(self, model):
            self.model = model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
        def encode(self, images, **kwargs):
            """Encode images to embeddings."""
            if isinstance(images, str):
                images = [images]
            
            embeddings = []
            for img_path in images:
                try:
                    # Load and preprocess image
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    # Standard ImageNet preprocessing
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.model(img_tensor)
                        embeddings.append(embedding.cpu().numpy())
                        
                except Exception as e:
                    print(f"    Error processing {img_path}: {e}")
                    # Return zero embedding on error
                    embedding_shape = self.model(torch.randn(1, 3, 224, 224).to(self.device)).shape[1]
                    embeddings.append(np.zeros((1, embedding_shape)))
            
            return np.vstack(embeddings)
    
    model_wrapper = TimmModelWrapper(model)
    
    # Run evaluation
    print("Running evaluation...")
    evaluation = mteb.MTEB(tasks=single_task)
    
    try:
        results = evaluation.run(model_wrapper)
        print(f"Results type: {type(results)}")
        print(f"Results: {results}")
        
        if isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            print(f"First result type: {type(first_result)}")
            print(f"First result: {first_result}")
            
            if hasattr(first_result, 'scores'):
                print(f"Scores: {first_result.scores}")
            else:
                print("No scores attribute")
                
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_task() 