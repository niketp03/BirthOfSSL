#!/usr/bin/env python3
"""
Debug script to run a single MIEB task directly.
"""

import mteb
import timm
import torch
import numpy as np

def debug_single_task():
    """Debug a single task execution."""
    
    # Get a single task
    benchmark = mteb.get_benchmarks(['MIEB(Multilingual)'])[0]
    task = benchmark.tasks[0]  # BirdsnapClassification
    
    print(f"Debugging task: {task}")
    print(f"Task type: {type(task)}")
    print(f"Task methods: {[m for m in dir(task) if not m.startswith('_')]}")
    
    # Load a simple model
    model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    model.eval()
    
    # Create wrapper
    class TimmModelWrapper:
        def __init__(self, model):
            self.model = model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model_name = "timm_model"
            self.max_seq_length = 512
            self.device = str(self.device)
            
        def encode(self, images, **kwargs):
            """Encode images to embeddings."""
            print(f"    encode called with: {type(images)}, length: {len(images) if hasattr(images, '__len__') else 'N/A'}")
            
            if isinstance(images, str):
                images = [images]
            
            embeddings = []
            for img_path in images:
                try:
                    print(f"    Processing image: {img_path}")
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
                        print(f"    Generated embedding shape: {embedding.shape}")
                        
                except Exception as e:
                    print(f"    Error processing {img_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return zero embedding on error
                    embedding_shape = self.model(torch.randn(1, 3, 224, 224).to(self.device)).shape[1]
                    embeddings.append(np.zeros((1, embedding_shape)))
            
            result = np.vstack(embeddings)
            print(f"    Returning embeddings shape: {result.shape}")
            return result
        
        def get_sentence_embedding_dimension(self):
            """Return the dimension of the sentence embeddings."""
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                dummy_output = self.model(dummy_input)
            return dummy_output.shape[1]
    
    model_wrapper = TimmModelWrapper(model)
    
    # Try to run the task directly
    print(f"\nTrying to run task directly...")
    
    try:
        # Check if task has data loaded
        if hasattr(task, 'data_loaded'):
            print(f"Task data loaded: {task.data_loaded}")
        
        if hasattr(task, 'load_data'):
            print("Loading task data...")
            task.load_data()
            print(f"Task data loaded: {task.data_loaded}")
        
        # Check what data the task has
        if hasattr(task, 'dataset'):
            print(f"Task dataset: {type(task.dataset)}")
            if hasattr(task.dataset, 'keys'):
                print(f"Dataset keys: {list(task.dataset.keys())}")
        
        # Try to evaluate
        if hasattr(task, 'evaluate'):
            print("Task has evaluate method, trying to call it...")
            try:
                result = task.evaluate(model_wrapper)
                print(f"Direct evaluation result: {result}")
            except Exception as e:
                print(f"Direct evaluation error: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"Task execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_task() 