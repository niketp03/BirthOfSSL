#!/usr/bin/env python3
"""
Simple working evaluation script for timm models on MIEB tasks.
This bypasses the complex MIEB framework and directly evaluates models.
"""

import mteb
import timm
import torch
import numpy as np
import json
import os
from typing import List, Dict, Any
from PIL import Image
import torchvision.transforms as transforms

def get_mieb_tasks():
    """Get MIEB tasks for image classification and clustering."""
    benchmark = mteb.get_benchmarks(['MIEB(Multilingual)'])[0]
    return benchmark.tasks

def create_model_wrapper(model_name: str):
    """Create a working model wrapper."""
    # Load the timm model
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    
    # Create a simple but working wrapper
    class SimpleModelWrapper:
        def __init__(self, model):
            self.model = model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            # Standard ImageNet preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def get_embeddings(self, images, batch_size=32):
            """Get embeddings for a list of PIL images in batches."""
            if not isinstance(images, list):
                images = [images]
            
            embeddings = []
            total_images = len(images)
            
            for i in range(0, total_images, batch_size):
                batch_images = images[i:i + batch_size]
                batch_embeddings = []
                
                for img in batch_images:
                    try:
                        # Convert PIL image to tensor
                        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            embedding = self.model(img_tensor)
                            batch_embeddings.append(embedding.cpu().numpy())
                            
                    except Exception as e:
                        print(f"    Error processing image: {e}")
                        # Return zero embedding on error
                        embedding_shape = self.model(torch.randn(1, 3, 224, 224).to(self.device)).shape[1]
                        batch_embeddings.append(np.zeros((1, embedding_shape)))
                
                embeddings.extend(batch_embeddings)
                print(f"      Processed {min(i + batch_size, total_images)}/{total_images} images")
            
            return np.vstack(embeddings)
    
    return SimpleModelWrapper(model)

def evaluate_single_task(model_wrapper, task):
    """Evaluate a single task with the model."""
    print(f"  Evaluating task: {task}")
    
    try:
        # Load task data
        if not task.data_loaded:
            task.load_data()
        
        # Get dataset
        dataset = task.dataset
        if 'train' in dataset and 'test' in dataset:
            train_data = dataset['train']
            test_data = dataset['test']
            
            print(f"    Train samples: {len(train_data)}")
            print(f"    Test samples: {len(test_data)}")
            
            # Use smaller subset for testing to avoid memory issues
            max_train_samples = 1000
            max_test_samples = 200
            
            if len(train_data) > max_train_samples:
                print(f"    Using subset of {max_train_samples} train samples")
                train_data = train_data.select(range(max_train_samples))
            
            if len(test_data) > max_test_samples:
                print(f"    Using subset of {max_test_samples} test samples")
                test_data = test_data.select(range(max_test_samples))
            
            # Get embeddings for train and test data
            print(f"    Getting train embeddings...")
            train_images = train_data[task.image_column_name]
            train_embeddings = model_wrapper.get_embeddings(train_images)
            
            print(f"    Getting test embeddings...")
            test_images = test_data[task.image_column_name]
            test_embeddings = model_wrapper.get_embeddings(test_images)
            
            # Simple k-NN classification
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score
            
            # Train k-NN classifier
            print(f"    Training k-NN classifier...")
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(train_embeddings, train_data[task.label_column_name])
            
            # Predict and evaluate
            print(f"    Making predictions...")
            predictions = knn.predict(test_embeddings)
            accuracy = accuracy_score(test_data[task.label_column_name], predictions)
            
            print(f"    Accuracy: {accuracy:.4f}")
            
            return {
                "score": float(accuracy),
                "status": "completed",
                "method": "k-NN",
                "train_samples": len(train_data),
                "test_samples": len(test_data)
            }
            
        else:
            print(f"    No train/test split found")
            return {
                "score": 0.0,
                "status": "no_data",
                "note": "No train/test split available"
            }
            
    except Exception as e:
        print(f"    Task evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "score": 0.0,
            "status": "error",
            "error": str(e)
        }

def main():
    """Main evaluation function."""
    
    # Get MIEB tasks
    print("Loading MIEB tasks...")
    tasks = get_mieb_tasks()
    print(f"Found {len(tasks)} tasks")
    
    # Filter for image classification tasks only (simpler to evaluate)
    image_tasks = [task for task in tasks if 'Classification' in str(task) and 'ZeroShot' not in str(task)]
    print(f"Found {len(image_tasks)} image classification tasks")
    
    # Define models to test
    model_names = [
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "pvt_v2_b2",
        "resnet34",
        "resnet18",
        "beit_base_patch16_224",
        "convnext_small",
        "vit_base_patch16_224",
        "mixer_b16_224",
        "swin_tiny_patch4_window7_224",
        "regnety_016",
        "beit_large_patch16_224",
        "gcvit_base",
        "convnextv2_base",
        "ese_vovnet39b",
        "densenet121",
        "ghostnet_100",
        "efficientnet_b0",
        "mobilenetv3_large_100",
        "regnety_032",
        "densenet201",
        "resnet152",
        "resnet101",
        "wide_resnet50_2",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet101_2",
        "resnet50",
        "dm_nfnet_f0",
        "deit_base_patch16_224",
        "swin_base_patch4_window7_224",
        "convnext_base",
        "efficientnet_b5"
    ]

    
    # Evaluate each model
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        # Create model wrapper
        model_wrapper = create_model_wrapper(model_name)
        
        # Evaluate on all available image classification tasks
        test_tasks = image_tasks  # Use all tasks
        print(f"Evaluating on {len(test_tasks)} tasks...")
        
        results = {}
        for task in test_tasks:
            result = evaluate_single_task(model_wrapper, task)
            task_name = str(task)
            results[task_name] = result
        
        all_results[model_name] = results
        
        # Save individual model results
        output_file = f"results/{model_name}_simple_results.json"
        os.makedirs("results", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    # Save combined results
    combined_file = "results/all_models_simple_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nCombined results saved to {combined_file}")
    print("\nThis is a simplified but working implementation!")

if __name__ == "__main__":
    main() 