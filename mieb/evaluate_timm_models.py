#!/usr/bin/env python3
"""
Simple script to evaluate timm models on MIEB benchmarks.
Focuses on image classification and image clustering tasks.
"""

import mteb
import timm
import torch
import json
import numpy as np
from typing import List, Dict, Any
import os

def get_mieb_tasks():
    """Get MIEB tasks for image classification and clustering."""
    benchmark = mteb.get_benchmarks(['MIEB(Multilingual)'])[0]
    
    # Get all tasks from the benchmark
    all_tasks = benchmark.tasks
    
    # Filter for image classification and clustering tasks
    image_tasks = []
    for task in all_tasks:
        task_name = str(task)
        if any(task_type in task_name.lower() for task_type in ['classification', 'clustering']):
            image_tasks.append(task)
    
    return image_tasks

def evaluate_model(model_name: str, tasks: List) -> Dict[str, Any]:
    """Evaluate a single timm model on MIEB tasks."""
    print(f"Evaluating {model_name}...")
    
    try:
        # Load the timm model
        model = timm.create_model(model_name, pretrained=True, num_classes=0)  # Remove classifier head
        model.eval()
        
        # Create a comprehensive wrapper for mteb compatibility
        class TimmModelWrapper:
            def __init__(self, model):
                self.model = model
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                
                # Add required attributes that MIEB expects
                self.model_name = "timm_model"
                self.max_seq_length = 512  # Default value
                self.device = str(self.device)
                
                # Add model_card_data to avoid warnings
                self.model_card_data = {
                    "name": "timm_model",
                    "language": ["en"],
                    "license": "unknown",
                    "tags": ["image-classification", "feature-extraction"]
                }
                
            def get_image_embeddings(self, images, **kwargs):
                """Get image embeddings - this is what MIEB expects."""
                # MIEB passes PIL Image objects, not file paths
                if not isinstance(images, list):
                    images = [images]
                
                embeddings = []
                for img in images:
                    try:
                        # Convert PIL Image to tensor
                        import torchvision.transforms as transforms
                        
                        # Standard ImageNet preprocessing
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
                        ])
                        
                        img_tensor = transform(img).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            embedding = self.model(img_tensor)
                            embeddings.append(embedding.cpu().numpy())
                            
                    except Exception as e:
                        print(f"    Error processing image: {e}")
                        # Return zero embedding on error
                        embedding_shape = self.model(torch.randn(1, 3, 224, 224).to(self.device)).shape[1]
                        embeddings.append(np.zeros((1, embedding_shape)))
                
                return np.vstack(embeddings)
            
            def encode(self, images, **kwargs):
                """Encode images to embeddings - alias for get_image_embeddings."""
                return self.get_image_embeddings(images, **kwargs)
            
            def __call__(self, images, **kwargs):
                """Make the wrapper callable like the original model."""
                return self.get_image_embeddings(images, **kwargs)
            
            def get_sentence_embedding_dimension(self):
                """Return the dimension of the sentence embeddings."""
                # Get embedding dimension from a dummy forward pass
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    dummy_output = self.model(dummy_input)
                return dummy_output.shape[1]
        
        # Create wrapper
        model_wrapper = TimmModelWrapper(model)
        
        # Evaluate on tasks - try one task first for debugging
        results = {}
        
        # Try with just one task first to debug
        test_tasks = tasks[:1]  # Just the first task
        print(f"  Testing with 1 task first: {test_tasks[0]}")
        
        evaluation = mteb.MTEB(tasks=test_tasks)
        
        # Add debugging
        print(f"  Model wrapper type: {type(model_wrapper)}")
        print(f"  Model wrapper methods: {[m for m in dir(model_wrapper) if not m.startswith('_')]}")
        print(f"  Model embedding dimension: {model_wrapper.get_sentence_embedding_dimension()}")
        
        try:
            print(f"  Starting evaluation...")
            task_results = evaluation.run(model_wrapper)
            print(f"  Evaluation completed. Results type: {type(task_results)}")
            print(f"  Results length: {len(task_results) if isinstance(task_results, list) else 'N/A'}")
            
            # Process results - task_results is a list, not a dict
            if isinstance(task_results, list):
                for i, task_result in enumerate(task_results):
                    print(f"    Processing result {i+1}: {type(task_result)}")
                    
                    if hasattr(task_result, 'task_name'):
                        task_name = task_result.task_name
                    else:
                        task_name = str(task_result)
                    
                    if hasattr(task_result, 'scores'):
                        # Extract main score
                        main_score = task_result.scores.get('main', 0.0)
                        results[task_name] = {
                            "score": float(main_score),
                            "status": "completed",
                            "scores": dict(task_result.scores)
                        }
                        print(f"      Task {task_name}: score = {main_score}")
                    else:
                        results[task_name] = {
                            "score": 0.0,
                            "status": "no_scores",
                            "note": "Task completed but no scores available"
                        }
                        print(f"      Task {task_name}: no scores")
            else:
                # Fallback if results are in different format
                results = {"error": f"Unexpected result format: {type(task_results)}"}
                print(f"    Unexpected result format: {type(task_results)}")
                
        except Exception as e:
            print(f"    Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            results = {"error": str(e)}
        
        # If the single task worked, try all tasks
        if results and not any("error" in str(v) for v in results.values()):
            print(f"  Single task worked! Now trying all {len(tasks)} tasks...")
            try:
                full_evaluation = mteb.MTEB(tasks=tasks)
                full_results = full_evaluation.run(model_wrapper)
                
                # Process full results
                for task_result in full_results:
                    if hasattr(task_result, 'task_name'):
                        task_name = task_result.task_name
                    else:
                        task_name = str(task_result)
                    
                    if hasattr(task_result, 'scores'):
                        main_score = task_result.scores.get('main', 0.0)
                        results[task_name] = {
                            "score": float(main_score),
                            "status": "completed",
                            "scores": dict(task_result.scores)
                        }
                    else:
                        results[task_name] = {
                            "score": 0.0,
                            "status": "no_scores",
                            "note": "Task completed but no scores available"
                        }
                        
            except Exception as e:
                print(f"    Full evaluation error: {e}")
                results["full_evaluation_error"] = str(e)
        
        return results
        
    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    """Main function to evaluate multiple timm models."""
    
    # Get MIEB tasks
    print("Loading MIEB tasks...")
    tasks = get_mieb_tasks()
    print(f"Found {len(tasks)} image-related tasks")
    
    # Show some example tasks
    print("\nExample tasks:")
    for i, task in enumerate(tasks[:5]):
        print(f"  {i+1}. {task}")
    if len(tasks) > 5:
        print(f"  ... and {len(tasks) - 5} more tasks")
    
    # Define models to test (you can modify this list)
    model_names = [
        "resnet18",
        "resnet50"
    ]
    
    # Evaluate each model
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        results = evaluate_model(model_name, tasks)
        all_results[model_name] = results
        
        # Save individual model results
        output_file = f"results/{model_name}_mieb_results.json"
        os.makedirs("results", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    # Save combined results
    combined_file = "results/all_models_mieb_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nCombined results saved to {combined_file}")
    print("\nNote: This is a simplified implementation.")
    print("You'll need to implement proper model wrapping for mteb compatibility.")

if __name__ == "__main__":
    main() 