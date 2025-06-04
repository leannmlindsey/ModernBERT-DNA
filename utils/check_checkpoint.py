#!/usr/bin/env python
"""Quick script to check what's in a checkpoint file."""

import torch
import sys
from typing import Any, Dict, List, Set

def find_metrics_recursive(obj: Any, path: str = "", depth: int = 0, max_depth: int = 5) -> List[tuple]:
    """Recursively search for metric-like values in nested structures."""
    metrics = []
    metric_keywords = ['loss', 'perplexity', 'accuracy', 'metric', 'score', 'eval', 
                      'train', 'val', 'test', 'f1', 'precision', 'recall', 'mcc', 
                      'auc', 'bleu', 'rouge', 'acc']
    
    if depth > max_depth:
        return metrics
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            
            # Check if key contains metric keywords
            key_lower = str(key).lower()
            if any(keyword in key_lower for keyword in metric_keywords):
                if isinstance(value, (int, float)):
                    metrics.append((new_path, value))
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    metrics.append((new_path, value.item()))
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                    metrics.append((new_path, f"List[{len(value)}]: latest={value[-1]}"))
            
            # Recurse into nested structures
            metrics.extend(find_metrics_recursive(value, new_path, depth + 1, max_depth))
    
    elif isinstance(obj, list) and len(obj) > 0:
        # Check first element to avoid processing huge lists
        if isinstance(obj[0], dict):
            metrics.extend(find_metrics_recursive(obj[0], f"{path}[0]", depth + 1, max_depth))
    
    return metrics


if len(sys.argv) != 2:
    print("Usage: python check_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check what's in the checkpoint
if isinstance(checkpoint, dict):
    print("\nCheckpoint is a dictionary with keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # If it has 'state_dict' key, examine it
    if 'state_dict' in checkpoint:
        print("\nExamining 'state_dict':")
        state_dict = checkpoint['state_dict']
        
        # Get all unique prefixes
        prefixes = set()
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) > 1:
                prefixes.add(parts[0])
        
        print(f"Unique prefixes in state_dict: {prefixes}")
        
        # Show first 10 keys
        print("\nFirst 10 keys in state_dict:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}")
        
        # Check for model wrapper prefixes
        example_key = list(state_dict.keys())[0]
        print(f"\nExample key: {example_key}")
        
        # Count layers
        layer_keys = [k for k in state_dict.keys() if 'encoder.layers' in k]
        layer_numbers = set()
        for k in layer_keys:
            if 'encoder.layers.' in k:
                parts = k.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i+1 < len(parts):
                        try:
                            layer_num = int(parts[i+1])
                            layer_numbers.add(layer_num)
                        except:
                            pass
        
        if layer_numbers:
            print(f"\nFound layers: {sorted(layer_numbers)}")
            print(f"Total number of layers: {len(layer_numbers)}")
    
    # Also check 'state' if it exists
    if 'state' in checkpoint:
        print("\nCheckpoint has 'state' key")
        state = checkpoint['state']
        if isinstance(state, dict):
            print(f"State dict keys: {list(state.keys())}")
            
            if 'model' in state:
                print("\nFound model in checkpoint['state']['model']")
                model_state = state['model']
                
                # Show first 10 model keys
                print("\nFirst 10 keys in state['model']:")
                for i, key in enumerate(list(model_state.keys())[:10]):
                    print(f"  {key}")
                
                # Count layers in state['model']
                layer_keys = [k for k in model_state.keys() if 'encoder.layers' in k]
                layer_numbers = set()
                for k in layer_keys:
                    if 'encoder.layers.' in k:
                        parts = k.split('.')
                        for i, part in enumerate(parts):
                            if part == 'layers' and i+1 < len(parts):
                                try:
                                    layer_num = int(parts[i+1])
                                    layer_numbers.add(layer_num)
                                except:
                                    pass
                
                if layer_numbers:
                    print(f"\nFound layers in state['model']: {sorted(layer_numbers)}")
                    print(f"Total number of layers: {len(layer_numbers)}")
    
    # Check for metrics
    print("\n" + "="*50)
    print("SEARCHING FOR METRICS")
    print("="*50)
    
    # Common places where metrics might be stored
    metrics_found = False
    
    # Check direct keys for metrics
    metric_keywords = ['loss', 'perplexity', 'accuracy', 'metric', 'score', 'eval', 'train', 'val', 'test']
    for key in checkpoint.keys():
        for keyword in metric_keywords:
            if keyword in key.lower():
                print(f"\nFound potential metric key: '{key}'")
                value = checkpoint[key]
                if isinstance(value, (int, float)):
                    print(f"  Value: {value}")
                elif isinstance(value, dict):
                    print(f"  Dict with keys: {list(value.keys())}")
                    # Print some values if they're numeric
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            print(f"    {k}: {v}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"  List with {len(value)} items")
                    if isinstance(value[0], (int, float)):
                        print(f"    First few values: {value[:5]}")
                metrics_found = True
    
    # Check in 'state' for metrics
    if 'state' in checkpoint and isinstance(checkpoint['state'], dict):
        state = checkpoint['state']
        
        # Check for common metric storage locations
        metric_locations = ['metrics', 'train_metrics', 'eval_metrics', 'test_metrics', 
                          'loss', 'losses', 'history', 'logs', 'results']
        
        for loc in metric_locations:
            if loc in state:
                print(f"\nFound '{loc}' in checkpoint['state']:")
                metrics_found = True
                value = state[loc]
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            print(f"  {k}: {v}")
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                            print(f"  {k}: {v[-1]} (latest of {len(v)} values)")
                elif isinstance(value, (int, float)):
                    print(f"  Value: {value}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"  List with {len(value)} items")
                    if isinstance(value[0], (int, float)):
                        print(f"    Latest values: {value[-5:]}")
        
        # Check for step/epoch information
        for key in ['step', 'global_step', 'epoch', 'iteration', 'timestamp']:
            if key in state:
                print(f"\n{key}: {state[key]}")
                metrics_found = True
    
    # Check for optimizer state (might contain loss)
    if 'optimizer' in checkpoint:
        print("\nFound 'optimizer' in checkpoint")
        optimizer = checkpoint['optimizer']
        if isinstance(optimizer, dict) and 'state' in optimizer:
            print("  Optimizer has state")
            # Check if there's loss scale or other metrics
            for key in ['loss_scale', 'grad_scale', 'found_inf']:
                if key in optimizer:
                    print(f"  {key}: {optimizer[key]}")
                    metrics_found = True
    
    # Check for scheduler state (might contain learning rate info)
    if 'scheduler' in checkpoint or 'lr_scheduler' in checkpoint:
        scheduler_key = 'scheduler' if 'scheduler' in checkpoint else 'lr_scheduler'
        print(f"\nFound '{scheduler_key}' in checkpoint")
        scheduler = checkpoint[scheduler_key]
        if isinstance(scheduler, dict):
            for key in ['last_epoch', 'last_lr', '_last_lr', 'base_lrs']:
                if key in scheduler:
                    print(f"  {key}: {scheduler[key]}")
                    metrics_found = True
    
    # Check for any keys containing numbers that might be metrics
    print("\n" + "-"*30)
    print("Numeric values in top-level keys:")
    for key, value in checkpoint.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
            metrics_found = True
    
    if not metrics_found:
        print("\nNo obvious metrics found in the checkpoint.")
        print("The checkpoint might be a pure model state without training metrics.")
    
    # Do a deep recursive search for any missed metrics
    print("\n" + "="*50)
    print("DEEP RECURSIVE SEARCH FOR METRICS")
    print("="*50)
    
    all_metrics = find_metrics_recursive(checkpoint)
    if all_metrics:
        print(f"\nFound {len(all_metrics)} metric-like values:")
        # Group by metric type
        grouped_metrics = {}
        for path, value in all_metrics:
            # Extract the metric name from the path
            metric_name = path.split('.')[-1]
            if metric_name not in grouped_metrics:
                grouped_metrics[metric_name] = []
            grouped_metrics[metric_name].append((path, value))
        
        # Print grouped metrics
        for metric_name, metric_list in sorted(grouped_metrics.items()):
            print(f"\n{metric_name}:")
            for path, value in metric_list[:5]:  # Show max 5 examples per metric type
                print(f"  {path}: {value}")
            if len(metric_list) > 5:
                print(f"  ... and {len(metric_list) - 5} more")
    else:
        print("\nNo metrics found in deep search either.")
    
    # Print checkpoint file size
    import os
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Convert to MB
    print(f"\nCheckpoint file size: {file_size:.2f} MB")
    
else:
    print("\nCheckpoint is not a dictionary, it's a:", type(checkpoint))
    
print("\nDone!")