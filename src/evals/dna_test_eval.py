# Copyright 2024 ModernBERT-DNA authors
# SPDX-License-Identifier: Apache-2.0

"""Test evaluation utilities for DNA benchmarks with comprehensive metrics."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_on_test_set(
    model,
    test_dataloader: DataLoader,
    task_name: str,
    num_labels: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: The trained model
        test_dataloader: DataLoader for test set
        task_name: Name of the task
        num_labels: Number of labels (2 for binary, 3+ for multiclass)
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics
    """
    # Handle HuggingFaceModel wrapper
    if hasattr(model, 'model'):
        # This is a HuggingFaceModel wrapper, get the underlying model
        actual_model = model.model
    else:
        actual_model = model
        
    actual_model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating {task_name} test set"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get model predictions
            # Rename labels to label for compatibility
            if 'labels' in batch:
                labels = batch.pop('labels')
            else:
                labels = batch.pop('label')
            
            # Forward pass with just input_ids and attention_mask
            outputs = actual_model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=labels
            )
            
            # Extract logits from output
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Collect predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'mcc': matthews_corrcoef(all_labels, all_predictions),
    }
    
    if num_labels == 2:
        # Binary classification metrics
        metrics['f1'] = f1_score(all_labels, all_predictions, average='binary')
        metrics['precision'] = precision_score(all_labels, all_predictions, average='binary')
        metrics['recall'] = recall_score(all_labels, all_predictions, average='binary')
    else:
        # Multi-class metrics
        metrics['f1_macro'] = f1_score(all_labels, all_predictions, average='macro')
        metrics['f1_weighted'] = f1_score(all_labels, all_predictions, average='weighted')
        metrics['precision_macro'] = precision_score(all_labels, all_predictions, average='macro')
        metrics['precision_weighted'] = precision_score(all_labels, all_predictions, average='weighted')
        metrics['recall_macro'] = recall_score(all_labels, all_predictions, average='macro')
        metrics['recall_weighted'] = recall_score(all_labels, all_predictions, average='weighted')
    
    return metrics


def print_test_metrics(metrics: Dict[str, float], task_name: str):
    """Pretty print test metrics."""
    print(f"\nTest Results for {task_name}:")
    print("-" * 40)
    
    # Print in a specific order
    metric_order = ['accuracy', 'mcc', 'f1', 'precision', 'recall', 
                    'f1_macro', 'f1_weighted', 'precision_macro', 
                    'precision_weighted', 'recall_macro', 'recall_weighted']
    
    for metric in metric_order:
        if metric in metrics:
            print(f"{metric:20s}: {metrics[metric]:.4f}")
    
    print("-" * 40)