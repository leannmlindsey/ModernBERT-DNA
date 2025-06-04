#!/usr/bin/env python
"""Extract and display information from a WandB run directory."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def load_json_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file and return its contents."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_config_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant configuration information."""
    info = {}
    
    # Key configuration fields to extract
    config_fields = [
        'run_name',
        'tokenizer_name',
        'mlm_probability',
        'data_local',
        'device_train_batch_size',
        'device_train_microbatch_size',
        'device_eval_batch_size',
        'n_gpus',
        'max_duration',
        'max_seq_len',
        'model_name',
        'vocab_size',
        'learning_rate',
        'weight_decay',
        'num_warmup_steps',
        'seed',
    ]
    
    # Extract values from config
    for field in config_fields:
        if field in config:
            info[field] = config[field]['value'] if isinstance(config[field], dict) else config[field]
    
    # Also check nested locations
    if 'model' in config:
        model_config = config['model']['value'] if isinstance(config['model'], dict) else config['model']
        if isinstance(model_config, dict):
            for key in ['name', 'pretrained_model_name', 'model_config']:
                if key in model_config:
                    info[f'model_{key}'] = model_config[key]
    
    if 'optimizer' in config:
        opt_config = config['optimizer']['value'] if isinstance(config['optimizer'], dict) else config['optimizer']
        if isinstance(opt_config, dict):
            if 'lr' in opt_config:
                info['learning_rate'] = opt_config['lr']
    
    return info


def extract_final_metrics(wandb_dir: Path) -> Dict[str, float]:
    """Extract final loss and perplexity from wandb logs."""
    metrics = {}
    
    # Check for wandb summary file
    summary_file = wandb_dir / 'files' / 'wandb-summary.json'
    if summary_file.exists():
        summary = load_json_file(summary_file)
        if summary:
            # Look for final metrics
            metric_keys = ['loss', 'train/loss', 'eval/loss', 'perplexity', 'train/perplexity', 'eval/perplexity',
                          'train_loss', 'eval_loss', 'train_perplexity', 'eval_perplexity']
            for key in metric_keys:
                if key in summary:
                    metrics[key] = summary[key]
    
    # Check for history file (contains all logged metrics)
    history_file = wandb_dir / 'files' / 'wandb-history.jsonl'
    if history_file.exists() and not metrics:
        # Read last line of history for final metrics
        try:
            with open(history_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = json.loads(lines[-1])
                    for key in ['loss', 'train/loss', 'eval/loss', 'perplexity', 'train/perplexity', 'eval/perplexity']:
                        if key in last_line:
                            metrics[f'final_{key}'] = last_line[key]
        except Exception as e:
            print(f"Error reading history file: {e}")
    
    return metrics


def pretty_print_info(info: Dict[str, Any], metrics: Dict[str, float], run_dir: str):
    """Pretty print the extracted information."""
    print("=" * 70)
    print(f"WANDB RUN INFORMATION")
    print(f"Directory: {run_dir}")
    print("=" * 70)
    
    # Group information by category
    categories = {
        'Run Information': ['run_name', 'seed'],
        'Model Configuration': ['model_name', 'tokenizer_name', 'vocab_size', 'mlm_probability'],
        'Data Configuration': ['data_local', 'max_seq_len'],
        'Training Configuration': ['device_train_batch_size', 'device_train_microbatch_size', 
                                 'device_eval_batch_size', 'n_gpus', 'max_duration'],
        'Optimizer Configuration': ['learning_rate', 'weight_decay', 'num_warmup_steps'],
    }
    
    for category, fields in categories.items():
        print(f"\n{category}:")
        print("-" * 50)
        for field in fields:
            if field in info:
                value = info[field]
                # Format the field name nicely
                display_name = field.replace('_', ' ').title()
                print(f"{display_name:<30}: {value}")
    
    # Print any additional fields not in categories
    printed_fields = set()
    for fields in categories.values():
        printed_fields.update(fields)
    
    other_fields = set(info.keys()) - printed_fields
    if other_fields:
        print(f"\nOther Configuration:")
        print("-" * 50)
        for field in sorted(other_fields):
            display_name = field.replace('_', ' ').title()
            print(f"{display_name:<30}: {info[field]}")
    
    # Print metrics
    if metrics:
        print(f"\nFinal Metrics:")
        print("-" * 50)
        for metric, value in sorted(metrics.items()):
            display_name = metric.replace('_', ' ').replace('/', ' ').title()
            if isinstance(value, float):
                print(f"{display_name:<30}: {value:.6f}")
            else:
                print(f"{display_name:<30}: {value}")
    else:
        print(f"\nNo final metrics found in wandb logs")
    
    print("=" * 70)


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_wandb_info.py <wandb_run_directory>")
        print("Example: python extract_wandb_info.py ./wandb/run-20240104_123456-abcd1234")
        sys.exit(1)
    
    wandb_dir = Path(sys.argv[1])
    
    if not wandb_dir.exists():
        print(f"Error: Directory {wandb_dir} does not exist")
        sys.exit(1)
    
    # Load config file
    config_file = wandb_dir / 'files' / 'config.yaml'
    if not config_file.exists():
        # Try JSON format
        config_file = wandb_dir / 'files' / 'config.json'
    
    if config_file.exists():
        if config_file.suffix == '.yaml':
            import yaml
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading YAML config: {e}")
                config = {}
        else:
            config = load_json_file(config_file) or {}
    else:
        print("Warning: No config file found")
        config = {}
    
    # Extract information
    info = extract_config_info(config)
    metrics = extract_final_metrics(wandb_dir)
    
    # Pretty print
    pretty_print_info(info, metrics, str(wandb_dir))
    
    # Additional info about the run
    print("\nAdditional Files in WandB Directory:")
    print("-" * 50)
    files_dir = wandb_dir / 'files'
    if files_dir.exists():
        for file in sorted(files_dir.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"{file.name:<40}: {size_mb:>10.2f} MB")


if __name__ == "__main__":
    main()