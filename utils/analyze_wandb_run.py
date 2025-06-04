#!/usr/bin/env python
"""Analyze WandB run data including metrics over time."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


def load_metrics_history(wandb_dir: Path) -> pd.DataFrame:
    """Load metrics history from wandb-history.jsonl file."""
    history_file = wandb_dir / 'files' / 'wandb-history.jsonl'
    
    if not history_file.exists():
        return pd.DataFrame()
    
    metrics_data = []
    try:
        with open(history_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    metrics_data.append(data)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading history file: {e}")
        return pd.DataFrame()
    
    if not metrics_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics_data)
    return df


def analyze_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze metrics DataFrame to extract key statistics."""
    analysis = {}
    
    if df.empty:
        return analysis
    
    # Identify metric columns
    metric_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['loss', 'perplexity', 'accuracy', 'f1', 'lr']):
            metric_cols.append(col)
    
    # For each metric, get statistics
    for col in metric_cols:
        if col in df and df[col].notna().any():
            values = df[col].dropna()
            if len(values) > 0:
                analysis[col] = {
                    'initial': values.iloc[0],
                    'final': values.iloc[-1],
                    'min': values.min(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'num_values': len(values)
                }
    
    # Get training progress info
    if '_step' in df.columns:
        analysis['total_steps'] = df['_step'].max()
    if 'epoch' in df.columns:
        analysis['total_epochs'] = df['epoch'].max()
    
    return analysis


def extract_run_info(wandb_dir: Path) -> Dict[str, Any]:
    """Extract comprehensive run information from WandB directory."""
    info = {
        'config': {},
        'summary': {},
        'metrics_analysis': {},
        'system_info': {}
    }
    
    # Load config
    config_file = wandb_dir / 'files' / 'config.yaml'
    if not config_file.exists():
        config_file = wandb_dir / 'files' / 'config.json'
    
    if config_file.exists():
        try:
            if config_file.suffix == '.yaml':
                import yaml
                with open(config_file, 'r') as f:
                    raw_config = yaml.safe_load(f)
            else:
                with open(config_file, 'r') as f:
                    raw_config = json.load(f)
            
            # Extract key fields
            key_fields = [
                'run_name', 'tokenizer_name', 'mlm_probability', 'data_local',
                'device_train_batch_size', 'device_train_microbatch_size',
                'device_eval_batch_size', 'n_gpus', 'max_duration', 'max_seq_len',
                'vocab_size', 'seed', 'precision', 'num_workers'
            ]
            
            for field in key_fields:
                if field in raw_config:
                    value = raw_config[field]
                    info['config'][field] = value['value'] if isinstance(value, dict) and 'value' in value else value
            
            # Extract nested model config
            if 'model' in raw_config:
                model = raw_config['model']
                if isinstance(model, dict):
                    if 'value' in model:
                        model = model['value']
                    for key in ['name', 'model_config']:
                        if key in model:
                            info['config'][f'model_{key}'] = model[key]
            
            # Extract optimizer config
            if 'optimizer' in raw_config:
                opt = raw_config['optimizer']
                if isinstance(opt, dict):
                    if 'value' in opt:
                        opt = opt['value']
                    if 'lr' in opt:
                        info['config']['learning_rate'] = opt['lr']
                    if 'weight_decay' in opt:
                        info['config']['weight_decay'] = opt['weight_decay']
            
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Load summary
    summary_file = wandb_dir / 'files' / 'wandb-summary.json'
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                info['summary'] = json.load(f)
        except Exception as e:
            print(f"Error loading summary: {e}")
    
    # Load and analyze metrics history
    df = load_metrics_history(wandb_dir)
    if not df.empty:
        info['metrics_analysis'] = analyze_metrics(df)
    
    # Load system info
    metadata_file = wandb_dir / 'files' / 'wandb-metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if 'gpu' in metadata:
                    info['system_info']['gpu'] = metadata['gpu']
                if 'cpu' in metadata:
                    info['system_info']['cpu_count'] = metadata['cpu']['count']
                if 'cuda' in metadata:
                    info['system_info']['cuda_version'] = metadata['cuda']
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    return info


def pretty_print_analysis(info: Dict[str, Any], wandb_dir: str):
    """Pretty print the analysis results."""
    print("=" * 80)
    print(f"WANDB RUN ANALYSIS")
    print(f"Directory: {wandb_dir}")
    print("=" * 80)
    
    # Configuration
    config = info['config']
    if config:
        print("\nRUN CONFIGURATION")
        print("-" * 80)
        
        sections = {
            'Basic Info': ['run_name', 'seed', 'precision'],
            'Model': ['model_name', 'tokenizer_name', 'vocab_size', 'mlm_probability'],
            'Data': ['data_local', 'max_seq_len', 'num_workers'],
            'Training': ['device_train_batch_size', 'device_train_microbatch_size', 
                        'device_eval_batch_size', 'n_gpus', 'max_duration'],
            'Optimizer': ['learning_rate', 'weight_decay']
        }
        
        for section, fields in sections.items():
            print(f"\n{section}:")
            for field in fields:
                if field in config:
                    display_name = field.replace('_', ' ').title()
                    print(f"  {display_name:<35}: {config[field]}")
    
    # Metrics Analysis
    analysis = info['metrics_analysis']
    if analysis:
        print("\n\nMETRICS ANALYSIS")
        print("-" * 80)
        
        # Extract metric analyses (excluding metadata like total_steps)
        metric_analyses = {k: v for k, v in analysis.items() 
                          if isinstance(v, dict) and 'final' in v}
        
        if metric_analyses:
            # Create a table view
            print(f"\n{'Metric':<25} {'Initial':>12} {'Final':>12} {'Min':>12} {'Max':>12} {'Mean':>12}")
            print("-" * 85)
            
            for metric, stats in sorted(metric_analyses.items()):
                print(f"{metric:<25} "
                      f"{stats['initial']:>12.6f} "
                      f"{stats['final']:>12.6f} "
                      f"{stats['min']:>12.6f} "
                      f"{stats['max']:>12.6f} "
                      f"{stats['mean']:>12.6f}")
        
        # Training progress
        if 'total_steps' in analysis:
            print(f"\nTotal Training Steps: {analysis['total_steps']}")
        if 'total_epochs' in analysis:
            print(f"Total Epochs: {analysis['total_epochs']}")
    
    # Final Summary Metrics
    summary = info['summary']
    if summary:
        print("\n\nFINAL SUMMARY METRICS")
        print("-" * 80)
        
        # Filter for relevant metrics
        relevant_metrics = {}
        for key, value in summary.items():
            if any(keyword in key.lower() for keyword in ['loss', 'perplexity', 'accuracy', 'f1', 'lr', '_runtime']):
                if isinstance(value, (int, float)):
                    relevant_metrics[key] = value
        
        if relevant_metrics:
            for key, value in sorted(relevant_metrics.items()):
                display_key = key.replace('_', ' ').replace('/', ' > ').title()
                if isinstance(value, float):
                    print(f"{display_key:<40}: {value:.6f}")
                else:
                    print(f"{display_key:<40}: {value}")
    
    # System Info
    if info['system_info']:
        print("\n\nSYSTEM INFORMATION")
        print("-" * 80)
        for key, value in info['system_info'].items():
            print(f"{key.replace('_', ' ').title():<40}: {value}")
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_wandb_run.py <wandb_run_directory>")
        print("Example: python analyze_wandb_run.py ./wandb/run-20240104_123456-abcd1234")
        sys.exit(1)
    
    wandb_dir = Path(sys.argv[1])
    
    if not wandb_dir.exists():
        print(f"Error: Directory {wandb_dir} does not exist")
        sys.exit(1)
    
    # Extract and analyze
    info = extract_run_info(wandb_dir)
    
    # Pretty print
    pretty_print_analysis(info, str(wandb_dir))
    
    # Check if metrics plot is needed
    print("\nTo plot metrics over time, use: python plot_wandb_metrics.py " + str(wandb_dir))


if __name__ == "__main__":
    main()