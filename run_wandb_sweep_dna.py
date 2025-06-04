#!/usr/bin/env python3
"""
WandB Sweep runner for DNA model finetuning.
This script is called by WandB sweep agent with hyperparameters.
"""

import os
import sys
import wandb
import argparse
import subprocess
from pathlib import Path

def main():
    """Run a single sweep trial with hyperparameters from WandB."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True,
                        help="Base YAML config file (e.g., yamls/dna_eval_ntv2.yaml)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., enhancers, H3K4me3)")
    parser.add_argument("--model_type", type=str, default="bpe",
                        choices=["bpe", "char"], help="Model type")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--tokenizer_name", type=str, required=True,
                        help="Tokenizer name")
    parser.add_argument("--output_parent_dir", type=str, default="./outputs",
                        help="Parent directory for outputs")
    parser.add_argument("--vocab_size", type=int, default=4096,
                        help="Vocabulary size")
    
    args = parser.parse_args()
    
    # Initialize wandb run
    wandb.init()
    
    # Get hyperparameters from wandb config
    config = wandb.config
    
    # Build output directory with sweep ID
    output_dir = os.path.join(
        args.output_parent_dir,
        f"sweep_{wandb.run.sweep_id}",
        f"run_{wandb.run.id}",
        f"{args.task}_{args.model_type}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command with hyperparameters from sweep
    cmd = [
        "python", "dna_eval.py",
        args.config_file,
        f"tokenizer_name={args.tokenizer_name}",
        f"model.pretrained_checkpoint={args.pretrained_checkpoint}",
        f"vocab_size={args.vocab_size}",
        f"output_dir={output_dir}",
        f"eval_tasks=[{args.task}]",
        # Hyperparameters from sweep
        f"jobs.{args.task}.lr={config.lr}",
        f"jobs.{args.task}.weight_decay={config.weight_decay}",
        f"jobs.{args.task}.batch_size={config.batch_size}",
        f"jobs.{args.task}.max_duration={config.num_epochs}ep",
        f"jobs.{args.task}.betas=[{config.beta1},{config.beta2}]",
        # Scheduler configuration
        f"jobs.{args.task}.scheduler.name={config.scheduler_type}",
        f"jobs.{args.task}.scheduler.t_warmup={config.warmup_ratio}dur",
    ]
    
    # Add scheduler-specific parameters
    if config.scheduler_type in ["cosine_with_warmup", "linear_decay_with_warmup"]:
        cmd.append(f"jobs.{args.task}.scheduler.alpha_f={config.scheduler_alpha_f}")
    
    # Add gradient clipping if specified
    if config.gradient_clip_val > 0:
        cmd.append(f"algorithms.gradient_clipping.clipping_threshold={config.gradient_clip_val}")
        cmd.append("algorithms.gradient_clipping.clipping_type=norm")
    
    # Add WandB configuration
    cmd.extend([
        "loggers.wandb.project=modernBERT-DNA-sweeps",
        "loggers.wandb.entity=leannmlindsey",
        f"loggers.wandb.tags=[{args.task},{args.model_type},sweep]",
        f"loggers.wandb.run_id={wandb.run.id}",
        f"loggers.wandb.log_dir={output_dir}/wandb",
    ])
    
    # Print command for debugging
    print("Running command:")
    print(" ".join(cmd))
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        wandb.log({"error": str(e)})
        raise
    
    # Parse metrics from output or CSV file
    # WandB will automatically sync metrics logged during training
    
    wandb.finish()


if __name__ == "__main__":
    main()