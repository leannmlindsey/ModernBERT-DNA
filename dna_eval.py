# Copyright 2024 ModernBERT-DNA authors
# SPDX-License-Identifier: Apache-2.0

"""DNA benchmark evaluation script for NTv2, GUE, and Genomic Benchmarks."""

import copy
import gc
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing.managers import DictProxy, SyncManager
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Add folder root to path to allow us to use relative imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import omegaconf as om
import src.evals.dna_jobs as dna_jobs_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
import src.flex_bert as flex_bert_module
from src.dna_checkpoint_utils import load_pretrained_dna_model
from src.evals.dna_test_eval import evaluate_on_test_set, print_test_metrics
from src.evals.dna_data import NTV2_TASK_CONFIG
import torch
import transformers
from composer import algorithms
from composer.callbacks import (
    LRMonitor,
    MemoryMonitor,
    OptimizerMonitor,
    RuntimeEstimator,
    SpeedMonitor,
)
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
from src.scheduler import WarmupStableDecayScheduler
from composer.utils import reproducibility
from composer.utils.file_helpers import get_file
from omegaconf import DictConfig

# DNA task mappings
NTV2_TASKS = {
    "H2AFZ": dna_jobs_module.H2AFZJob,
    "H3K27ac": dna_jobs_module.H3K27acJob,
    "H3K27me3": dna_jobs_module.H3K27me3Job,
    "H3K36me3": dna_jobs_module.H3K36me3Job,
    "H3K4me1": dna_jobs_module.H3K4me1Job,
    "H3K4me2": dna_jobs_module.H3K4me2Job,
    "H3K4me3": dna_jobs_module.H3K4me3Job,
    "H3K9ac": dna_jobs_module.H3K9acJob,
    "H3K9me3": dna_jobs_module.H3K9me3Job,
    "H4K20me1": dna_jobs_module.H4K20me1Job,
    "enhancers": dna_jobs_module.EnhancersJob,
    "enhancers_types": dna_jobs_module.EnhancersTypesJob,
    "promoter_all": dna_jobs_module.PromoterAllJob,
    "promoter_no_tata": dna_jobs_module.PromoterNoTataJob,
    "promoter_tata": dna_jobs_module.PromoterTataJob,
    "splice_sites_acceptors": dna_jobs_module.SpliceSitesAcceptorsJob,
    "splice_sites_all": dna_jobs_module.SpliceSitesAllJob,
    "splice_sites_donors": dna_jobs_module.SpliceSitesDonorsJob,
}

TASK_NAME_TO_CLASS = NTV2_TASKS  # Can extend with GUE and GB tasks later


def build_algorithm(name, kwargs):
    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


def build_callback(name, kwargs):
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "memory_monitor":
        return MemoryMonitor()
    elif name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1),
            gpu_flops_available=kwargs.get("gpu_flops_available", None),
        )
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "optimizer_monitor":
        return OptimizerMonitor(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
        )
    else:
        raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name, kwargs):
    if name == "wandb":
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f"Not sure how to build logger: {name}")


def build_scheduler(name, kwargs):
    if name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=kwargs["t_warmup"])
    elif name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(
            t_warmup=kwargs["t_warmup"], alpha_f=kwargs["alpha_f"]
        )
    elif name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(
            t_warmup=kwargs["t_warmup"], alpha_f=kwargs["alpha_f"]
        )
    elif name == "warmup_stable_decay":
        return WarmupStableDecayScheduler(
            t_warmup=kwargs["t_warmup"], alpha_f=kwargs["alpha_f"]
        )
    else:
        raise ValueError(f"Not sure how to build scheduler: {name}")


def run_job_worker(
    cfg: dict,
    job_name: str,
    model_cfg: dict,
    metrics: Optional[DictProxy] = None,
):
    """Run a single DNA evaluation job."""
    
    if job_name not in TASK_NAME_TO_CLASS:
        raise ValueError(f"Job {job_name} not found in TASK_NAME_TO_CLASS")
    
    # Set up for specific job
    job_cfg = cfg["jobs"].get(job_name, {})
    
    # Build model
    model_name = model_cfg["name"]
    model_args = {
        "num_labels": model_cfg.get("num_labels", 2),
        "pretrained_model_name": model_cfg.get("pretrained_model_name", "bert-base-uncased"),
        "model_config": model_cfg.get("model_config", {}),
        "tokenizer_name": model_cfg.get("tokenizer_name"),
        "gradient_checkpointing": model_cfg.get("gradient_checkpointing", False),
    }
    
    checkpoint_path = model_cfg.get("pretrained_checkpoint")
    
    if model_name == "flex_bert":
        if checkpoint_path:
            model = load_pretrained_dna_model(flex_bert_module, model_args, checkpoint_path)
        else:
            model = flex_bert_module.create_flex_bert_classification(**model_args)
    elif model_name == "mosaic_bert":
        if checkpoint_path:
            model = load_pretrained_dna_model(mosaic_bert_module, model_args, checkpoint_path)
        else:
            model = mosaic_bert_module.create_mosaic_bert_classification(**model_args)
    elif model_name == "hf_bert":
        if checkpoint_path:
            model = load_pretrained_dna_model(hf_bert_module, model_args, checkpoint_path)
        else:
            model = hf_bert_module.create_hf_bert_classification(**model_args)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Update optimizer configuration for DNA
    if "optimizer" not in job_cfg or job_cfg.get("optimizer") is None:
        optimizer = DecoupledAdamW(
            model.parameters(),
            lr=job_cfg.get("lr", 2.0e-05),
            betas=job_cfg.get("betas", (0.9, 0.98)),
            eps=job_cfg.get("eps", 1.0e-06),
            weight_decay=job_cfg.get("weight_decay", 1.0e-05),
        )
    else:
        optimizer = None
    
    # Build callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get("callbacks", {}).items()
    ]
    
    # Build loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get("loggers", {}).items()
    ]
    
    # Build scheduler
    scheduler_cfg = job_cfg.get("scheduler", cfg.get("scheduler"))
    if scheduler_cfg:
        scheduler = build_scheduler(scheduler_cfg["name"], scheduler_cfg)
    else:
        scheduler = None
    
    # Build the job
    # Extract specific args to avoid duplicates
    job_args = {
        "model": model,
        "tokenizer_name": cfg["tokenizer_name"],
        "dataset_base_path": cfg["dataset_base_path"],
        "batch_size": job_cfg.get("batch_size", cfg["default_batch_size"]),
        "seed": job_cfg.get("seed", cfg["seed"]),
        "scheduler": scheduler,
        "optimizer": optimizer,
        "callbacks": callbacks,
        "loggers": loggers,
        "max_duration": job_cfg.get("max_duration", cfg.get("max_duration", "5ep")),
        "eval_interval": job_cfg.get("eval_interval", cfg.get("eval_interval", "1ep")),
        "save_folder": job_cfg.get("save_folder", None),
        "precision": cfg.get("precision", None),
        "max_sequence_length": job_cfg.get("max_sequence_length"),
    }
    
    job = TASK_NAME_TO_CLASS[job_name](**job_args)
    
    # Train the job
    trainer = job.get_trainer()
    if job_cfg.get("skip_training", cfg.get("skip_training", False)):
        # Just evaluate without training
        eval_metrics = trainer.eval()
    else:
        trainer.fit()
        eval_metrics = trainer.state.eval_metrics
    
    # Optionally evaluate on test set with comprehensive metrics
    if cfg.get("eval_on_test", False):
        # Create test dataloader
        from src.evals.dna_data import create_ntv2_dataset
        test_dataset = create_ntv2_dataset(
            task=job_name,
            tokenizer_name=cfg["tokenizer_name"],
            split="test",
            dataset_base_path=cfg["dataset_base_path"],
            max_seq_length=job_cfg.get("max_sequence_length", 512)
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=job_cfg.get("batch_size", cfg["default_batch_size"]),
            collate_fn=transformers.default_data_collator,
            num_workers=4,
            pin_memory=True,
        )
        
        # Get test metrics
        test_metrics = evaluate_on_test_set(
            model=trainer.state.model,
            test_dataloader=test_dataloader,
            task_name=job_name,
            num_labels=NTV2_TASK_CONFIG[job_name]["num_labels"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Print test metrics
        print_test_metrics(test_metrics, job_name)
        
        # Add test metrics to eval_metrics
        eval_metrics["test"] = test_metrics
    
    # Update metrics
    if metrics is not None:
        metrics[job_name] = eval_metrics
    
    # Clean up
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return eval_metrics


def main(cfg: dict):
    """Main evaluation function."""
    
    # Set seed
    if "seed" in cfg:
        reproducibility.seed_all(cfg["seed"])
    
    # Get list of jobs to run
    if "eval_tasks" in cfg and cfg["eval_tasks"]:
        jobs_to_run = cfg["eval_tasks"]
    else:
        jobs_to_run = list(cfg["jobs"].keys())
    
    print(f"Running DNA evaluation jobs: {jobs_to_run}")
    
    # Build model configuration
    model_cfg = cfg["model"]
    
    # Get number of labels for each task
    from src.evals.dna_data import NTV2_TASK_CONFIG
    
    # Run jobs
    metrics = {}
    for job_name in jobs_to_run:
        if job_name not in TASK_NAME_TO_CLASS:
            print(f"Warning: Skipping unknown job {job_name}")
            continue
        
        print(f"\nRunning job: {job_name}")
        
        # Update model config with task-specific number of labels
        task_model_cfg = copy.deepcopy(model_cfg)
        if job_name in NTV2_TASK_CONFIG:
            task_model_cfg["num_labels"] = NTV2_TASK_CONFIG[job_name]["num_labels"]
        
        # Run the job
        job_metrics = run_job_worker(cfg, job_name, task_model_cfg)
        metrics[job_name] = job_metrics
        
        # Print results
        print(f"Results for {job_name}:")
        if isinstance(job_metrics, dict):
            for metric_name, metric_value in job_metrics.items():
                if isinstance(metric_value, dict):
                    # Handle nested metric dictionaries
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        if isinstance(sub_metric_value, torch.Tensor):
                            sub_metric_value = sub_metric_value.item()
                        if isinstance(sub_metric_value, (int, float)):
                            print(f"  {metric_name}/{sub_metric_name}: {sub_metric_value:.4f}")
                else:
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    if isinstance(metric_value, (int, float)):
                        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("DNA EVALUATION SUMMARY")
    print("="*60)
    for job_name, job_metrics in metrics.items():
        accuracy = None
        if isinstance(job_metrics, dict):
            for metric_name, metric_value in job_metrics.items():
                if isinstance(metric_value, dict):
                    # Check nested metrics
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        if "accuracy" in sub_metric_name.lower():
                            if isinstance(sub_metric_value, torch.Tensor):
                                accuracy = sub_metric_value.item()
                            elif isinstance(sub_metric_value, (int, float)):
                                accuracy = sub_metric_value
                            break
                elif "accuracy" in metric_name.lower():
                    if isinstance(metric_value, torch.Tensor):
                        accuracy = metric_value.item()
                    elif isinstance(metric_value, (int, float)):
                        accuracy = metric_value
                    break
        if accuracy is not None:
            print(f"{job_name}: {accuracy:.4f}")
    
    return metrics


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.OmegaConf.load(f)
    
    # Load defaults if they exist
    if os.path.exists("yamls/defaults.yaml"):
        with open("yamls/defaults.yaml") as f:
            default_cfg = om.OmegaConf.load(f)
        cfg = om.OmegaConf.merge(default_cfg, yaml_cfg)
    else:
        cfg = yaml_cfg
    
    # Merge with command line arguments
    cli_cfg = om.OmegaConf.from_cli(args_list)
    cfg = om.OmegaConf.merge(cfg, cli_cfg)
    cfg = om.OmegaConf.to_container(cfg, resolve=True)
    
    main(cfg)