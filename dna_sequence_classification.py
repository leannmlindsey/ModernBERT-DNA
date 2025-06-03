# Copyright 2024 ModernBERT-DNA authors
# SPDX-License-Identifier: Apache-2.0

"""Fine-tuning script for DNA sequence classification tasks (Nucleotide Transformer v2 tasks)."""

import os
import sys
from typing import Optional, cast
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

# Add folder root to path to allow us to use relative imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
from torch.utils.data import Dataset as TorchDataset
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
import src.flex_bert as flex_bert_module
import transformers
from composer import Trainer, algorithms, Evaluator
from composer.callbacks import LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor
from composer.core.types import Dataset as ComposerDataset
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
# Metrics will be passed as strings to Evaluator
from src.scheduler import WarmupStableDecayScheduler
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dna_tokenizer import DNACharacterTokenizer


class DNASequenceDataset(TorchDataset):
    """Dataset for DNA sequence classification tasks."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        task_name: str = None,
        split: str = "train",
        data_format: str = "ntv2",  # ntv2, gb, or gue
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
        self.split = split
        self.data_format = data_format
        
        # Load data - try to auto-detect format
        # First attempt to load as CSV (works for standardized format across all benchmarks)
        self.data = pd.read_csv(data_path)
        
        # Check if it's the standardized format with "sequence" and "label" columns
        if "sequence" in self.data.columns and "label" in self.data.columns:
            # Standardized format for all benchmarks
            self.texts = self.data["sequence"].tolist()
            self.labels = self.data["label"].tolist()
            self.is_pair = False
        elif "sequence1" in self.data.columns and "sequence2" in self.data.columns and "label" in self.data.columns:
            # Sequence pair task with named columns
            self.texts1 = self.data["sequence1"].tolist()
            self.texts2 = self.data["sequence2"].tolist()
            self.labels = self.data["label"].tolist()
            self.is_pair = True
        else:
            # Fall back to original format detection based on data_format parameter
            if data_format == "gb":
                # Genomic Benchmarks format: TSV files with sequence and label columns
                self.data = pd.read_csv(data_path, sep="\t", header=None, names=["sequence", "label"])
                self.texts = self.data["sequence"].tolist()
                self.labels = self.data["label"].tolist()
                self.is_pair = False
            elif data_format == "gue":
                # GUE format: Can be JSON or TSV depending on task
                if data_path.endswith(".json"):
                    import json
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    self.texts = [item["sequence"] for item in data]
                    self.labels = [item["label"] for item in data]
                    self.is_pair = False
                else:
                    # TSV format for GUE
                    self.data = pd.read_csv(data_path, sep="\t")
                    if "sequence1" in self.data.columns and "sequence2" in self.data.columns:
                        # Sequence pair task
                        self.texts1 = self.data["sequence1"].tolist()
                        self.texts2 = self.data["sequence2"].tolist()
                        self.labels = self.data["label"].tolist()
                        self.is_pair = True
                    else:
                        # Single sequence task
                        self.texts = self.data["sequence"].tolist()
                        self.labels = self.data["label"].tolist()
                        self.is_pair = False
            else:
                # Default NT v2 format: CSV with positional columns
                # Determine if this is a sequence pair task
                self.is_pair = len(self.data.columns) == 3
                
                if self.is_pair:
                    # Sequence pair classification (e.g., enhancer types)
                    self.texts1 = self.data.iloc[:, 0].tolist()
                    self.texts2 = self.data.iloc[:, 1].tolist()
                    self.labels = self.data.iloc[:, 2].tolist()
                else:
                    # Single sequence classification
                    self.texts = self.data.iloc[:, 0].tolist()
                    self.labels = self.data.iloc[:, 1].tolist()
        
        # Convert labels to integers if they're not already
        if isinstance(self.labels[0], str):
            self.label_map = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
            self.labels = [self.label_map[label] for label in self.labels]
        else:
            # Labels are already integers
            unique_labels = sorted(set(self.labels))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.labels = [self.label_map[label] for label in self.labels]
        
        self.num_labels = len(self.label_map)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.is_pair:
            text1 = str(self.texts1[idx])
            text2 = str(self.texts2[idx])
            
            # Tokenize sequence pair
            encoding = self.tokenizer(
                text1,
                text2,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            text = str(self.texts[idx])
            
            # Tokenize single sequence
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def get_task_max_length(task_name: str) -> int:
    """Get the appropriate max sequence length for each task."""
    task_lengths = {
        # Histone modifications
        "H2AFZ": 600,
        "H3K27ac": 600,
        "H3K27me3": 600,
        "H3K36me3": 600,
        "H3K4me1": 600,
        "H3K4me2": 600,
        "H3K4me3": 600,
        "H3K9ac": 600,
        "H3K9me3": 600,
        "H4K20me1": 600,
        # Enhancers
        "enhancers": 190,
        "enhancers_types": 190,
        # Promoters
        "promoter_all": 300,
        "promoter_no_tata": 300,
        "promoter_tata": 300,
        # Splice sites
        "splice_sites_acceptors": 600,
        "splice_sites_all": 200,
        "splice_sites_donors": 600,
    }
    return task_lengths.get(task_name, 512)


def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f"WARNING: device_train_microbatch_size > device_train_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_train_batch_size}."
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size

    # Safely set `device_eval_microbatch_size` if not provided by user
    if "device_eval_microbatch_size" not in cfg:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_microbatch_size = 1
        else:
            cfg.device_eval_microbatch_size = cfg.device_train_microbatch_size

    global_eval_batch_size, device_eval_microbatch_size = (
        cfg.get("global_eval_batch_size", global_batch_size),
        cfg.device_eval_microbatch_size,
    )
    device_eval_batch_size = global_eval_batch_size // dist.get_world_size()
    if isinstance(device_eval_microbatch_size, int):
        if device_eval_microbatch_size > device_eval_batch_size:
            print(
                f"WARNING: device_eval_microbatch_size > device_eval_batch_size, "
                f"will be reduced from {device_eval_microbatch_size} -> {device_eval_batch_size}."
            )
            device_eval_microbatch_size = device_eval_batch_size
    cfg.device_eval_batch_size = device_eval_batch_size
    cfg.device_eval_microbatch_size = device_eval_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if "wandb" in cfg.get("loggers", {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


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
            window_size=kwargs.get("window_size", 1), gpu_flops_available=kwargs.get("gpu_flops_available", None)
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


def build_scheduler(cfg):
    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "warmup_stable_decay":
        return WarmupStableDecayScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg, model):
    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(
            model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


def build_tokenizer(cfg: DictConfig):
    """Build tokenizer based on configuration."""
    if cfg.tokenizer_name == "dna_char":
        return DNACharacterTokenizer()
    else:
        return AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)


# GUE task directory mapping
GUE_TASK_MAPPING = {
    # EMP tasks (Epigenetic Marks Prediction)
    "human_h3": "EMP/H3",
    "human_h3k14ac": "EMP/H3K14ac",
    "human_h3k36me3": "EMP/H3K36me3",
    "human_h3k4me1": "EMP/H3K4me1",
    "human_h3k4me2": "EMP/H3K4me2",
    "human_h3k4me3": "EMP/H3K4me3",
    "human_h3k79me3": "EMP/H3K79me3",
    "human_h3k9ac": "EMP/H3K9ac",
    "human_h4": "EMP/H4",
    "human_h4ac": "EMP/H4ac",
    # Mouse tasks
    "mouse_0": "mouse/0",
    "mouse_1": "mouse/1",
    "mouse_2": "mouse/2",
    "mouse_3": "mouse/3",
    "mouse_4": "mouse/4",
    # Promoter tasks
    "prom_300_all": "prom/prom_300_all",
    "prom_300_notata": "prom/prom_300_notata",
    "prom_300_tata": "prom/prom_300_tata",
    "prom_core_all": "prom/prom_core_all",
    "prom_core_notata": "prom/prom_core_notata",
    "prom_core_tata": "prom/prom_core_tata",
    # Splice site tasks
    "splice_reconstructed": "splice/reconstructed",
    # TF binding tasks
    "tf_0": "tf/0",
    "tf_1": "tf/1",
    "tf_2": "tf/2",
    "tf_3": "tf/3",
    "tf_4": "tf/4",
    # Virus tasks
    "covid": "virus/covid",
    "virus_covid": "virus/covid",
}


def build_dna_dataloader(cfg: DictConfig, device_batch_size: int, tokenizer):
    """Create a dataloader for DNA sequence classification.
    
    Args:
        cfg (DictConfig): Configuration for dataset/dataloader creation.
        device_batch_size (int): The size of the batches.
        tokenizer: The tokenizer to use for preprocessing.
        
    Returns:
        dataloader: A dataloader for the Composer Trainer.
    """
    # Get task-specific max length
    max_length = cfg.get("max_seq_len", get_task_max_length(cfg.task_name))
    
    # Get data format
    data_format = cfg.get("data_format", "ntv2")
    
    # Build data path
    # Get the base path from config, with no default fallback to ensure it's always specified
    data_dir = cfg.get("data_dir")
    if data_dir is None:
        raise ValueError(
            "data_dir must be specified in the configuration. "
            "Please ensure your config includes data_dir or inherits from a base config that does."
        )
    split_name = "dev" if cfg.split == "validation" else cfg.split
    
    # Handle GUE task subdirectory structure
    if data_format == "gue" and cfg.task_name in GUE_TASK_MAPPING:
        task_subdir = GUE_TASK_MAPPING[cfg.task_name]
        csv_path = os.path.join(data_dir, task_subdir, f"{split_name}.csv")
    else:
        # For NT v2 and GB, tasks are at root level
        csv_path = os.path.join(data_dir, cfg.task_name, f"{split_name}.csv")
    
    if os.path.exists(csv_path):
        # Use the standardized CSV format
        data_path = csv_path
    else:
        # Fallback for non-standard formats (kept for compatibility)
        if data_format == "gb":
            data_path = os.path.join(data_dir, cfg.task_name, f"{split_name}.tsv")
        elif data_format == "gue":
            # Try JSON first, then TSV
            if cfg.task_name in GUE_TASK_MAPPING:
                task_subdir = GUE_TASK_MAPPING[cfg.task_name]
                json_path = os.path.join(data_dir, task_subdir, f"{split_name}.json")
                tsv_path = os.path.join(data_dir, task_subdir, f"{split_name}.tsv")
            else:
                json_path = os.path.join(data_dir, cfg.task_name, f"{split_name}.json")
                tsv_path = os.path.join(data_dir, cfg.task_name, f"{split_name}.tsv")
            
            if os.path.exists(json_path):
                data_path = json_path
            else:
                data_path = tsv_path
        else:
            # Default to CSV path
            data_path = csv_path
    
    # Create dataset
    dataset = DNASequenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        task_name=cfg.task_name,
        split=cfg.split,
        data_format=data_format,
    )
    
    # Store num_labels for model creation
    if cfg.split == "train":
        cfg.num_labels = dataset.num_labels
    
    dataset = cast(ComposerDataset, dataset)
    dataloader = DataLoader(
        dataset,
        collate_fn=transformers.default_data_collator,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset, drop_last=cfg.drop_last, shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )
    
    return dataloader


def build_model(cfg: DictConfig):
    if cfg.name == "hf_bert":
        return hf_bert_module.create_hf_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get("use_pretrained", False),
            model_config=cfg.get("model_config"),
            tokenizer_name=cfg.get("tokenizer_name"),
            gradient_checkpointing=cfg.get("gradient_checkpointing"),
        )
    elif cfg.name == "mosaic_bert":
        return mosaic_bert_module.create_mosaic_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
            model_config=cfg.get("model_config"),
            tokenizer_name=cfg.get("tokenizer_name"),
            gradient_checkpointing=cfg.get("gradient_checkpointing"),
        )
    elif cfg.name == "flex_bert":
        return flex_bert_module.create_flex_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
            model_config=cfg.get("model_config"),
            tokenizer_name=cfg.get("tokenizer_name"),
            gradient_checkpointing=cfg.get("gradient_checkpointing"),
        )
    else:
        raise ValueError(f"Not sure how to build model with name={cfg.name}")


def train(cfg: DictConfig, return_trainer: bool = False, do_train: bool = True) -> Optional[Trainer]:
    print(f"Training DNA sequence classification on task: {cfg.task_name}")
    print("Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = build_tokenizer(cfg)

    # Dataloaders - build train first to get num_labels
    print("Building train loader...")
    # Merge train_loader config with top-level config for access to task_name and other fields
    train_loader_cfg = om.merge(cfg, cfg.train_loader) if "train_loader" in cfg else cfg
    train_loader = build_dna_dataloader(
        train_loader_cfg,
        cfg.global_train_batch_size // dist.get_world_size(),
        tokenizer,
    )
    
    # Build Model (after dataloader so we have num_labels)
    print("Initializing model...")
    # Get num_labels from the merged config (which includes task-specific settings)
    cfg.model.num_labels = train_loader_cfg.get("num_labels", 2)  # Default to binary classification
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params=:.4e}")
    
    print("Building eval loader...")
    global_eval_batch_size = cfg.get("global_eval_batch_size", cfg.global_train_batch_size)
    # Merge eval_loader config with top-level config for access to task_name and other fields
    eval_loader_cfg = om.merge(cfg, cfg.eval_loader) if "eval_loader" in cfg else cfg
    eval_loader = build_dna_dataloader(
        eval_loader_cfg,
        cfg.get("device_eval_batch_size", global_eval_batch_size // dist.get_world_size()),
        tokenizer,
    )
    eval_evaluator = Evaluator(
        label="eval",
        dataloader=eval_loader,
        device_eval_microbatch_size=cfg.get("device_eval_microbatch_size", None),
        metric_names=["MulticlassAccuracy"]
    )

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get("callbacks", {}).items()]

    # Algorithms
    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in cfg.get("algorithms", {}).items()]

    if cfg.get("run_name") is None:
        cfg.run_name = f"dna-{cfg.task_name}-{cfg.model.pretrained_model_name.split('/')[-1]}"

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_evaluator,
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get("device"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        save_folder=cfg.get("save_folder"),
        save_interval=cfg.get("save_interval", "1000ba"),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        save_overwrite=cfg.get("save_overwrite", False),
        load_path=cfg.get("load_path"),
        load_weights_only=cfg.get("load_weights_only", True),
    )

    print("Logging config...")
    log_config(cfg)

    if do_train:
        print("Starting training...")
        trainer.fit()

    if return_trainer:
        return trainer


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open("yamls/defaults.yaml") as f:
        default_cfg = om.load(f)
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(default_cfg, yaml_cfg, cli_cfg)
    
    # Handle task-specific configuration if present
    if "tasks" in cfg and cfg.get("task_name") and cfg.task_name in cfg.tasks:
        task_cfg = cfg.tasks[cfg.task_name]
        # Merge task-specific config with main config
        # Priority: CLI args > task config > yaml config > defaults
        cfg = om.merge(cfg, task_cfg, cli_cfg)
    
    cfg = cast(DictConfig, cfg)  # for type checking
    train(cfg)