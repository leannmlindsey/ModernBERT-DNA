# Copyright 2024 ModernBERT-DNA authors
# SPDX-License-Identifier: Apache-2.0

"""DNA benchmark job classes for NTv2 evaluation."""

import os
import sys
from typing import List, Optional
from multiprocessing import cpu_count
import torch

# Add folder root to path to allow us to use relative imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer import ComposerModel
from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler, DecoupledAdamW
from torch.optim import Optimizer
from src.evals.dna_data import create_ntv2_dataset, NTV2_TASK_CONFIG
from src.evals.finetuning_jobs import build_dataloader, ClassificationJob


class NTv2Job(ClassificationJob):
    """Base class for NTv2 tasks."""
    
    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        task_name: str,
        dataset_base_path: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "1ep",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = None,
        max_duration: Optional[str] = "5ep",
        batch_size: Optional[int] = 64,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        # Get task configuration
        if task_name not in NTV2_TASK_CONFIG:
            raise ValueError(f"Unknown NTv2 task: {task_name}")
        
        task_config = NTV2_TASK_CONFIG[task_name]
        self.num_labels = task_config["num_labels"]
        
        # Use task-specific max sequence length if not provided
        if max_sequence_length is None:
            max_sequence_length = task_config["max_seq_len"]
        
        # Initialize parent class
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name or f"ntv2_{task_name}",
            seed=seed,
            task_name=task_name,
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )
        
        self.dataset_base_path = dataset_base_path
        
        # Set up optimizer if not provided
        if optimizer is None:
            self.optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr=2.0e-05,
                betas=(0.9, 0.98),
                eps=1.0e-06,
                weight_decay=1.0e-05,
            )
        
        # Create dataset kwargs
        dataset_kwargs = {
            "task": self.task_name,
            "tokenizer_name": self.tokenizer_name,
            "max_seq_length": self.max_sequence_length,
            "dataset_base_path": self.dataset_base_path,
        }
        
        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }
        
        # Create train dataloader
        train_dataset = create_ntv2_dataset(split="train", **dataset_kwargs)
        self.train_dataloader = build_dataloader(train_dataset, **dataloader_kwargs)
        
        # Create eval dataloader
        eval_dataset = create_ntv2_dataset(split="validation", **dataset_kwargs)
        self.evaluators = [
            Evaluator(
                label=f"ntv2_{task_name}",
                dataloader=build_dataloader(eval_dataset, **dataloader_kwargs),
                metric_names=["MulticlassAccuracy"],
            )
        ]


# Create specific job classes for each NTv2 task
def create_ntv2_job_class(task_name: str):
    """Factory function to create task-specific job classes."""
    
    class SpecificNTv2Job(NTv2Job):
        def __init__(self, **kwargs):
            kwargs["task_name"] = task_name
            super().__init__(**kwargs)
    
    SpecificNTv2Job.__name__ = f"{task_name}Job"
    return SpecificNTv2Job


# Create job classes for all NTv2 tasks
H2AFZJob = create_ntv2_job_class("H2AFZ")
H3K27acJob = create_ntv2_job_class("H3K27ac")
H3K27me3Job = create_ntv2_job_class("H3K27me3")
H3K36me3Job = create_ntv2_job_class("H3K36me3")
H3K4me1Job = create_ntv2_job_class("H3K4me1")
H3K4me2Job = create_ntv2_job_class("H3K4me2")
H3K4me3Job = create_ntv2_job_class("H3K4me3")
H3K9acJob = create_ntv2_job_class("H3K9ac")
H3K9me3Job = create_ntv2_job_class("H3K9me3")
H4K20me1Job = create_ntv2_job_class("H4K20me1")
EnhancersJob = create_ntv2_job_class("enhancers")
EnhancersTypesJob = create_ntv2_job_class("enhancers_types")
PromoterAllJob = create_ntv2_job_class("promoter_all")
PromoterNoTataJob = create_ntv2_job_class("promoter_no_tata")
PromoterTataJob = create_ntv2_job_class("promoter_tata")
SpliceSitesAcceptorsJob = create_ntv2_job_class("splice_sites_acceptors")
SpliceSitesAllJob = create_ntv2_job_class("splice_sites_all")
SpliceSitesDonorsJob = create_ntv2_job_class("splice_sites_donors")