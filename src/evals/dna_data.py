# Copyright 2024 ModernBERT-DNA authors
# SPDX-License-Identifier: Apache-2.0

"""DNA dataset loading functions for NTv2, GUE, and Genomic Benchmarks."""

import os
import pandas as pd
import logging
from typing import Optional, Dict, List, Any
from composer.utils import MissingConditionalImportError, dist

log = logging.getLogger(__name__)

# NTv2 task configurations with max sequence lengths
NTV2_TASK_CONFIG = {
    # Histone modification tasks (600bp sequences)
    "H2AFZ": {"max_seq_len": 600, "num_labels": 2},
    "H3K27ac": {"max_seq_len": 600, "num_labels": 2},
    "H3K27me3": {"max_seq_len": 600, "num_labels": 2},
    "H3K36me3": {"max_seq_len": 600, "num_labels": 2},
    "H3K4me1": {"max_seq_len": 600, "num_labels": 2},
    "H3K4me2": {"max_seq_len": 600, "num_labels": 2},
    "H3K4me3": {"max_seq_len": 600, "num_labels": 2},
    "H3K9ac": {"max_seq_len": 600, "num_labels": 2},
    "H3K9me3": {"max_seq_len": 600, "num_labels": 2},
    "H4K20me1": {"max_seq_len": 600, "num_labels": 2},
    # Enhancer tasks (200bp sequences)
    "enhancers": {"max_seq_len": 200, "num_labels": 2},
    "enhancers_types": {"max_seq_len": 200, "num_labels": 3},
    # Promoter tasks (300bp sequences)
    "promoter_all": {"max_seq_len": 300, "num_labels": 2},
    "promoter_no_tata": {"max_seq_len": 300, "num_labels": 2},
    "promoter_tata": {"max_seq_len": 300, "num_labels": 2},
    # Splice site tasks (400bp sequences)
    "splice_sites_acceptors": {"max_seq_len": 400, "num_labels": 2},
    "splice_sites_all": {"max_seq_len": 400, "num_labels": 3},
    "splice_sites_donors": {"max_seq_len": 400, "num_labels": 2},
}


class DNADataset:
    """Simple DNA dataset class that mimics HuggingFace datasets format."""
    
    def __init__(self, sequences: List[str], labels: List[int]):
        self.sequences = sequences
        self.labels = labels
        assert len(sequences) == len(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "text": self.sequences[idx],
            "label": self.labels[idx]
        }
    
    def map(self, function, batched=True, num_proc=None, batch_size=1000, 
            remove_columns=None, load_from_cache_file=True):
        """Mimics HuggingFace dataset map function."""
        if batched:
            # Process in batches
            results = []
            for i in range(0, len(self), batch_size):
                batch = [self[j] for j in range(i, min(i + batch_size, len(self)))]
                batch_dict = {
                    "text": [item["text"] for item in batch],
                    "label": [item["label"] for item in batch]
                }
                batch_results = function(batch_dict)
                results.extend([
                    {k: batch_results[k][j] for k in batch_results}
                    for j in range(len(batch))
                ])
        else:
            # Process one by one
            results = [function(self[i]) for i in range(len(self))]
        
        # Create new dataset from results
        if results and "input_ids" in results[0]:
            # Tokenized dataset
            class TokenizedDataset:
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            return TokenizedDataset(results)
        
        return self


def create_dna_dataset(
    task: str,
    tokenizer_name: str,
    split: str,
    dataset_base_path: str,
    max_seq_length: int = 512,
    tokenize_fn_factory: callable = None,
):
    """Create a DNA dataset for NTv2, GUE, or Genomic Benchmarks.
    
    Args:
        task: Task name (e.g., 'H3K27ac' for NTv2)
        tokenizer_name: Name of tokenizer to use
        split: Data split ('train', 'validation'/'dev', 'test')
        dataset_base_path: Base path to datasets
        max_seq_length: Maximum sequence length
        tokenize_fn_factory: Optional custom tokenization function factory
    
    Returns:
        Dataset object compatible with Composer
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group="nlp", conda_package="transformers"
        ) from e
    
    # Handle tokenizer
    if tokenizer_name == "dna_char":
        from src.dna_tokenizer import DNACharacterTokenizer
        tokenizer = DNACharacterTokenizer(max_len=max_seq_length)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    log.info(f"Loading DNA dataset {task} split {split} on rank {dist.get_global_rank()}")
    
    # Convert split names
    if split == "validation":
        split = "dev"
    
    # Determine dataset type and load data
    if task in NTV2_TASK_CONFIG:
        # NTv2 dataset
        data_path = os.path.join(dataset_base_path, "NTv2", task, f"{split}.csv")
    else:
        raise ValueError(f"Unknown DNA task: {task}")
    
    # Load CSV data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Extract sequences and labels
    if "sequence" in df.columns and "label" in df.columns:
        sequences = df["sequence"].tolist()
        labels = df["label"].tolist()
    else:
        raise ValueError(f"Expected 'sequence' and 'label' columns in {data_path}")
    
    # Create dataset
    dataset = DNADataset(sequences, labels)
    
    # Create tokenization function
    if tokenize_fn_factory is None:
        if hasattr(tokenizer, '__call__'):
            # DNACharacterTokenizer
            tokenize_fn_factory = lambda tokenizer, max_seq_length: lambda inp: {
                "input_ids": [tokenizer(seq, max_length=max_seq_length)["input_ids"] for seq in inp["text"]],
                "attention_mask": [[1] * len(tokenizer(seq, max_length=max_seq_length)["input_ids"]) for seq in inp["text"]],
                "label": inp["label"]
            }
        else:
            # HuggingFace tokenizer
            tokenize_fn_factory = lambda tokenizer, max_seq_length: lambda inp: tokenizer(
                text=inp["text"],
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
                return_tensors=None,
            )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_fn_factory(tokenizer, max_seq_length),
        batched=True,
        batch_size=1000,
    )
    
    return tokenized_dataset


def create_ntv2_dataset(**kwargs):
    """Create an NTv2 dataset."""
    return create_dna_dataset(**kwargs)