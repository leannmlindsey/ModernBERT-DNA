# ModernBERT-DNA: Adapting ModernBERT for Genomic Sequence Modeling

This repository extends [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) for DNA sequence modeling, implementing both BPE and character-level tokenization strategies.

**If you use any work from this repository, please cite:**

```bibtex
@article{lindsey2024comparison,
  title={A Comparison of Tokenization Impact in Attention Based and State Space Genomic Language Models},
  author={Lindsey, LeAnn M. and Pershing, Nicole L. and Habib, Anisa and Stephens, W. Zac and Blaschke, Anne J. and Sundar, Hari},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.09.09.612081}
}
```

## Overview

This repository adapts the ModernBERT architecture for genomic sequence modeling, supporting:
- **Two tokenization strategies**: BPE (DNABERT-2) and character-level (4-mer vocabulary)
- **Pretraining** on large-scale genomic data
- **Finetuning** on DNA benchmark tasks (NTv2, GUE, GB)
- **Hyperparameter optimization** using Weights & Biases sweeps

## Setup

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate bert24

# Install flash attention 2
pip install "flash_attn==2.6.3" --no-build-isolation

# For DNA-specific dependencies
pip install biopython
```

## DNA Model Pretraining

### Starting New Pretraining

#### BPE Model
```bash
# Single node (4 GPUs)
composer -n 4 main.py \
    yamls/pretrain/modernbert-base-pretrain_modified.yaml \
    tokenizer_name=zhihan1996/DNABERT-2-117M \
    run_name=dna-modernbert-base-bpe \
    save_folder=./checkpoints/bpe_model
```

#### Character Model
```bash
# Single node (4 GPUs)
composer -n 4 main.py \
    yamls/pretrain/modernbert-base-pretrain_modified_char.yaml \
    tokenizer_name=./char_tokenizer_4kmer \
    run_name=dna-modernbert-base-char \
    save_folder=./checkpoints/char_model
```

#### SLURM Cluster
```bash
# Submit pretraining job
sbatch slurm_pretrain.sh <config_file> <output_dir>

# Example:
sbatch slurm_pretrain.sh yamls/pretrain/modernbert-base-pretrain_modified.yaml ./outputs/pretrain_bpe
```

### Resuming Pretraining

```bash
# Resume from checkpoint
sbatch slurm_pretrain_resume.sh <config_file> <output_dir> <checkpoint_path>

# Example:
sbatch slurm_pretrain_resume.sh \
    yamls/pretrain/modernbert-base-pretrain_modified.yaml \
    ./outputs/pretrain_bpe \
    ./outputs/pretrain_bpe/checkpoints/latest-rank0.pt
```

## DNA Model Finetuning

### NTv2 Benchmark Tasks

The Nucleotide Transformer v2 (NTv2) benchmark includes 18 tasks:
- **Histone modifications** (10 tasks): H3K4me3, H3K27ac, etc. (600bp sequences)
- **Enhancers** (2 tasks): Binary and type classification (200bp sequences)
- **Promoters** (3 tasks): All, TATA, and non-TATA (300bp sequences)
- **Splice sites** (3 tasks): Acceptors, donors, and all (400bp sequences)

### Running Single Task Finetuning

```bash
# BPE model on enhancers task
./run_dna_eval_single_task.sh enhancers bpe /path/to/bpe_checkpoint.pt ./outputs

# Character model on H3K4me3 task
./run_dna_eval_single_task.sh H3K4me3 char /path/to/char_checkpoint.pt ./outputs
```

### Running All NTv2 Tasks

```bash
# Run all tasks for BPE model
./run_all_ntv2_tasks.sh

# Run all tasks for character model (edit script to set MODEL_TYPE=char)
./run_all_ntv2_tasks.sh
```

### SLURM Submission

```bash
# Single task
sbatch slurm_dna_eval_single.sh ntv2 enhancers bpe /path/to/checkpoint.pt ./outputs zhihan1996/DNABERT-2-117M

# Multiple tasks (edit TASK_LIST in script)
sbatch slurm_dna_eval.sh ntv2 bpe /path/to/checkpoint.pt ./outputs
```

## Hyperparameter Optimization

We provide comprehensive hyperparameter sweep functionality using Weights & Biases.

### Quick Start
```bash
# Create sweep
./create_wandb_sweep.sh ntv2 enhancers

# Submit SLURM array job (20 parallel trials)
sbatch slurm_wandb_sweep.sh <SWEEP_ID> ntv2 enhancers bpe /path/to/checkpoint.pt
```

For detailed instructions, see [WANDB_SWEEPS_README.md](WANDB_SWEEPS_README.md).

## Output Structure

### Finetuning Outputs
```
outputs/
└── ntv2_bpe_enhancers_20241206_143022/
    ├── eval_log.txt
    ├── flex_bert_zhihan1996_DNABERT-2-117M_ntv2_enhancers_20241206_143025.csv
    └── wandb/
```

## Configuration Files

### Pretraining Configs
- `yamls/pretrain/modernbert-base-pretrain_modified.yaml` - BPE model pretraining
- `yamls/pretrain/modernbert-base-pretrain_modified_char.yaml` - Character model pretraining

### Evaluation Configs
- `yamls/dna_eval_ntv2.yaml` - NTv2 benchmark evaluation
- `yamls/dna_eval_gue.yaml` - GUE benchmark evaluation (coming soon)
- `yamls/dna_eval_gb.yaml` - Genomic Benchmarks evaluation (coming soon)

## Key Differences from Original ModernBERT

1. **Tokenization**: Support for genomic-specific tokenizers (BPE and character-level)
2. **Vocabulary Size**: 4096 for BPE, 4101 for character model
3. **RoPE Configuration**: Adapted for DNA sequences up to 10k base pairs
4. **Task-Specific Heads**: Classification heads for genomic tasks
5. **Metrics**: Added perplexity tracking during pretraining

---

For information about the original ModernBERT architecture and implementation, please refer to the [original ModernBERT repository](https://github.com/AnswerDotAI/ModernBERT).