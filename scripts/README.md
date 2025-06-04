# Scripts Directory Structure

This directory contains all scripts for ModernBERT-DNA organized by functionality:

## Directory Organization

### `slurm/`
Contains all SLURM submission scripts for cluster computing:
- `slurm_pretrain.sh` - Submit pretraining jobs
- `slurm_pretrain_resume.sh` - Resume interrupted pretraining
- `slurm_dna_eval.sh` - Run multiple DNA finetuning tasks
- `slurm_dna_eval_single.sh` - Run single DNA finetuning task
- `slurm_finetune_single.sh` - General single task finetuning
- `slurm_wandb_sweep.sh` - Submit WandB hyperparameter sweep jobs

### `finetuning/`
Contains scripts for running finetuning and evaluation:
- `run_dna_eval.sh` - Run DNA model evaluation
- `run_dna_eval_single_task.sh` - Run single DNA evaluation task
- `run_all_ntv2_tasks.sh` - Run all NTv2 benchmark tasks
- `run_all_gue_tasks.sh` - Run all GUE benchmark tasks
- `run_all_gb_tasks.sh` - Run all Genomic Benchmarks tasks
- `run_ntv2_finetuning.sh` - Run NTv2 finetuning
- `run_gue_finetuning.sh` - Run GUE finetuning
- `run_gb_finetuning.sh` - Run GB finetuning
- `run_benchmark.sh` - General benchmark runner

### `hyperparameter_tuning/`
Contains scripts and configs for WandB hyperparameter sweeps:
- `create_wandb_sweep.sh` - Create new sweep
- `launch_wandb_sweep.sh` - Launch sweep locally
- `run_wandb_sweep_dna.py` - Python script for individual sweep runs
- `wandb_sweep_dna_finetuning.yaml` - Sweep configuration
- `WANDB_SWEEPS_README.md` - Detailed documentation

## Usage Examples

### Pretraining
```bash
# Start new pretraining
sbatch scripts/slurm/slurm_pretrain.sh yamls/pretrain/modernbert-base-pretrain_modified.yaml ./outputs

# Resume pretraining
sbatch scripts/slurm/slurm_pretrain_resume.sh yamls/pretrain/modernbert-base-pretrain_modified.yaml ./outputs ./checkpoint.pt
```

### Finetuning
```bash
# Run single task
./scripts/finetuning/run_dna_eval_single_task.sh enhancers bpe /path/to/checkpoint.pt ./outputs

# Run all NTv2 tasks
./scripts/finetuning/run_all_ntv2_tasks.sh
```

### Hyperparameter Tuning
```bash
# Create and run sweep
./scripts/hyperparameter_tuning/create_wandb_sweep.sh ntv2 enhancers
sbatch scripts/slurm/slurm_wandb_sweep.sh <SWEEP_ID> ntv2 enhancers bpe /path/to/checkpoint.pt
```