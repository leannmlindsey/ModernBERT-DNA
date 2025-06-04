# WandB Sweeps for DNA Model Finetuning

This document describes how to run hyperparameter sweeps for DNA model finetuning using Weights & Biases (WandB).

## Overview

The sweep system allows you to automatically search for optimal hyperparameters across:
- Learning rate (1e-6 to 1e-4, log scale)
- Weight decay (1e-6 to 1e-4, log scale)
- Batch size (16, 32, 64)
- Warmup ratio (0.0 to 0.1)
- Number of epochs (3, 5, 10)
- Optimizer betas
- Gradient clipping (0.0, 0.5, 1.0)
- Scheduler type (constant, cosine, linear decay)

## Files

- `wandb_sweep_dna_finetuning.yaml`: Sweep configuration defining hyperparameter search space
- `run_wandb_sweep_dna.py`: Python script that runs a single sweep trial
- `create_wandb_sweep.sh`: Creates a new sweep and returns the sweep ID
- `launch_wandb_sweep.sh`: Local script for running sweeps on a single machine
- `../slurm/slurm_wandb_sweep.sh`: SLURM script for running sweep agents on a cluster

## Workflow

### Option 1: Running on SLURM Cluster (Recommended)

1. **Create the sweep:**
   ```bash
   ./scripts/hyperparameter_tuning/create_wandb_sweep.sh ntv2 enhancers
   ```
   This will output a SWEEP_ID.

2. **Submit SLURM array job:**
   ```bash
   # For BPE model
   sbatch scripts/slurm/slurm_wandb_sweep.sh <SWEEP_ID> ntv2 enhancers bpe /path/to/pretrained/checkpoint.pt
   
   # For character model
   sbatch scripts/slurm/slurm_wandb_sweep.sh <SWEEP_ID> ntv2 enhancers char /path/to/pretrained/checkpoint.pt
   ```

   The SLURM script will:
   - Launch 20 parallel jobs (configurable via `--array=1-20`)
   - Each job runs exactly 1 hyperparameter configuration (`--count 1`)
   - Follows WandB best practices for SLURM integration

3. **Monitor progress:**
   ```
   https://wandb.ai/leannmlindsey/modernBERT-DNA-sweeps/sweeps/<SWEEP_ID>
   ```

### Option 2: Running Locally

1. **Launch sweep with multiple agents:**
   ```bash
   ./scripts/hyperparameter_tuning/launch_wandb_sweep.sh ntv2 enhancers bpe /path/to/checkpoint.pt 4
   ```
   This will create a sweep and launch 4 parallel agents locally.

## Example Commands

### NTv2 Benchmark
```bash
# Create sweep
./scripts/hyperparameter_tuning/create_wandb_sweep.sh ntv2 enhancers

# Run on SLURM (assuming sweep ID is abc123)
sbatch scripts/slurm/slurm_wandb_sweep.sh abc123 ntv2 enhancers bpe /data/models/modernbert-base-dna.pt
```

### GUE Benchmark
```bash
# Create sweep
./scripts/hyperparameter_tuning/create_wandb_sweep.sh gue human_h3k4me3

# Run on SLURM
sbatch scripts/slurm/slurm_wandb_sweep.sh abc123 gue human_h3k4me3 bpe /data/models/modernbert-base-dna.pt
```

### GB Benchmark
```bash
# Create sweep
./scripts/hyperparameter_tuning/create_wandb_sweep.sh gb human_nontata_promoters

# Run on SLURM
sbatch scripts/slurm/slurm_wandb_sweep.sh abc123 gb human_nontata_promoters char /data/models/modernbert-base-dna-char.pt
```

## Customizing Hyperparameter Search

To modify the search space, edit `wandb_sweep_dna_finetuning.yaml`:

```yaml
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-4
  
  batch_size:
    values: [16, 32, 64, 128]  # Add more values
  
  # Add new parameters
  new_param:
    distribution: uniform
    min: 0.0
    max: 1.0
```

## Output Structure

Results are saved to:
```
outputs/wandb_sweeps/
└── <SWEEP_ID>/
    └── <benchmark>_<task>_<model_type>/
        └── run_<RUN_ID>/
            └── <task>_<model_type>/
                ├── eval_log.txt
                ├── metrics_*.csv
                └── wandb/
```

## Integration with Existing Code

The sweep system integrates with the existing DNA evaluation pipeline:
- Uses the same `dna_eval.py` script
- Respects all configuration in `yamls/dna_eval_*.yaml`
- Saves metrics to CSV files
- Logs all metrics to WandB for analysis

## Best Practices

1. **Start with fewer trials**: Test with `--array=1-5` before scaling up
2. **Monitor early results**: Check WandB dashboard to see if search space is reasonable
3. **Use early stopping**: The sweep config includes Hyperband early termination
4. **Resource allocation**: Each trial needs 1 GPU, plan accordingly
5. **Checkpoint management**: The system saves checkpoints, but consider cleanup after sweeps

## Troubleshooting

1. **Sweep creation fails**: Check WandB login (`wandb login`)
2. **SLURM jobs fail**: Check logs in `logs/wandb_sweep_*.out`
3. **Out of memory**: Reduce batch size in sweep config
4. **Slow training**: Check if gradient accumulation is needed for large batch sizes