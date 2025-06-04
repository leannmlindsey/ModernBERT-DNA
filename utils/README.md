# Utility Scripts

This directory contains utility and helper scripts that support the main ModernBERT-DNA workflows.

## Scripts

### WandB Utilities
- `analyze_wandb_run.py` - Analyze metrics and performance from WandB runs
- `download_artifacts_from_wandb.py` - Download model artifacts from WandB
- `extract_wandb_info.py` - Extract and summarize information from WandB runs

### Model Utilities
- `check_checkpoint.py` - Inspect and verify model checkpoints
- `convert_to_hf.py` - Convert models to HuggingFace format

### Configuration Utilities
- `generate_eval_config.py` - Generate evaluation configuration files
- `run_evals.py` - Run evaluation workflows (wrapper script)

## Usage Examples

### Check a checkpoint
```bash
python utils/check_checkpoint.py /path/to/checkpoint.pt
```

### Convert to HuggingFace format
```bash
python utils/convert_to_hf.py --checkpoint /path/to/checkpoint.pt --output /path/to/hf_model
```

### Download WandB artifacts
```bash
python utils/download_artifacts_from_wandb.py --run-id <RUN_ID> --project <PROJECT>
```