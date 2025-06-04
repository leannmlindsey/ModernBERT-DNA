#!/bin/bash
#SBATCH -t 48:00:00                         # Time limit (hh:mm:ss) - 48 hours for pretraining
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4                   # 4 GPUs for pretraining
#SBATCH --job-name=modernbert_resume        # Job name
#SBATCH -o /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/logs/pretrain_resume_%j.outerror  # Log file

# SLURM script for resuming ModernBERT-DNA pretraining from checkpoint
# Usage: sbatch slurm_pretrain_resume.sh <model_type> [checkpoint_path]
# Example: sbatch slurm_pretrain_resume.sh bpe
# Example: sbatch slurm_pretrain_resume.sh char /path/to/checkpoint.pt

# Parse command line arguments
MODEL_TYPE=${1:-bpe}  # Default to BPE
CHECKPOINT_PATH=$2

if [ "$MODEL_TYPE" != "bpe" ] && [ "$MODEL_TYPE" != "char" ]; then
    echo "Error: MODEL_TYPE must be 'bpe' or 'char'"
    echo "Usage: sbatch slurm_pretrain_resume.sh <model_type> [checkpoint_path]"
    exit 1
fi

# Activate conda environment
conda activate bert24_2

# Change to project directory
cd /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT-DNA

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of GPUs: $SLURM_GPUS"
echo "=========================================="
echo "Model Type: $MODEL_TYPE"
echo "Resuming from checkpoint: ${CHECKPOINT_PATH:-auto}"
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4

# Select the appropriate config file and checkpoint
if [ "$MODEL_TYPE" == "char" ]; then
    CONFIG_FILE="yamls/pretrain/modernbert-base-pretrain_modified_char.yaml"
    DEFAULT_CHECKPOINT="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/dnamodernbertbase/checkpoints/dna-char-modernbert-basemod-pretrain-4gpu/ep0-ba70000-rank0.pt"
    OUTPUT_DIR="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/checkpoints/dna-char-modernbert-base-pretrain-4gpu-continued"
    echo "Using character-level tokenization config"
else
    CONFIG_FILE="yamls/pretrain/modernbert-base-pretrain_modified.yaml"
    DEFAULT_CHECKPOINT="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/dnamodernbertbase/checkpoints/dna-modernbert-basemod-pretrain-4gpu/ep2-ba52000-rank0.pt"
    OUTPUT_DIR="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/checkpoints/dna-bpe-modernbert-base-pretrain-4gpu-continued"
    echo "Using BPE tokenization config"
fi

# Use provided checkpoint or default
if [ -z "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH=$DEFAULT_CHECKPOINT
    echo "Using default checkpoint: $CHECKPOINT_PATH"
fi

# Create output directory for checkpoints
mkdir -p $OUTPUT_DIR

# Run pretraining with resume
echo "Starting pretraining with config: $CONFIG_FILE"
echo "Resuming from checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

# Additional arguments for resuming
RESUME_ARGS="load_path=$CHECKPOINT_PATH autoresume=true restart_override=true"

# Use composer launcher for multi-GPU training
composer -n 4 main.py \
    $CONFIG_FILE \
    save_folder=$OUTPUT_DIR \
    $RESUME_ARGS \
    2>&1 | tee $OUTPUT_DIR/pretrain_resume_${SLURM_JOB_ID}.log

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Job completed successfully"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Job failed with exit code: $?"
    echo "End time: $(date)"
    echo "=========================================="
    exit 1
fi
