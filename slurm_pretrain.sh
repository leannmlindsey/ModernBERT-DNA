#!/bin/bash
#SBATCH -t 48:00:00                         # Time limit (hh:mm:ss) - 48 hours for pretraining
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4                   # 4 GPUs for pretraining
#SBATCH --job-name=modernbert_pretrain      # Job name
#SBATCH -o /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/logs/pretrain_%j.outerror  # Log file

# SLURM script for ModernBERT-DNA pretraining
# Usage: sbatch slurm_pretrain.sh <model_type>
# Example: sbatch slurm_pretrain.sh bpe
# Example: sbatch slurm_pretrain.sh char

# Parse command line arguments
MODEL_TYPE=${1:-bpe}  # Default to BPE

if [ "$MODEL_TYPE" != "bpe" ] && [ "$MODEL_TYPE" != "char" ]; then
    echo "Error: MODEL_TYPE must be 'bpe' or 'char'"
    echo "Usage: sbatch slurm_pretrain.sh <model_type>"
    exit 1
fi

# Load necessary modules
module load python/3.9
module load cuda/11.8  # Adjust based on your CUDA requirements
module load gcc/11.3.0

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
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4

# Select the appropriate config file
if [ "$MODEL_TYPE" == "char" ]; then
    CONFIG_FILE="yamls/pretrain/modernbert-base-pretrain_modified_char.yaml"
    echo "Using character-level tokenization config"
else
    CONFIG_FILE="yamls/pretrain/modernbert-base-pretrain_modified.yaml"
    echo "Using BPE tokenization config"
fi

# Create output directory for checkpoints
OUTPUT_DIR="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/checkpoints/dna-${MODEL_TYPE}-modernbert-base-pretrain-4gpu"
mkdir -p $OUTPUT_DIR

# Run pretraining
echo "Starting pretraining with config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

# Use composer launcher for multi-GPU training
composer -n 4 main.py \
    $CONFIG_FILE \
    save_folder=$OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/pretrain_${SLURM_JOB_ID}.log

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