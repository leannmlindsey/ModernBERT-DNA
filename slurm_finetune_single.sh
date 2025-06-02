#!/bin/bash
#SBATCH -t 8:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=modernbert_finetune      # Job name
#SBATCH -o /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/logs/finetune_%j.outerror  # Log file

# SLURM script for running single ModernBERT-DNA fine-tuning task
# Usage: sbatch slurm_finetune_single.sh <benchmark> <task_name> <model_type>
# Example: sbatch slurm_finetune_single.sh ntv2 H3K27ac bpe
# Example: sbatch slurm_finetune_single.sh gue human_h3k4me3 char
# Example: sbatch slurm_finetune_single.sh gb human_nontata_promoters bpe

# Parse command line arguments
BENCHMARK=$1
TASK_NAME=$2
MODEL_TYPE=${3:-bpe}  # Default to BPE

if [ -z "$BENCHMARK" ] || [ -z "$TASK_NAME" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch slurm_finetune_single.sh <benchmark> <task_name> <model_type>"
    echo "Benchmarks: ntv2, gb, gue"
    exit 1
fi

# Load necessary modules
module load python/3.9
module load cuda/11.8  # Adjust based on your CUDA requirements
module load gcc/11.3.0

# Activate conda environment
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate bert24

# Change to project directory
cd /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT-DNA

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of GPUs: $SLURM_GPUS"
echo "=========================================="
echo "Benchmark: $BENCHMARK"
echo "Task: $TASK_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables for single GPU
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1

# Run the appropriate fine-tuning script
case $BENCHMARK in
    "ntv2")
        echo "Running NT v2 fine-tuning for task: $TASK_NAME"
        ./run_ntv2_finetuning.sh $TASK_NAME $MODEL_TYPE
        ;;
    "gb")
        echo "Running GB fine-tuning for task: $TASK_NAME"
        ./run_gb_finetuning.sh $TASK_NAME $MODEL_TYPE
        ;;
    "gue")
        echo "Running GUE fine-tuning for task: $TASK_NAME"
        ./run_gue_finetuning.sh $TASK_NAME $MODEL_TYPE
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo "Valid benchmarks are: ntv2, gb, gue"
        exit 1
        ;;
esac

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