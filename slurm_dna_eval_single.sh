#!/bin/bash
#SBATCH -t 4:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dna_eval                 # Job name
#SBATCH -o /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/logs/dna_eval_%j.outerror  # Log file

# SLURM script for running single DNA evaluation task
# Usage: sbatch slurm_dna_eval_single.sh <task_name> <model_type> [eval_on_test] [tokenizer] [output_parent_dir]
# Example: sbatch slurm_dna_eval_single.sh enhancers bpe true
# Example: sbatch slurm_dna_eval_single.sh H3K27ac char false
# Example: sbatch slurm_dna_eval_single.sh enhancers bpe true InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
# Example: sbatch slurm_dna_eval_single.sh enhancers bpe true "" /my/custom/output/path

# Parse command line arguments
TASK_NAME=$1
MODEL_TYPE=${2:-bpe}  # Default to BPE
EVAL_ON_TEST=${3:-true}  # Default to evaluating on test set
CUSTOM_TOKENIZER=$4  # Optional custom tokenizer
OUTPUT_PARENT_DIR=$5  # Optional output parent directory

if [ -z "$TASK_NAME" ]; then
    echo "Error: Missing required task name"
    echo "Usage: sbatch slurm_dna_eval_single.sh <task_name> <model_type> [eval_on_test] [tokenizer]"
    echo "Tasks: enhancers, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K9ac, H3K9me3, etc."
    echo "Model types: bpe, char"
    echo "Tokenizers: zhihan1996/DNABERT-2-117M, InstaDeepAI/nucleotide-transformer-v2-100m-multi-species, dna_char, etc."
    exit 1
fi

# Load necessary modules
#module load python/3.9
#module load cuda/11.8  # Adjust based on your CUDA requirements
#module load gcc/11.3.0

# Activate conda environment
source activate bert24_2

# Change to project directory
cd /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT-DNA

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of GPUs: $SLURM_GPUS"
echo "=========================================="
echo "Task: $TASK_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Evaluate on test: $EVAL_ON_TEST"
if [ ! -z "$CUSTOM_TOKENIZER" ]; then
    echo "Custom Tokenizer: $CUSTOM_TOKENIZER"
fi
if [ ! -z "$OUTPUT_PARENT_DIR" ]; then
    echo "Output Parent Directory: $OUTPUT_PARENT_DIR"
fi
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables for single GPU
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1

# Run DNA evaluation
echo "Running DNA evaluation for $TASK_NAME task with $MODEL_TYPE tokenizer"

# Build the command
CMD="./run_dna_eval_single_task.sh $TASK_NAME $MODEL_TYPE eval_on_test=$EVAL_ON_TEST"

if [ ! -z "$CUSTOM_TOKENIZER" ]; then
    CMD="$CMD tokenizer_name=$CUSTOM_TOKENIZER"
fi

if [ ! -z "$OUTPUT_PARENT_DIR" ]; then
    CMD="$CMD output_parent_dir=$OUTPUT_PARENT_DIR"
fi

# Execute the command
$CMD

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
