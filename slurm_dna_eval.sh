#!/bin/bash
#SBATCH -t 8:00:00                          # Time limit (hh:mm:ss) - increased for multiple tasks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dna_eval_benchmark       # Job name
#SBATCH -o /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/logs/dna_eval_%j.outerror  # Log file

# SLURM script for running DNA evaluation on all tasks in a benchmark
# Usage: sbatch slurm_dna_eval.sh <benchmark> <model_type> [eval_on_test] [tokenizer]
# Example: sbatch slurm_dna_eval.sh ntv2 bpe true
# Example: sbatch slurm_dna_eval.sh ntv2 char false
# Example: sbatch slurm_dna_eval.sh ntv2 bpe true InstaDeepAI/nucleotide-transformer-v2-100m-multi-species

# Parse command line arguments
BENCHMARK=$1
MODEL_TYPE=${2:-bpe}  # Default to BPE
EVAL_ON_TEST=${3:-true}  # Default to evaluating on test set
CUSTOM_TOKENIZER=$4  # Optional custom tokenizer

if [ -z "$BENCHMARK" ]; then
    echo "Error: Missing required benchmark name"
    echo "Usage: sbatch slurm_dna_eval.sh <benchmark> <model_type> [eval_on_test] [tokenizer]"
    echo "Benchmarks: ntv2, gue (not implemented yet), gb (not implemented yet)"
    echo "Model types: bpe, char"
    exit 1
fi

# Define tasks for each benchmark
case $BENCHMARK in
    "ntv2")
        TASKS="H2AFZ,H3K27ac,H3K27me3,H3K36me3,H3K4me1,H3K4me2,H3K4me3,H3K9ac,H3K9me3,H4K20me1,enhancers,enhancers_types,promoter_all,promoter_no_tata,promoter_tata,splice_sites_acceptors,splice_sites_all,splice_sites_donors"
        ;;
    "gue")
        echo "Error: GUE benchmark not implemented yet"
        exit 1
        ;;
    "gb")
        echo "Error: GB (Genomic Benchmarks) not implemented yet"
        exit 1
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo "Valid benchmarks: ntv2, gue (not implemented), gb (not implemented)"
        exit 1
        ;;
esac

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
echo "Benchmark: $BENCHMARK"
echo "Tasks: $TASKS"
echo "Model Type: $MODEL_TYPE"
echo "Evaluate on test: $EVAL_ON_TEST"
if [ ! -z "$CUSTOM_TOKENIZER" ]; then
    echo "Custom Tokenizer: $CUSTOM_TOKENIZER"
fi
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables for single GPU
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1

# Run DNA evaluation for all tasks
echo "Running DNA evaluation for all $BENCHMARK tasks with $MODEL_TYPE tokenizer"

# Convert comma-separated tasks to list format for eval_tasks parameter
TASK_LIST="[${TASKS}]"

if [ ! -z "$CUSTOM_TOKENIZER" ]; then
    # If custom tokenizer is provided, override the default
    ./run_dna_eval.sh $MODEL_TYPE eval_tasks=$TASK_LIST eval_on_test=$EVAL_ON_TEST tokenizer_name=$CUSTOM_TOKENIZER benchmark=$BENCHMARK
else
    # Use default tokenizer based on model type
    ./run_dna_eval.sh $MODEL_TYPE eval_tasks=$TASK_LIST eval_on_test=$EVAL_ON_TEST benchmark=$BENCHMARK
fi

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