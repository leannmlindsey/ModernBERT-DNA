#!/bin/bash

# Create a WandB sweep for DNA model hyperparameter tuning
# This should be run once before submitting SLURM array jobs

set -e

# Parse arguments
BENCHMARK=$1
TASK=$2

if [ -z "$BENCHMARK" ] || [ -z "$TASK" ]; then
    echo "Usage: $0 <benchmark> <task>"
    echo "  benchmark: ntv2, gue, or gb"
    echo "  task: specific task name (e.g., enhancers, H3K4me3)"
    echo ""
    echo "This will create a sweep and output the SWEEP_ID to use with slurm_wandb_sweep.sh"
    exit 1
fi

# Validate benchmark
case $BENCHMARK in
    ntv2|gue|gb)
        ;;
    *)
        echo "Invalid benchmark: $BENCHMARK (must be ntv2, gue, or gb)"
        exit 1
        ;;
esac

echo "Creating WandB sweep for $BENCHMARK / $TASK..."

# Create the sweep and capture the output
SWEEP_OUTPUT=$(wandb sweep wandb_sweep_dna_finetuning.yaml \
    --project modernBERT-DNA-sweeps \
    --entity leannmlindsey \
    --name "${BENCHMARK}_${TASK}_sweep" 2>&1)

# Extract the sweep ID from the output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^ ]+$' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create sweep. Output was:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

echo ""
echo "================================================"
echo "Sweep created successfully!"
echo "Sweep ID: $SWEEP_ID"
echo "================================================"
echo ""
echo "To run the sweep on SLURM, use:"
echo "sbatch slurm_wandb_sweep.sh $SWEEP_ID $BENCHMARK $TASK <model_type> <pretrained_checkpoint>"
echo ""
echo "Example:"
echo "sbatch slurm_wandb_sweep.sh $SWEEP_ID $BENCHMARK $TASK bpe /path/to/checkpoint.pt"
echo ""
echo "Monitor at: https://wandb.ai/leannmlindsey/modernBERT-DNA-sweeps/sweeps/$SWEEP_ID"