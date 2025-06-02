#!/bin/bash

# Script to run all Genomic Benchmarks (GB) tasks
# Usage: ./run_all_gb_tasks.sh [bpe|char]

set -e

MODEL_TYPE=${1:-bpe}  # Default to BPE tokenization

# All GB tasks
GB_TASKS=(
    "human_nontata_promoters"
    "human_ocr_ensembl"
    "human_enhancers_ensembl"
    "mouse_enhancers_ensembl"
    "demo_human_or_worm"
    "demo_coding_vs_intergenomic"
    "drosophila_enhancers_stark"
    "human_enhancers_cohn"
    "human_ensembl_regulatory"
)

echo "Running all GB tasks with $MODEL_TYPE tokenization"
echo "Total tasks: ${#GB_TASKS[@]}"

# Create log directory
LOG_DIR="logs/gb_finetuning_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Run each task
for TASK in "${GB_TASKS[@]}"; do
    echo "=========================================="
    echo "Starting GB task: $TASK"
    echo "=========================================="
    
    LOG_FILE="$LOG_DIR/${TASK}.log"
    
    # Run the task and capture output
    ./run_gb_finetuning.sh $TASK $MODEL_TYPE 2>&1 | tee $LOG_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Task $TASK completed successfully"
    else
        echo "Task $TASK failed! Check log at $LOG_FILE"
        # Continue with other tasks even if one fails
    fi
    
    echo ""
done

echo "All GB tasks completed. Logs saved to $LOG_DIR"