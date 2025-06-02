#!/bin/bash

# Script to run all NT v2 DNA classification tasks
# Usage: ./run_all_dna_tasks.sh [bpe|char]

set -e

MODEL_TYPE=${1:-bpe}  # Default to BPE tokenization

# All NT v2 tasks
HISTONE_TASKS=(
    "H2AFZ" "H3K27ac" "H3K27me3" "H3K36me3" "H3K4me1" 
    "H3K4me2" "H3K4me3" "H3K9ac" "H3K9me3" "H4K20me1"
)

ENHANCER_TASKS=("enhancers" "enhancers_types")
PROMOTER_TASKS=("promoter_all" "promoter_no_tata" "promoter_tata")
SPLICE_TASKS=("splice_sites_acceptors" "splice_sites_all" "splice_sites_donors")

ALL_TASKS=("${HISTONE_TASKS[@]}" "${ENHANCER_TASKS[@]}" "${PROMOTER_TASKS[@]}" "${SPLICE_TASKS[@]}")

echo "Running all NT v2 tasks with $MODEL_TYPE tokenization"
echo "Total tasks: ${#ALL_TASKS[@]}"

# Create log directory
LOG_DIR="logs/dna_finetuning_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Run each task
for TASK in "${ALL_TASKS[@]}"; do
    echo "=========================================="
    echo "Starting task: $TASK"
    echo "=========================================="
    
    LOG_FILE="$LOG_DIR/${TASK}.log"
    
    # Run the task and capture output
    ./run_dna_finetuning.sh $TASK $MODEL_TYPE 2>&1 | tee $LOG_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Task $TASK completed successfully"
    else
        echo "Task $TASK failed! Check log at $LOG_FILE"
        # Continue with other tasks even if one fails
    fi
    
    echo ""
done

echo "All tasks completed. Logs saved to $LOG_DIR"