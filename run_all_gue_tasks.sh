#!/bin/bash

# Script to run all Genome Understanding Evaluation (GUE) tasks
# Usage: ./run_all_gue_tasks.sh [bpe|char] [all|standard|long]

set -e

MODEL_TYPE=${1:-bpe}  # Default to BPE tokenization
TASK_SET=${2:-all}    # all, standard, or long

# Standard GUE tasks
STANDARD_TASKS=(
    # Promoter tasks
    "human_promoter" "mouse_promoter"
    "human_core_promoter" "mouse_core_promoter"
    # Splice site tasks
    "human_splice_acceptor" "human_splice_donor"
    "mouse_splice_acceptor" "mouse_splice_donor"
    # COVID variant
    "covid_variant"
    # Epigenetic marks
    "human_h3k4me3" "human_h3k27me3" "human_h3k36me3" "human_h3k9me3" "human_h3k27ac"
    "mouse_h3k4me3" "mouse_h3k27me3" "mouse_h3k36me3"
    # TF binding sites
    "human_tfbs_ctcf" "human_tfbs_nfkb" "human_tfbs_cebpb" "human_tfbs_foxa1"
    "mouse_tfbs_ctcf" "mouse_tfbs_cebpb"
    # Species-specific
    "yeast_utr3" "yeast_utr5" "fungi_promoter" "virus_classification"
)

# Long-range tasks (GUE+)
LONG_TASKS=(
    "human_long_enhancers"
    "human_chromatin_interactions"
    "mouse_long_enhancers"
    "human_tad_boundaries"
    "promoter_enhancer_interactions"
)

# Select tasks based on task set
case $TASK_SET in
    "standard")
        TASKS=("${STANDARD_TASKS[@]}")
        echo "Running standard GUE tasks"
        ;;
    "long")
        TASKS=("${LONG_TASKS[@]}")
        echo "Running GUE+ (long-range) tasks"
        ;;
    "all")
        TASKS=("${STANDARD_TASKS[@]}" "${LONG_TASKS[@]}")
        echo "Running all GUE tasks (standard + long-range)"
        ;;
    *)
        echo "Invalid task set: $TASK_SET. Use 'all', 'standard', or 'long'"
        exit 1
        ;;
esac

echo "Running ${#TASKS[@]} tasks with $MODEL_TYPE tokenization"

# Create log directory
LOG_DIR="logs/gue_finetuning_${MODEL_TYPE}_${TASK_SET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Run each task
for TASK in "${TASKS[@]}"; do
    echo "=========================================="
    echo "Starting GUE task: $TASK"
    echo "=========================================="
    
    LOG_FILE="$LOG_DIR/${TASK}.log"
    
    # Run the task and capture output
    ./run_gue_finetuning.sh $TASK $MODEL_TYPE 2>&1 | tee $LOG_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Task $TASK completed successfully"
    else
        echo "Task $TASK failed! Check log at $LOG_FILE"
        # Continue with other tasks even if one fails
    fi
    
    echo ""
done

echo "All GUE tasks completed. Logs saved to $LOG_DIR"