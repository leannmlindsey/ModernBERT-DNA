#!/bin/bash

# Script to run all Genome Understanding Evaluation (GUE) tasks
# Usage: ./run_all_gue_tasks.sh [bpe|char] [all|standard|long]

set -e

MODEL_TYPE=${1:-bpe}  # Default to BPE tokenization
TASK_SET=${2:-all}    # all, standard, or long

# Standard GUE tasks
STANDARD_TASKS=(
    # EMP tasks (Epigenetic Marks Prediction)
    "human_h3" "human_h3k14ac" "human_h3k36me3" "human_h3k4me1" "human_h3k4me2"
    "human_h3k4me3" "human_h3k79me3" "human_h3k9ac" "human_h4" "human_h4ac"
    # Mouse tasks
    "mouse_0" "mouse_1" "mouse_2" "mouse_3" "mouse_4"
    # Promoter tasks
    "prom_300_all" "prom_300_notata" "prom_300_tata"
    "prom_core_all" "prom_core_notata" "prom_core_tata"
    # Splice tasks
    "splice_reconstructed"
    # TF binding sites
    "tf_0" "tf_1" "tf_2" "tf_3" "tf_4"
    # Virus tasks
    "virus_covid"
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