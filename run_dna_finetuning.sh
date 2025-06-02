#!/bin/bash

# Script to run DNA sequence classification fine-tuning on NT v2 tasks
# Usage: ./run_dna_finetuning.sh <task_name> <model_type> [additional_args]
# Example: ./run_dna_finetuning.sh H3K27ac bpe
# Example: ./run_dna_finetuning.sh enhancers char --seed 42

set -e  # Exit on error

# Parse command line arguments
TASK_NAME=$1
MODEL_TYPE=${2:-bpe}  # Default to BPE tokenization
shift 2
ADDITIONAL_ARGS="$@"

# Validate task name
VALID_TASKS=(
    "H2AFZ" "H3K27ac" "H3K27me3" "H3K36me3" "H3K4me1" 
    "H3K4me2" "H3K4me3" "H3K9ac" "H3K9me3" "H4K20me1"
    "enhancers" "enhancers_types"
    "promoter_all" "promoter_no_tata" "promoter_tata"
    "splice_sites_acceptors" "splice_sites_all" "splice_sites_donors"
)

if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK_NAME} " ]]; then
    echo "Error: Invalid task name '${TASK_NAME}'"
    echo "Valid tasks are: ${VALID_TASKS[@]}"
    exit 1
fi

# Determine which config to use based on task category
if [[ $TASK_NAME == H* ]] || [[ $TASK_NAME == "H4K20me1" ]]; then
    TASK_CONFIG="yamls/dna_finetuning/histone_modifications.yaml"
elif [[ $TASK_NAME == enhancer* ]]; then
    TASK_CONFIG="yamls/dna_finetuning/enhancers.yaml"
elif [[ $TASK_NAME == promoter* ]]; then
    TASK_CONFIG="yamls/dna_finetuning/promoters.yaml"
elif [[ $TASK_NAME == splice_sites* ]]; then
    TASK_CONFIG="yamls/dna_finetuning/splice_sites.yaml"
fi

# Select base config based on model type
if [ "$MODEL_TYPE" == "char" ]; then
    BASE_CONFIG="yamls/dna_finetuning/modernbert_dna_char.yaml"
    echo "Using character-level tokenization"
else
    BASE_CONFIG="yamls/dna_finetuning/modernbert_dna_base.yaml"
    echo "Using BPE tokenization"
fi

# Create output directory
OUTPUT_DIR="outputs/dna_finetuning/${TASK_NAME}_${MODEL_TYPE}"
mkdir -p $OUTPUT_DIR

# Run training
echo "Running fine-tuning for task: $TASK_NAME"
echo "Task config: $TASK_CONFIG"
echo "Base config: $BASE_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Additional arguments: $ADDITIONAL_ARGS"

# Merge configs and run training
# The configs are merged in order: defaults.yaml -> base_config -> task_config -> CLI args
python dna_sequence_classification.py \
    $BASE_CONFIG \
    $TASK_CONFIG \
    task_name=$TASK_NAME \
    save_folder=$OUTPUT_DIR/checkpoints \
    run_name=${TASK_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S) \
    $ADDITIONAL_ARGS