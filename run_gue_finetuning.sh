#!/bin/bash

# Script to run Genome Understanding Evaluation (GUE) fine-tuning tasks
# Usage: ./run_gue_finetuning.sh <task_name> <model_type> [additional_args]
# Example: ./run_gue_finetuning.sh human_promoter bpe

set -e  # Exit on error

# Parse command line arguments
TASK_NAME=$1
MODEL_TYPE=${2:-bpe}  # Default to BPE tokenization
shift 2
ADDITIONAL_ARGS="$@"

# Validate task name
VALID_TASKS=(
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
    # Long-range tasks (GUE+)
    "human_long_enhancers" "human_chromatin_interactions"
    "mouse_long_enhancers" "human_tad_boundaries"
    "promoter_enhancer_interactions"
)

if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK_NAME} " ]]; then
    echo "Error: Invalid task name '${TASK_NAME}'"
    echo "Valid GUE tasks are: ${VALID_TASKS[@]}"
    exit 1
fi

# Check if this is a long-range task
LONG_TASKS=(
    "human_long_enhancers" "human_chromatin_interactions"
    "mouse_long_enhancers" "human_tad_boundaries"
    "promoter_enhancer_interactions"
)

# Select base config based on model type
if [ "$MODEL_TYPE" == "char" ]; then
    BASE_CONFIG="yamls/dna_finetuning/modernbert_dna_char.yaml"
    echo "Using character-level tokenization"
else
    BASE_CONFIG="yamls/dna_finetuning/gue/gue_base.yaml"
    echo "Using BPE tokenization"
fi

# Select task config
if [[ " ${LONG_TASKS[@]} " =~ " ${TASK_NAME} " ]]; then
    TASK_CONFIG="yamls/dna_finetuning/gue/gue_long.yaml"
    echo "Using long-range task configuration"
else
    TASK_CONFIG="yamls/dna_finetuning/gue/gue_tasks.yaml"
fi

# Create output directory
OUTPUT_DIR="outputs/gue/${TASK_NAME}_${MODEL_TYPE}"
mkdir -p $OUTPUT_DIR

# Run training
echo "Running GUE fine-tuning for task: $TASK_NAME"
echo "Base config: $BASE_CONFIG"
echo "Task config: $TASK_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Additional arguments: $ADDITIONAL_ARGS"

# Merge configs and run training
python dna_sequence_classification.py \
    $BASE_CONFIG \
    $TASK_CONFIG \
    task_name=$TASK_NAME \
    save_folder=$OUTPUT_DIR/checkpoints \
    run_name=gue_${TASK_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S) \
    $ADDITIONAL_ARGS