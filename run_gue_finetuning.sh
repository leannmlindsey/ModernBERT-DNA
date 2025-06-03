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
    # Long-range tasks (GUE+) - if you have these
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

# Use consolidated YAML config
CONFIG_FILE="yamls/dna_finetuning/gue.yaml"

# Select model path based on model type
if [ "$MODEL_TYPE" == "char" ]; then
    TOKENIZER="dna_char"
    MODEL_PATH="./checkpoints/modernbert-dna-base-char/checkpoint.pt"
    VOCAB_SIZE=10
    echo "Using character-level tokenization"
else
    TOKENIZER="zhihan1996/DNABERT-2-117M"
    MODEL_PATH="./checkpoints/modernbert-dna-base-bpe/checkpoint.pt"
    VOCAB_SIZE=4096
    echo "Using BPE tokenization"
fi

# Create output directory
OUTPUT_DIR="outputs/gue/${TASK_NAME}_${MODEL_TYPE}"
mkdir -p $OUTPUT_DIR

# Run training
echo "Running GUE fine-tuning for task: $TASK_NAME"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Additional arguments: $ADDITIONAL_ARGS"

# Run training with consolidated config
python dna_sequence_classification.py \
    $CONFIG_FILE \
    task_name=$TASK_NAME \
    tokenizer_name=$TOKENIZER \
    model.tokenizer_name=$TOKENIZER \
    model.pretrained_model_name=$TOKENIZER \
    model.pretrained_checkpoint=$MODEL_PATH \
    model.model_config.vocab_size=$VOCAB_SIZE \
    save_folder=$OUTPUT_DIR/checkpoints \
    run_name=gue_${TASK_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S) \
    $ADDITIONAL_ARGS