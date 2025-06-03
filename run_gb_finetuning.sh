#!/bin/bash

# Script to run Genomic Benchmarks (GB) fine-tuning tasks
# Usage: ./run_gb_finetuning.sh <task_name> <model_type> [additional_args]
# Example: ./run_gb_finetuning.sh human_nontata_promoters bpe

set -e  # Exit on error

# Parse command line arguments
TASK_NAME=$1
MODEL_TYPE=${2:-bpe}  # Default to BPE tokenization
shift 2
ADDITIONAL_ARGS="$@"

# Validate task name
VALID_TASKS=(
    "human_nontata_promoters"
    "human_ocr_ensembl"
    "human_enhancers_ensembl"
    "dummy_mouse_enhancers"  # Note: directory shows 'dummy_mouse_enhancers'
    "demo_human_or_worm"
    "demo_coding_vs_intergenomic"
    "drosophilia_enhancers"  # Note: directory shows 'drosophilia' (with i)
    "human_enhancers_cohn"
    "human_ensembl_regulatory"
)

if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK_NAME} " ]]; then
    echo "Error: Invalid task name '${TASK_NAME}'"
    echo "Valid GB tasks are: ${VALID_TASKS[@]}"
    exit 1
fi

# Use consolidated YAML config
CONFIG_FILE="yamls/dna_finetuning/gb.yaml"

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
OUTPUT_DIR="outputs/gb/${TASK_NAME}_${MODEL_TYPE}"
mkdir -p $OUTPUT_DIR

# Run training
echo "Running GB fine-tuning for task: $TASK_NAME"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Additional arguments: $ADDITIONAL_ARGS"

# Run training with consolidated config
python dna_sequence_classification.py \
    $CONFIG_FILE \
    task_name=$TASK_NAME \
    tokenizer_name=$TOKENIZER \
    model.tokenizer_name=$TOKENIZER \
    model.pretrained_checkpoint=$MODEL_PATH \
    model.model_config.vocab_size=$VOCAB_SIZE \
    save_folder=$OUTPUT_DIR/checkpoints \
    run_name=gb_${TASK_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S) \
    $ADDITIONAL_ARGS