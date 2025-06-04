#!/bin/bash

# Script to run DNA model evaluation on a single NTv2 task
# This ensures each task gets its own timestamped directory
# Usage: ./run_dna_eval_single_task.sh <task_name> <model_type> [additional_args]
# Example: ./run_dna_eval_single_task.sh enhancers bpe eval_on_test=true
# Example: ./run_dna_eval_single_task.sh H3K27ac char eval_on_test=true output_parent_dir=/my/custom/path
# Example: ./run_dna_eval_single_task.sh enhancers bpe eval_on_test=true tokenizer_name=custom_tokenizer output_parent_dir=/my/custom/path

set -e  # Exit on error

# Parse command line arguments
TASK_NAME=$1
MODEL_TYPE=${2:-bpe}  # Default to BPE tokenization
shift 2

# Parse remaining arguments and look for output_parent_dir
ADDITIONAL_ARGS=""
OUTPUT_PARENT_DIR=""
for arg in "$@"; do
    if [[ $arg == output_parent_dir=* ]]; then
        OUTPUT_PARENT_DIR="${arg#output_parent_dir=}"
    else
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS $arg"
    fi
done

# Set default parent directory if not provided
if [ -z "$OUTPUT_PARENT_DIR" ]; then
    OUTPUT_PARENT_DIR="outputs/dna_eval"
fi

if [ -z "$TASK_NAME" ]; then
    echo "Error: Task name is required"
    echo "Usage: ./run_dna_eval_single_task.sh <task_name> <model_type> [additional_args]"
    exit 1
fi

# Configuration file
CONFIG_FILE="yamls/dna_eval_ntv2.yaml"

# Select model path and tokenizer based on model type
if [ "$MODEL_TYPE" == "char" ]; then
    TOKENIZER="dna_char"
    MODEL_PATH="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/dnamodernbertbase/checkpoints/dna-char-modernbert-basemod-pretrain-4gpu/ep0-ba70000-rank0.pt"
    VOCAB_SIZE=10
    echo "Using character-level tokenization"
else
    TOKENIZER="zhihan1996/DNABERT-2-117M"
    MODEL_PATH="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/dnamodernbertbase/checkpoints/dna-modernbert-basemod-pretrain-4gpu/ep2-ba52000-rank0.pt"
    VOCAB_SIZE=4096
    echo "Using BPE tokenization"
fi

# Create task-specific output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_PARENT_DIR}/ntv2_${MODEL_TYPE}_${TASK_NAME}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Run evaluation
echo "Running DNA evaluation for task: $TASK_NAME"
echo "Config file: $CONFIG_FILE"
echo "Model type: $MODEL_TYPE"
echo "Model checkpoint: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Additional arguments: $ADDITIONAL_ARGS"

# Run evaluation for single task
python dna_eval.py \
    $CONFIG_FILE \
    tokenizer_name=$TOKENIZER \
    pretrained_checkpoint=$MODEL_PATH \
    vocab_size=$VOCAB_SIZE \
    save_folder=$OUTPUT_DIR \
    output_dir=$OUTPUT_DIR \
    eval_tasks=[$TASK_NAME] \
    $ADDITIONAL_ARGS 2>&1 | tee $OUTPUT_DIR/${TASK_NAME}_eval_log.txt

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"