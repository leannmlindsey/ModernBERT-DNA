#!/bin/bash

# Script to run DNA model evaluation on NTv2 benchmarks
# Usage: ./run_dna_eval.sh <model_type> [additional_args]
# Example: ./run_dna_eval.sh bpe
# Example: ./run_dna_eval.sh char --skip_training true

set -e  # Exit on error

# Parse command line arguments
MODEL_TYPE=${1:-bpe}  # Default to BPE tokenization
shift 1
ADDITIONAL_ARGS="$@"

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

# Create output directory
OUTPUT_DIR="outputs/dna_eval/ntv2_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Run evaluation
echo "Running DNA evaluation on NTv2 benchmarks"
echo "Config file: $CONFIG_FILE"
echo "Model type: $MODEL_TYPE"
echo "Model checkpoint: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Additional arguments: $ADDITIONAL_ARGS"

# Run evaluation
python dna_eval.py \
    $CONFIG_FILE \
    tokenizer_name=$TOKENIZER \
    pretrained_checkpoint=$MODEL_PATH \
    vocab_size=$VOCAB_SIZE \
    save_folder=$OUTPUT_DIR \
    output_dir=$OUTPUT_DIR \
    $ADDITIONAL_ARGS 2>&1 | tee $OUTPUT_DIR/eval_log.txt

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"