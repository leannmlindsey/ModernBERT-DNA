#!/bin/bash

# Script to run Nucleotide Transformer v2 (NT v2) fine-tuning tasks
# Usage: ./run_ntv2_finetuning.sh <task_name> <model_type> [additional_args]
# Example: ./run_ntv2_finetuning.sh H3K27ac bpe
# Example: ./run_ntv2_finetuning.sh enhancers char --seed 42

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
    echo "Error: Invalid NT v2 task name '${TASK_NAME}'"
    echo "Valid NT v2 tasks are: ${VALID_TASKS[@]}"
    exit 1
fi

# Determine which config to use based on task category
if [[ $TASK_NAME == H* ]] || [[ $TASK_NAME == "H4K20me1" ]]; then
    TASK_CONFIG="yamls/dna_finetuning/ntv2/histone_modifications.yaml"
elif [[ $TASK_NAME == enhancer* ]]; then
    TASK_CONFIG="yamls/dna_finetuning/ntv2/enhancers.yaml"
elif [[ $TASK_NAME == promoter* ]]; then
    TASK_CONFIG="yamls/dna_finetuning/ntv2/promoters.yaml"
elif [[ $TASK_NAME == splice_sites* ]]; then
    TASK_CONFIG="yamls/dna_finetuning/ntv2/splice_sites.yaml"
fi

# Select base config and model path based on model type
if [ "$MODEL_TYPE" == "char" ]; then
    BASE_CONFIG="yamls/dna_finetuning/ntv2/ntv2_base.yaml"
    TOKENIZER="dna_char"
    MODEL_PATH="./checkpoints/modernbert-dna-base-char/checkpoint.pt"
    VOCAB_SIZE=10
    echo "Using character-level tokenization"
else
    BASE_CONFIG="yamls/dna_finetuning/ntv2/ntv2_base.yaml"
    TOKENIZER="zhihan1996/DNABERT-2-117M"
    MODEL_PATH="./checkpoints/modernbert-dna-base-bpe/checkpoint.pt"
    VOCAB_SIZE=4096
    echo "Using BPE tokenization"
fi

# Create output directory
OUTPUT_DIR="outputs/ntv2/${TASK_NAME}_${MODEL_TYPE}"
mkdir -p $OUTPUT_DIR

# Run training
echo "Running NT v2 fine-tuning for task: $TASK_NAME"
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
    tokenizer_name=$TOKENIZER \
    model.tokenizer_name=$TOKENIZER \
    model.pretrained_model_name=$TOKENIZER \
    model.pretrained_checkpoint=$MODEL_PATH \
    model.model_config.vocab_size=$VOCAB_SIZE \
    save_folder=$OUTPUT_DIR/checkpoints \
    run_name=ntv2_${TASK_NAME}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S) \
    $ADDITIONAL_ARGS