#!/bin/bash

# Launch WandB sweep for DNA model finetuning
# Usage: ./launch_wandb_sweep.sh <benchmark> <task> <model_type> <pretrained_checkpoint> <num_agents>

set -e

# Parse arguments
BENCHMARK=$1
TASK=$2
MODEL_TYPE=$3  # bpe or char
PRETRAINED_CHECKPOINT=$4
NUM_AGENTS=${5:-1}  # Default to 1 agent

if [ -z "$BENCHMARK" ] || [ -z "$TASK" ] || [ -z "$MODEL_TYPE" ] || [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "Usage: $0 <benchmark> <task> <model_type> <pretrained_checkpoint> [num_agents]"
    echo "  benchmark: ntv2, gue, or gb"
    echo "  task: specific task name (e.g., enhancers, H3K4me3)"
    echo "  model_type: bpe or char"
    echo "  pretrained_checkpoint: path to pretrained model"
    echo "  num_agents: number of parallel sweep agents (default: 1)"
    exit 1
fi

# Set config file based on benchmark
case $BENCHMARK in
    ntv2)
        CONFIG_FILE="yamls/dna_eval_ntv2.yaml"
        ;;
    gue)
        CONFIG_FILE="yamls/dna_eval_gue.yaml"
        ;;
    gb)
        CONFIG_FILE="yamls/dna_eval_gb.yaml"
        ;;
    *)
        echo "Invalid benchmark: $BENCHMARK (must be ntv2, gue, or gb)"
        exit 1
        ;;
esac

# Set tokenizer and vocab size based on model type
if [ "$MODEL_TYPE" == "bpe" ]; then
    TOKENIZER_NAME="zhihan1996/DNABERT-2-117M"
    VOCAB_SIZE=4096
elif [ "$MODEL_TYPE" == "char" ]; then
    TOKENIZER_NAME="./char_tokenizer_4kmer"
    VOCAB_SIZE=4101
else
    echo "Invalid model type: $MODEL_TYPE (must be bpe or char)"
    exit 1
fi

# Create output directory
OUTPUT_PARENT_DIR="./outputs/wandb_sweeps/${BENCHMARK}_${TASK}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_PARENT_DIR

# Create sweep
echo "Creating WandB sweep for $BENCHMARK $TASK with $MODEL_TYPE model..."
SWEEP_ID=$(wandb sweep wandb_sweep_dna_finetuning.yaml --project modernBERT-DNA-sweeps --entity leannmlindsey 2>&1 | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create sweep"
    exit 1
fi

echo "Created sweep: $SWEEP_ID"

# Launch agents
echo "Launching $NUM_AGENTS sweep agents..."
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i..."
    wandb agent $SWEEP_ID \
        --count 20 \
        --project modernBERT-DNA-sweeps \
        --entity leannmlindsey \
        -- python run_wandb_sweep_dna.py \
        --config_file $CONFIG_FILE \
        --task $TASK \
        --model_type $MODEL_TYPE \
        --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
        --tokenizer_name "$TOKENIZER_NAME" \
        --vocab_size $VOCAB_SIZE \
        --output_parent_dir $OUTPUT_PARENT_DIR &
    
    sleep 5  # Small delay between agent launches
done

echo "Sweep launched with $NUM_AGENTS agents"
echo "Monitor progress at: https://wandb.ai/leannmlindsey/modernBERT-DNA-sweeps/sweeps/$SWEEP_ID"
echo "Output directory: $OUTPUT_PARENT_DIR"

# Wait for all agents to complete
wait

echo "All agents completed"