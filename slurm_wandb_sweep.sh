#!/bin/bash
#SBATCH --job-name=wandb_sweep_dna
#SBATCH --output=logs/wandb_sweep_%A_%a.out
#SBATCH --error=logs/wandb_sweep_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-20  # Number of hyperparameter trials to run

# Load required modules
module load cuda/11.8
module load python/3.9

# Activate conda environment
source activate modernbert_dna

# Parse arguments
SWEEP_ID=$1
BENCHMARK=$2
TASK=$3
MODEL_TYPE=$4
PRETRAINED_CHECKPOINT=$5

if [ -z "$SWEEP_ID" ] || [ -z "$BENCHMARK" ] || [ -z "$TASK" ] || [ -z "$MODEL_TYPE" ] || [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <sweep_id> <benchmark> <task> <model_type> <pretrained_checkpoint>"
    echo "  sweep_id: WandB sweep ID (create with create_wandb_sweep.sh first)"
    echo "  benchmark: ntv2, gue, or gb"
    echo "  task: specific task name (e.g., enhancers, H3K4me3)"
    echo "  model_type: bpe or char"
    echo "  pretrained_checkpoint: path to pretrained model"
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
        echo "Invalid benchmark: $BENCHMARK"
        exit 1
        ;;
esac

# Set tokenizer and vocab size based on model type
if [ "$MODEL_TYPE" == "bpe" ]; then
    TOKENIZER_NAME="zhihan1996/DNABERT-2-117M"
    VOCAB_SIZE=4096
elif [ "$MODEL_TYPE" == "char" ]; then
    TOKENIZER_NAME="./char_tokenizer"
    VOCAB_SIZE=10
else
    echo "Invalid model type: $MODEL_TYPE"
    exit 1
fi

# Create output directory
OUTPUT_PARENT_DIR="./outputs/wandb_sweeps/${SWEEP_ID}/${BENCHMARK}_${TASK}_${MODEL_TYPE}"
mkdir -p $OUTPUT_PARENT_DIR

echo "Starting sweep agent for array job $SLURM_ARRAY_TASK_ID"
echo "Sweep ID: $SWEEP_ID"
echo "Task: $BENCHMARK / $TASK"
echo "Model type: $MODEL_TYPE"

# Run single sweep trial per SLURM job as recommended by WandB docs
wandb agent $SWEEP_ID \
    --count 1 \
    --project modernBERT-DNA-sweeps \
    --entity leannmlindsey \
    -- python run_wandb_sweep_dna.py \
    --config_file $CONFIG_FILE \
    --task $TASK \
    --model_type $MODEL_TYPE \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --tokenizer_name "$TOKENIZER_NAME" \
    --vocab_size $VOCAB_SIZE \
    --output_parent_dir $OUTPUT_PARENT_DIR

echo "Agent for array job $SLURM_ARRAY_TASK_ID completed"