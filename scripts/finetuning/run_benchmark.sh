#!/bin/bash

# Main entry point for running DNA sequence classification benchmarks
# Usage: ./run_benchmark.sh <benchmark> <task_or_all> <model_type> [additional_args]

set -e

# Parse arguments
BENCHMARK=$1
TASK_OR_ALL=$2
MODEL_TYPE=${3:-bpe}

if [ -z "$BENCHMARK" ] || [ -z "$TASK_OR_ALL" ]; then
    echo "ModernBERT-DNA Fine-tuning Benchmarks"
    echo "====================================="
    echo ""
    echo "Usage: ./run_benchmark.sh <benchmark> <task_or_all> <model_type> [additional_args]"
    echo ""
    echo "Benchmarks:"
    echo "  ntv2  - Nucleotide Transformer v2 (18 tasks)"
    echo "  gb    - Genomic Benchmarks (9 tasks)"
    echo "  gue   - Genome Understanding Evaluation (28+ tasks)"
    echo ""
    echo "Examples:"
    echo "  # Run a single task"
    echo "  ./run_benchmark.sh ntv2 H3K27ac bpe"
    echo "  ./run_benchmark.sh gb human_nontata_promoters char"
    echo "  ./run_benchmark.sh gue human_promoter bpe"
    echo ""
    echo "  # Run all tasks in a benchmark"
    echo "  ./run_benchmark.sh ntv2 all bpe"
    echo "  ./run_benchmark.sh gb all char"
    echo "  ./run_benchmark.sh gue all bpe"
    echo ""
    echo "  # Run GUE subsets"
    echo "  ./run_benchmark.sh gue standard bpe  # Standard GUE tasks only"
    echo "  ./run_benchmark.sh gue long bpe      # GUE+ long-range tasks only"
    echo ""
    exit 1
fi

# Route to appropriate script
case $BENCHMARK in
    "ntv2")
        if [ "$TASK_OR_ALL" == "all" ]; then
            shift 3
            ./run_all_ntv2_tasks.sh $MODEL_TYPE "$@"
        else
            shift 3
            ./run_ntv2_finetuning.sh $TASK_OR_ALL $MODEL_TYPE "$@"
        fi
        ;;
    "gb")
        if [ "$TASK_OR_ALL" == "all" ]; then
            shift 3
            ./run_all_gb_tasks.sh $MODEL_TYPE "$@"
        else
            shift 3
            ./run_gb_finetuning.sh $TASK_OR_ALL $MODEL_TYPE "$@"
        fi
        ;;
    "gue")
        if [ "$TASK_OR_ALL" == "all" ] || [ "$TASK_OR_ALL" == "standard" ] || [ "$TASK_OR_ALL" == "long" ]; then
            shift 3
            ./run_all_gue_tasks.sh $MODEL_TYPE $TASK_OR_ALL "$@"
        else
            shift 3
            ./run_gue_finetuning.sh $TASK_OR_ALL $MODEL_TYPE "$@"
        fi
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo "Valid benchmarks are: ntv2, gb, gue"
        exit 1
        ;;
esac