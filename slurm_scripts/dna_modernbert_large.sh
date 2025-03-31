#!/bin/bash
#SBATCH --job-name=dna-large
#SBATCH --output=/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/slurm_scripts/dna-large_%j.out
#SBATCH --error=/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/slurm_scripts/dna-large_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leann.lindsey@utah.edu

source activate bert24_2
export WANDB_API_KEY='4231f30cf28322633fb26bdd3b9992cd0a9ce62d'
# Change to your working directory
cd /gpfs/gsfs12/users/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT

# Run the training script
composer main.py yamls/pretrain/modernbert-large-pretrain.yaml
