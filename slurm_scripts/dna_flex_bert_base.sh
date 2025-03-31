#!/bin/bash
#SBATCH --job-name=dna-flex-bert
#SBATCH --output=/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/slurm_scripts/dna-bert_%j.out
#SBATCH --error=/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/slurm_scripts/dna-bert_%j.err
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leann.lindsey@utah.edu

source activate bert24_2

# Change to your working directory
cd /gpfs/gsfs12/users/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT

# Run the training script
composer main.py yamls/main/flex-bert-rope-base.yaml
