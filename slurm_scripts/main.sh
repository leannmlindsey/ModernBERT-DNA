#!/bin/bash
#SBATCH --job-name=modernbert-base    # Job name
#SBATCH --output=%x-%j.out            # Standard output log (%x is job name, %j is job ID)
#SBATCH --error=%x-%j.err             # Standard error log
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # CPU cores per task
#SBATCH --mem=0                       # Memory per node
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu               # Request GPU partition
#SBATCH --gres=gpu:a100:4             # Request 4 GPU

source activate bert24 
conda info --envs
conda list

script_dir="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT"
yaml="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/MODELS/MODERNBERT/ModernBERT/yamls/test/main.yaml"

# Run the Composer job
composer $script_dir/main.py $yaml

