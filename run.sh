#!/bin/bash
#SBATCH --job-name=llama79
#SBATCH --partition=gpu_h100
#SBATCH --time=00-8:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=20G
#SBATCH --output=./slurmout/%x.out
#SBATCH --error=./slurmout/%x.err

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
python lib/rm_prompting.py