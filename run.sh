#!/bin/bash
#SBATCH --job-name=gemini
#SBATCH --partition=gpu_h100
#SBATCH --time=00-8:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=20G
#SBATCH --output=./slurmout/%x.out
#SBATCH --error=./slurmout/%x.err

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
# rm -rf ~/.cache/pip
# rm -rf ~/.cache/huggingface
#conda clean --all -y
python lib/rm_prompting.py