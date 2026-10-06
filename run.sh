#!/bin/bash
#To use this script you need to be already on the Surf server 
#SBATCH --job-name=trans
#SBATCH --partition=gpu_h100
#SBATCH --time=00-0:20:00
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=20G
#SBATCH --output=./slurmout/%x.out
#SBATCH --error=./slurmout/%x.err

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
# rm -rf ~/.cache/pip
# rm -rf ~/.cache/huggingface
#conda clean --all -y
python lib/translations.py