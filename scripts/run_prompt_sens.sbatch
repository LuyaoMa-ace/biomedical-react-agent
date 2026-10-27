#!/bin/bash
#SBATCH --job-name=psens
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/psens_%j.out
#SBATCH --gres=gpu:1

# Prompt-sensitivity check: Vanilla RAG and Baseline 0 under three reworded answer prompts (v4 paraphrase, v5 no examples, v6 minimal).
. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
for P in v4 v5 v6; do
  python 06_baselines.py --mode vanilla_rag --prompt $P
  python 06_baselines.py --mode none --prompt $P
done
