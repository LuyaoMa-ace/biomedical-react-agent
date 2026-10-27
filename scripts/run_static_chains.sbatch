#!/bin/bash
#SBATCH --job-name=chains
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/chains_%j.out
#SBATCH --gres=gpu:1

. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
for mode in reformulate_rag reasoning_rag fda_rag; do
    python 06_baselines.py --mode $mode ${EXTRA}
done
