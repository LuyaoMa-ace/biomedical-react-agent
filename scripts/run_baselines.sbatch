#!/bin/bash
#SBATCH --job-name=base
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/base_%j.out
#SBATCH --gres=gpu:1

. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
python 06_baselines.py --mode none ${EXTRA}
python 06_baselines.py --mode vanilla_rag ${EXTRA}
