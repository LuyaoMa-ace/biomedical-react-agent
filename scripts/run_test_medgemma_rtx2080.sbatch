#!/bin/bash
#SBATCH --job-name=mg2080
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx2080
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/mg2080_%j.out
#SBATCH --gres=gpu:1

. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
python 05_test_medgemma.py --dtype float16 --n 8
