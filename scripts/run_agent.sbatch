#!/bin/bash
#SBATCH --job-name=agent
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/agent_%j.out
#SBATCH --gres=gpu:1

# usage: sbatch --export=ALL,CONFIG=full,EXTRA="--n 32 --tag _smoke" run_agent.sbatch
. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
python 09_run_agent.py --config ${CONFIG:-full} ${EXTRA}
