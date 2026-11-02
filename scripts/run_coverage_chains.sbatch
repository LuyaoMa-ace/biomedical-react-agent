#!/bin/bash
#SBATCH --job-name=cov_chains
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/cov_chains_%j.out
#SBATCH --gres=gpu:1

# usage: sbatch --export=ALL,MODES="vanilla_rag pubmed_rag",EXTRA="--sample ../data/holdout.jsonl --tag _holdout_w50 --withhold 0.5" run_coverage_chains.sbatch
. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
for mode in ${MODES}; do
    python 06_baselines.py --mode $mode ${EXTRA}
done
