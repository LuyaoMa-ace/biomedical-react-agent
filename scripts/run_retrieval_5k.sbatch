#!/bin/bash
#SBATCH --job-name=retr5k
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx2080
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/retr5k_%j.out
#SBATCH --gres=gpu:1

. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
python 03_retrieval_eval.py --models pubmedbert jina bge-m3 \
    --distractors ../data/corpus_distractors.jsonl --tag _5k
