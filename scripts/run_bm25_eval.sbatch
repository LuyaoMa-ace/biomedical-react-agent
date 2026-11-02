#!/bin/bash
#SBATCH --job-name=bm25_eval
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/bm25_eval_%j.out
#SBATCH --gres=gpu:1

# BM25 + jina + hybrid retrieval metrics on both question samples (notebook section 3b); needs ../index (FAISS) from 03_retrieval_eval.py
. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
python 17_bm25_eval.py --hybrid --data_dir ../data --out ../results/bm25 --tag giano_cluster
