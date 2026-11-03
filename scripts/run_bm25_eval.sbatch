#!/bin/bash
#SBATCH --job-name=bm25_dense
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/luyao.ma/project/logs/bm25_dense_%j.out
#SBATCH --gres=gpu:1

# jina top-100 PMIDs for both question samples (notebook section 3b); BM25 and the hybrid are then computed on a CPU by 17_bm25_eval.py --hybrid
. /scratch.hpc/luyao.ma/venv/bin/activate
export HF_HOME=/scratch.hpc/luyao.ma/hf_cache
export XDG_CACHE_HOME=/scratch.hpc/luyao.ma/.cache
cd /scratch.hpc/luyao.ma/project/scripts
hostname; lscpu | grep -m1 "Model name"
python -X faulthandler 17b_dense_top100.py --data_dir ../data --out ../results/bm25
