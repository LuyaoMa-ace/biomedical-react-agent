#!/bin/bash
# Held-out validation: every frozen system (no further tuning) on 500 NEW questions (data/holdout.jsonl).
# Usage on giano:  python 15_sample_holdout.py && bash run_holdout.sh
# Outputs carry the suffix _holdout (traces/, results/). Analysis commands are at the bottom of README section 5.
cd /scratch.hpc/luyao.ma/project/scripts
H="--sample ../data/holdout.jsonl --tag _holdout"
sbatch --job-name=h_base   --export=ALL,EXTRA="$H" run_baselines.sbatch          # Baseline 0 + Vanilla RAG
sbatch --job-name=h_chains --export=ALL,EXTRA="$H" run_static_chains.sbatch      # reformulate / reasoning / FDA chains
sbatch --job-name=h_full   --export=ALL,CONFIG=full,EXTRA="$H" run_agent.sbatch
sbatch --job-name=h_index  --export=ALL,CONFIG=index_only,EXTRA="$H" run_agent.sbatch
sbatch --job-name=h_guard  --export=ALL,CONFIG=full,EXTRA="$H --guardrail" run_agent.sbatch
sbatch --job-name=h_g_full --export=ALL,CONFIG=full,EXTRA="$H --orchestrator granite" run_agent.sbatch
sbatch --job-name=h_g_grd  --export=ALL,CONFIG=full,EXTRA="$H --orchestrator granite --guardrail" run_agent.sbatch
sbatch --job-name=h_g_loc  --export=ALL,CONFIG=full,EXTRA="$H --orchestrator granite --guardrail_local" run_agent.sbatch
