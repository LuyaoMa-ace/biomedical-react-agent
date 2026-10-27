#!/bin/bash
# Submits the remaining ablation runs (all with the guardrail; Granite with the local-index variant).
# Usage on giano:  bash run_ablations.sh      (each run is resumable; re-submitting continues unfinished ones)
cd /scratch.hpc/luyao.ma/project/scripts
for C in no_fda no_pubmed no_reasoning no_reformulator; do
  sbatch --job-name=abl_m_$C --export=ALL,CONFIG=$C,EXTRA="--guardrail" run_agent.sbatch
done
for C in no_reformulator no_reasoning no_pubmed; do
  sbatch --job-name=abl_g_$C --export=ALL,CONFIG=$C,EXTRA="--orchestrator granite --guardrail_local" run_agent.sbatch
done
