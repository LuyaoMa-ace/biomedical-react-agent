#!/bin/bash
# Coverage experiment on the 500 held-out questions (systems frozen, predictions in README / PROGRESS written beforehand).
#   1. smoke test first (a few minutes):  bash run_coverage.sh smoke
#   2. full run:                          bash run_coverage.sh
# Level 0 re-uses the existing *_holdout traces for vanilla / reformulate / agent_full_guard; only the two live-PubMed chains
# and the no-PubMed control agent are new there. Levels 50 and 100 rerun everything that touches the index.
cd /scratch.hpc/luyao.ma/project/scripts
H="--sample ../data/holdout.jsonl"
if [ "$1" = "smoke" ]; then
  S="--n 16"
  sbatch --job-name=cs_chain --export=ALL,MODES="vanilla_rag pubmed_rag hybrid_rag",EXTRA="$H --tag _smoke_w100 --withhold 1.0 $S" run_coverage_chains.sbatch
  sbatch --job-name=cs_agent --export=ALL,CONFIG=full,EXTRA="$H --tag _smoke_w50 --withhold 0.5 --guardrail $S" run_agent.sbatch
  exit 0
fi
# --- level 0: new systems only
sbatch --job-name=c0_chains --export=ALL,MODES="pubmed_rag hybrid_rag",EXTRA="$H --tag _holdout" run_coverage_chains.sbatch
sbatch --job-name=c0_nopub  --export=ALL,CONFIG=no_pubmed,EXTRA="$H --tag _holdout --guardrail" run_agent.sbatch
# --- levels 50 and 100: everything that retrieves from the index
for L in 50 100; do
  F=$(python3 -c "print($L/100)")
  sbatch --job-name=c${L}_chains --export=ALL,MODES="vanilla_rag reformulate_rag pubmed_rag hybrid_rag",EXTRA="$H --tag _holdout_w$L --withhold $F" run_coverage_chains.sbatch
  sbatch --job-name=c${L}_full   --export=ALL,CONFIG=full,EXTRA="$H --tag _holdout_w$L --withhold $F --guardrail" run_agent.sbatch
  sbatch --job-name=c${L}_nopub  --export=ALL,CONFIG=no_pubmed,EXTRA="$H --tag _holdout_w$L --withhold $F --guardrail" run_agent.sbatch
done
