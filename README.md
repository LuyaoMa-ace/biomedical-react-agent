# Multi-tool ReAct agent for biomedical question answering (PaperSearchQA)

Course project for *Big Data Analytics and Text Mining* (University of Bologna).
Author: Luyao Ma — project proposal by Lorenzo Molfetta.

## 1. What this project does

We study whether a **dynamic, tool-using agent** (LangGraph, ReAct) beats **fixed retrieval pipelines** on
PaperSearchQA, a biomedical factoid QA benchmark built on PubMed abstracts.

* **Data**: `jmhb/PaperSearchQA` (train split), stratified sample of **500 questions** (seed 42, one question per
  PMID, at least 20 per category, categories 0–10). Local corpus = the 500 gold abstracts + **4,500 distractor
  abstracts** (also from the dataset), fetched with `metapub` (NCBI E-utilities).
* **Five tools**: `retrieve_from_index` (FAISS + jina-embeddings-v5), `query_reformulator`,
  `generate_reasoning_query`, `fetch_pubmed_live` (PubMed), `query_openfda` (OpenFDA drug labels).
* **Systems**: Baseline 0 (no tools); four fixed chains (Vanilla RAG, reformulate→retrieve, reasoning→retrieve,
  FDA-augmented RAG); the dynamic ReAct agent with different schedulers (MedGemma-4B, Granite-4.0-micro) and
  ablations (full, index-only, one tool removed, + guardrail).
* **Models**: `google/medgemma-1.5-4b-it` (answers, LLM tools, ReAct scheduler);
  `ibm-granite/granite-4.0-micro` (alternative scheduler); embeddings compared:
  `jinaai/jina-embeddings-v5-text-small`, `NeuML/pubmedbert-base-embeddings`, `BAAI/bge-m3`.
* **Design choice**: the *scheduler* only decides which tool to call and when to stop; the **final answer is
  always written by the same answer generator** (MedGemma + one fixed prompt, `prompts.py`) from the evidence the
  agent collected. Every system therefore shares one answer generator, and comparisons isolate the retrieval policy.

## 2. Main results

500 questions, answer prompt v3, EM = exact match (full tables in `results/exp_*.md`; refresh after the ablation runs).

| System | EM | Contains | Tools/q | Steps | Latency (s/q) | Recall@1 | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| Baseline 0 (no tools) | 0.296 | 0.310 | 0 | 1 | 0.09 | – | – | – |
| Vanilla RAG | 0.764 | 0.820 | 1.00 | 1 | 0.19 | 0.830 | 0.968 | 0.904 |
| Reformulate → retrieve | 0.766 | 0.822 | 2.00 | 2 | 0.29 | 0.840 | 0.968 | 0.908 |
| Reasoning → retrieve | 0.736 | 0.804 | 2.00 | 2 | 0.61 | 0.820 | 0.960 | 0.892 |
| FDA-augmented RAG | 0.760 | 0.816 | 2.00 | 2 | 0.54 | 0.830 | 0.968 | 0.904 |
| Agent: index only (MedGemma) | 0.752 | 0.802 | 1.26 | 3.05 | 3.21 | 0.832 | 0.968 | 0.904 |
| Agent: full (MedGemma) | 0.684 | 0.730 | 1.29 | 3.21 | 3.48 | 0.716 | 0.832 | 0.778 |
| Agent: full + guardrail (MedGemma) | 0.742 | 0.796 | 1.44 | 3.21 | 3.41 | 0.832 | 0.968 | 0.905 |
| Agent: full (Granite) | 0.550 | 0.580 | 2.06 | 3.09 | 2.93 | 0.448 | 0.526 | 0.486 |
| Agent: full + guardrail (Granite) | 0.700 | 0.750 | 2.38 | 3.07 | 2.97 | 0.698 | 0.830 | 0.767 |
| Agent: full + local-index guardrail (Granite, post hoc) | 0.744 | 0.786 | 2.51 | 3.07 | 3.00 | 0.796 | 0.946 | 0.874 |

**Findings.**
1. *Retrieval is what matters*: EM 0.30 → 0.76 (paired +0.47, CI excludes 0). Reformulation, reasoning queries and FDA labels do not improve on plain retrieval (best oracle routing over the four chains: 0.798, +3.2 points).
2. *Dynamic orchestration does not beat fixed pipelines.* The unmodified full agent is significantly worse than every chain (−0.05 to −0.08). Cause (trace analysis): the rewriting tools return a *query*, not documents; small schedulers treat the call as "searched" and stop with no evidence (MedGemma: 62 questions, EM 0.31; Granite: 133 questions, EM 0.23).
3. *A guardrail that retrieves once when no document evidence exists* (tool semantics unchanged; effect predicted before the run) lifts MedGemma by +0.058 [0.032, 0.084] to 0.742 and Granite to 0.700 (0.744 with the post-hoc local-index variant). Both then sit about 2 points below Vanilla RAG, not significantly (p = 0.10 and 0.16), at 15–18× the latency.
4. *Ablations under the guardrail* (each tool removed in turn): no single tool changes EM by more than 0.014 (all CIs include 0); OpenFDA is never called. Removing the reasoning tool or the reformulator from Granite keeps EM (0.746 / 0.752 vs 0.744) at 40–45% lower latency.
5. *Trace analysis*: the number of reasoning steps is unrelated to retrieval quality (Spearman −0.03, n.s.); most remaining errors are reading errors (~20%), not scheduling errors.
6. *Robustness*: answer-prompt wording moves EM by up to 7 points (see Methodological notes), but the ranking of the fixed chains and the "no chain beats Vanilla" conclusion hold under a second prompt.

## 3. Repository layout

```
README.md                     this file
notebook/                     final notebook (step-by-step, reads results/)
scripts/                      all code (numbered in execution order) and SLURM scripts
  common.py qa_utils.py prompts.py llm.py tools.py agent.py        shared modules
  01_sample_data.py 02_fetch_abstracts.py 03_retrieval_eval.py     data + retrieval comparison
  04_sample_distractors.py 05_test_medgemma.py                     distractors, model smoke test
  06_baselines.py                                                   Baseline 0 + four fixed chains
  07_rescore.py 08_stats.py                                         re-scoring, paired statistics
  09_run_agent.py 10_agent_diag.py                                  the agent and its diagnostics
  11_exp_tables.py 12_trace_analysis.py 13_paraphrase_split.py      experiments 1-3 + extra analyses
  14_prompt_sensitivity.py                                          answer-prompt wording check
  15_sample_holdout.py run_holdout.sh                               held-out validation set (500 new questions)
  16_coverage.py run_coverage.sh run_coverage_chains.sbatch         coverage experiment (gold papers withheld from the index) and its analysis
  17_bm25_eval.py 17b_dense_top100.py run_bm25_eval.sbatch          BM25 and BM25 + dense hybrid retrieval baselines (retrieval metrics only)
  run_*.sbatch                                                      SLURM job scripts (L40 / RTX 2080 Ti)
data/    sample.jsonl corpus.jsonl corpus_distractors.jsonl distractor_pmids.jsonl
results/ metrics (json/csv), figures (png), tables (md), analysis logs
traces/  per-question predictions and full ReAct traces (gzip)
cache/   cached OpenFDA / PubMed responses (makes the live tools reproducible)
env/     requirements.txt (pip freeze of the environment used)
```

## 4. Environment

* Python 3.13, PyTorch 2.11 (CUDA 12.8 wheels), transformers 5.x, sentence-transformers, faiss-cpu, langgraph,
  langchain-core, metapub, datasets, rouge-score, peft, pandas, scipy, matplotlib (`env/requirements.txt`).
* Compute: DISI GPU cluster (SLURM). MedGemma runs in **bfloat16**; fp16 overflows (empty outputs).
  Partition `l40` (48 GB) for everything; `rtx2080` (11 GB) works for MedGemma alone (bf16, retriever on CPU).
* Credentials (never stored in the repo): `NCBI_API_KEY` (environment variable) for PubMed;
  a Hugging Face token with access to the gated MedGemma model.

## 5. How to reproduce (all commands run from `scripts/`)

```bash
# 0. environment (once)
python -m venv venv && . venv/bin/activate && pip install -r ../env/requirements.txt
export HF_HOME=/path/to/hf_cache NCBI_API_KEY=...           # + `hf auth login` for MedGemma

# 1. data
python 01_sample_data.py --n 500 --seed 42 --min_per_cat 20          # -> data/sample.jsonl
python 02_fetch_abstracts.py                                         # -> data/corpus.jsonl (needs NCBI key)
python 04_sample_distractors.py --n 4500
python 02_fetch_abstracts.py --sample ../data/distractor_pmids.jsonl --out ../data/corpus_distractors.jsonl

# 2. retrieval: 3 embedding models x (500-doc, 5000-doc index); index files go to ../index
sbatch run_retrieval.sbatch && sbatch run_retrieval_5k.sbatch

# 3. baselines (Baseline 0 + four fixed chains), answer prompt v3, 500 questions
sbatch run_baselines.sbatch && sbatch run_static_chains.sbatch

# 4. dynamic agent (experiment 1/2/3 runs); CONFIG in {full,no_fda,no_pubmed,no_reasoning,no_reformulator,index_only}
sbatch --export=ALL,CONFIG=full run_agent.sbatch
sbatch --export=ALL,CONFIG=full,EXTRA="--orchestrator granite" run_agent.sbatch
sbatch --export=ALL,CONFIG=full,EXTRA="--guardrail" run_agent.sbatch
# the remaining runs of the report are submitted by two wrappers (read them for the exact flags):
bash run_ablations.sh      # one tool removed at a time, with and without the guardrail; Granite local-index variants
bash run_holdout.sh        # all frozen systems on the 500 held-out questions (after python 15_sample_holdout.py)

# 5. analysis (CPU)
python 11_exp_tables.py            # ladder, experiment 1 (ablations), experiment 2 (vs fixed chains)
python 12_trace_analysis.py --agent agent_full --static baseline_vanilla_rag_v3     # experiment 3
python 08_stats.py --systems none vanilla_rag reformulate_rag reasoning_rag fda_rag --ref vanilla_rag
python 13_paraphrase_split.py
# held-out validation: 500 NEW questions (their abstracts are already in the 5k index), all systems frozen
python 15_sample_holdout.py && bash run_holdout.sh
python 11_exp_tables.py --tag _holdout --out ../results/holdout
python 08_stats.py --systems none vanilla_rag reformulate_rag reasoning_rag fda_rag --ref vanilla_rag --suffix _v3_holdout --out stats_baselines_holdout

# 6. coverage experiment (frozen systems, 500 held-out questions; needs NCBI_API_KEY for live PubMed)
bash run_coverage.sh smoke   # 16 questions, a few minutes
bash run_coverage.sh         # 8 jobs
python 16_coverage.py        # -> ../results/coverage/

# 7. classical retrieval baselines (retrieval metrics only; BM25 needs no GPU)
sbatch run_bm25_eval.sbatch  # jina top-100 per question -> ../results/bm25/dense_top100_*.json (needs the FAISS index from step 2)
python 17_bm25_eval.py --hybrid --data_dir ../data --out ../results/bm25   # BM25, jina and hybrid -> retrieval_bm25.csv
# prompt sensitivity (3 reworded answer prompts x {Vanilla RAG, Baseline 0})
sbatch run_prompt_sens.sbatch && python 14_prompt_sensitivity.py
```

Every run is **resumable** (finished question ids are skipped) and writes `traces/<name>.jsonl` + `results/<name>.json`.
Generation is greedy (`do_sample=False`), so reruns are deterministic up to GPU numerics.

## 6. Methodological notes

* **Metrics**: Recall@1/5/10, MRR, MAP (= MRR: one relevant paper per question), nDCG@k; exact match (SQuAD-style
  normalisation, any gold synonym), ROUGE-1/L, and a looser *Contains*; tool calls, reasoning steps, latency
  (batch time / batch size, i.e. amortised). Paired bootstrap CIs and McNemar tests for system comparisons.
* **Answer format**: prompt v3 pre-fills the model turn with `A:` (otherwise MedGemma sometimes writes a
  chain of thought instead of the answer, costing ~7 EM points). See `prompts.py`, `qa_utils.py`.
* **Honest caveats**: the answer prompt and the agent prompt were adjusted after reading failures on this same
  500-question set (no separate dev set); the guardrail is a post-hoc fix whose predicted effect was written
  down before running. Absolute numbers may be slightly optimistic; comparisons between systems are fair because
  all systems share the same answer generator.

### Limitations

* **Prompts are not optimised and not tuned on held-out data.** Answer and scheduler prompts were iterated by hand
  (v1→v3, agent v1→v3) while reading failures on the same 500 questions; there is no separate dev set, and no
  systematic prompt search was run. We only claim that prompts are *shared and fixed* across systems, not optimal.
  `14_prompt_sensitivity.py` re-runs Vanilla RAG and Baseline 0 with three reworded answer prompts to measure how
  much the wording matters (`results/prompt_sensitivity.md`): paraphrasing the instruction (v4) changes nothing
  significant (+0.012 EM), but dropping the format examples costs 4.6–6.0 EM points (Contains moves far less, so the
  loss is answer form, not knowledge). v3 is therefore on a plateau, not proven optimal; read absolute EM with about
  ±5 points of prompt uncertainty. Re-running the four fixed chains under v4 keeps the picture: Vanilla 0.776,
  reformulate 0.778, FDA 0.776, reasoning 0.764 (no chain beats Vanilla; oracle headroom +0.030), but the
  reasoning-chain penalty seen under v3 (−0.028, p=0.02) is no longer significant (−0.012, p=0.33).
* **The scheduler prompt was developed with MedGemma.** Granite-4.0-micro received the same text-format ReAct prompt
  (no native function calling, no prompt of its own), so "Granite < MedGemma" may partly reflect prompt fit.
* **Zero-shot small schedulers only**; no fine-tuning, no few-shot demonstrations of tool use, no larger LLM.
* **500 questions**: paired differences below roughly ±2–3 EM points are not detectable.
* **Closed 5,000-paper corpus** (paper: 16M abstracts), so absolute accuracies are not comparable to the paper.
* **EM underestimates** correct answers with different wording (Contains / ROUGE are reported alongside).
* **The guardrail is a post-hoc fix**; its effect was predicted before running, but it was designed after seeing the failures.

## 7. Notes on the data

Our HuggingFace copy of PaperSearchQA has 11 categories (`cat_num` 0–10, 0 = "Other"); the paper lists ten.
Only the `train` split is used (nothing is trained). The paper's RAG baseline searches 16M abstracts; ours searches
5,000, so absolute accuracies are not comparable.

### Held-out validation (500 new questions)
Systems were frozen and rerun on `data/holdout.jsonl`; all six predictions written beforehand held. Baseline 0 0.248; Vanilla RAG 0.746; fixed chains 0.744-0.746; MedGemma full agent 0.644 (-0.102 vs Vanilla, significant); with guardrail 0.722 (-0.024, p = 0.073); index-only 0.726 (-0.020, n.s.); Granite full 0.518, guarded 0.682, local-index guarded 0.708 (-0.038, significant). No agent configuration beats the fixed chains on either sample. Tables in `results/holdout/`.

### Coverage experiment (gold papers withheld from the index)
The gold paper of a fixed random 0 / 50 / 100 % of the held-out questions is removed from the index (nested sets, `common.withheld_qids`); only live PubMed can still supply it. Fixed chains with the same tools as the agent (`pubmed_rag`, `hybrid_rag`) and an agent without PubMed serve as controls. Predictions were written first (four of five held; the PubMed recovery rate was 8-9 %, just below the predicted 10-40 %).
Vanilla RAG 0.746 / 0.488 / 0.220; retrieve + PubMed chain 0.730 / 0.528 / 0.338; agent + guardrail 0.722 / 0.490 / 0.240; agent without PubMed 0.730 / 0.494 / 0.240. The agent calls PubMed on 10.8 % of the withheld and 3.6 % of the retained questions, so it adds nothing over the same agent without PubMed, and the always-PubMed chain beats it by 0.038 (50 %) and 0.098 (100 %). Tables and figure in `results/coverage/`.

### BM25 and hybrid retrieval baselines (retrieval only)
Predictions were written first (B2, B4 and B5 held; B1 in part, BM25 Recall@1 was higher than predicted; B3 did not hold). Development / held-out Recall@10: BM25 0.922 / 0.920, `jina` 0.968 / 0.956, equal-weight RRF hybrid 0.954 / 0.958. The dense retriever's advantage lies in the paraphrased questions (Recall@1 on original vs paraphrased: BM25 0.873 / 0.690, `jina` 0.825 / 0.834), and the hybrid does not reliably beat `jina`. BM25 builds in about a second with a 5.7 MB index. We did not run the answer chains or the agent with these retrievers (stopped at the retrieval table for time). Note: on the cluster's generic virtual CPU, FAISS crashes with `Illegal instruction` when more than about 20 queries are searched in one call, so `17b_dense_top100.py` searches in batches of 16 (as `06_baselines.py` does). Tables in `results/bm25/`.
