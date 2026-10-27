"""EM split by `is_paraphrased` (CPU only). PaperSearchQA paraphrases 50% of questions to reduce keyword
overlap; the dataset paper reports paraphrased questions are harder (Search-R1: 57.2 vs 44.9).

Usage: python 13_paraphrase_split.py [system ...]   (default: all systems whose trace file exists)
"""
import json, os, sys
import numpy as np, pandas as pd
from common import load_data

DEFAULT = ["baseline_none_v3", "baseline_vanilla_rag_v3", "baseline_reformulate_rag_v3",
           "baseline_reasoning_rag_v3", "baseline_fda_rag_v3", "agent_full", "agent_index_only",
           "agent_no_fda", "agent_no_pubmed", "agent_no_reasoning", "agent_no_reformulator", "agent_full_granite"]
names = sys.argv[1:] or DEFAULT
sample, _ = load_data()
para = sample.set_index("qid")["is_paraphrased"].astype(bool)
print(f"questions: {len(para)} | paraphrased: {int(para.sum())} ({para.mean():.0%}) | original: {int((~para).sum())}\n")
rng = np.random.default_rng(0)
rows = []
for n in names:
    path = f"../traces/{n}.jsonl"
    if not os.path.exists(path):
        continue
    d = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("qid").set_index("qid")
    d["para"] = para.reindex(d.index)
    o, p = d[~d.para].EM.to_numpy(), d[d.para].EM.to_numpy()
    boot = [rng.choice(o, len(o)).mean() - rng.choice(p, len(p)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    r = {"system": n, "EM_original": o.mean(), "EM_paraphrased": p.mean(), "gap(orig-para)": o.mean() - p.mean(),
         "gap_ci_lo": lo, "gap_ci_hi": hi, "n_orig": len(o), "n_para": len(p)}
    if "gold_rank" in d:
        g = d[d.gold_rank < 10 ** 6]
        if len(g):
            r["R@1_orig"] = (d[~d.para].gold_rank == 1).mean(); r["R@1_para"] = (d[d.para].gold_rank == 1).mean()
    rows.append(r)
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
os.makedirs("../results", exist_ok=True)
out.to_csv("../results/paraphrase_split.csv", index=False)
print("\nsaved ../results/paraphrase_split.csv  (gap CI excluding 0 => paraphrased questions are significantly harder/easier)")
