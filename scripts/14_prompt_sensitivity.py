"""How much does the answer-prompt wording matter? Compares Vanilla RAG / Baseline 0 under prompts v3 (used everywhere) and v4-v6.
Run on giano after run_prompt_sens.sbatch:  python 14_prompt_sensitivity.py
Writes ../results/prompt_sensitivity.{csv,md}. Paired bootstrap CI of each variant vs v3 (same 500 questions)."""
import json, os
import numpy as np, pandas as pd

LABEL = {"v3": "v3 (used in all experiments)", "v4": "v4 paraphrased instruction", "v5": "v5 no format examples", "v6": "v6 minimal instruction"}
rng = np.random.default_rng(0)
rows = []
for mode in ["vanilla_rag", "none"]:
    base = pd.DataFrame([json.loads(l) for l in open(f"../traces/baseline_{mode}_v3.jsonl")]).set_index("qid")
    for v in ["v3", "v4", "v5", "v6"]:
        p = f"../traces/baseline_{mode}_{v}.jsonl"
        if not os.path.exists(p):
            continue
        d = pd.DataFrame([json.loads(l) for l in open(p)]).set_index("qid")
        ids = base.index.intersection(d.index)
        diff = (d.loc[ids, "EM"] - base.loc[ids, "EM"]).to_numpy()
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        rows.append({"system": mode, "prompt": LABEL[v], "n": len(ids), "EM": d.loc[ids, "EM"].mean(),
                     "Contains": d.loc[ids, "Contains"].mean(), "ROUGE-1": d.loc[ids, "ROUGE-1"].mean(),
                     "diff_vs_v3": diff.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
df = pd.DataFrame(rows)
df.to_csv("../results/prompt_sensitivity.csv", index=False)
cols = list(df.columns)
md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
    "| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in df.itertuples(index=False))
open("../results/prompt_sensitivity.md", "w").write(md + "\n")
print(md)
for mode, g in df.groupby("system"):
    print(f"\n{mode}: EM range over prompts = {g.EM.min():.3f} .. {g.EM.max():.3f} (spread {g.EM.max()-g.EM.min():.3f})")
