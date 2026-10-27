"""Tables for the performance ladder and experiments 1 and 2 (CPU only).

Reads ../traces/<system>.jsonl for every system that exists and writes
  ../results/exp_ladder.csv / .md      all systems, absolute metrics + delta vs Baseline 0
  ../results/exp1_ablation.csv         every agent config vs the full agent (paired, bootstrap CI)
  ../results/exp2_vs_static.csv        full agent vs each fixed chain (paired CI, McNemar)
  ../results/exp2_per_category.csv     EM per category: agent vs each fixed chain

Usage: python 11_exp_tables.py [--agent agent_full] [--best_static baseline_vanilla_rag_v3]
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from common import retrieval_metrics

SYSTEMS = [
    ("Baseline 0 (no tools)", "baseline_none_v3"),
    ("Vanilla RAG", "baseline_vanilla_rag_v3"),
    ("Reformulate -> retrieve", "baseline_reformulate_rag_v3"),
    ("Reasoning -> retrieve", "baseline_reasoning_rag_v3"),
    ("FDA-augmented RAG", "baseline_fda_rag_v3"),
    ("Agent: full (5 tools)", "agent_full"),
    ("Agent: no OpenFDA", "agent_no_fda"),
    ("Agent: no live PubMed", "agent_no_pubmed"),
    ("Agent: no reasoning query", "agent_no_reasoning"),
    ("Agent: no reformulator", "agent_no_reformulator"),
    ("Agent: index only", "agent_index_only"),
    ("Agent (Granite scheduler): full", "agent_full_granite"),
    ("Agent: full + guardrail", "agent_full_guard"),
    ("Agent (Granite): full + guardrail", "agent_full_granite_guard"),
    ("Agent (Granite): full + guardrail (local-index variant)", "agent_full_granite_guardlocal"),
    ("Agent: no OpenFDA + guardrail", "agent_no_fda_guard"),
    ("Agent: no live PubMed + guardrail", "agent_no_pubmed_guard"),
    ("Agent: no reasoning query + guardrail", "agent_no_reasoning_guard"),
    ("Agent: no reformulator + guardrail", "agent_no_reformulator_guard"),
    ("Agent: index only + guardrail", "agent_index_only_guard"),
]
ANS = ["EM", "Contains", "ROUGE-1", "ROUGE-L"]
EFF = ["tool_calls", "steps", "latency_s"]

p = argparse.ArgumentParser()
p.add_argument("--agent", default="agent_full", help="the dynamic system compared in experiment 2")
p.add_argument("--best_static", default="baseline_vanilla_rag_v3")
p.add_argument("--traces", default="../traces")
p.add_argument("--out", default="../results")
p.add_argument("--n_boot", type=int, default=10000)
args = p.parse_args()
rng = np.random.default_rng(0)

data, labels = {}, {}
for label, key in SYSTEMS:
    path = f"{args.traces}/{key}.jsonl"
    if os.path.exists(path):
        d = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("qid").set_index("qid").sort_index()
        data[key], labels[key] = d, label
    else:
        print(f"[skip] {key}: no trace file")
common = sorted(set.intersection(*[set(d.index) for d in data.values()]))
data = {k: d.loc[common] for k, d in data.items()}
n = len(common)
idx = rng.integers(0, n, size=(args.n_boot, n))       # shared resamples -> paired CIs
print(f"{len(data)} systems, {n} common questions\n")
os.makedirs(args.out, exist_ok=True)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5]); return float(lo), float(hi)


# ---------------- ladder: absolute metrics + delta vs Baseline 0 ----------------
base = data.get("baseline_none_v3")
rows = []
for k, d in data.items():
    r = {"system": labels[k]}
    for m in ANS:
        r[m] = d[m].mean()
        if base is not None:
            r["Δ" + m] = d[m].mean() - base[m].mean()
    for m in EFF:
        r[m] = d[m].mean()
    if "gold_rank" in d and (d.gold_rank < 10 ** 6).any():
        rm = retrieval_metrics(d.gold_rank.to_numpy())
        r.update({"R@1": rm["Recall@1"], "R@5": rm["Recall@5"], "R@10": rm["Recall@10"],
                  "MRR": rm["MRR"], "nDCG@10": rm["nDCG@10"]})
    rows.append(r)
ladder = pd.DataFrame(rows)
ladder.to_csv(f"{args.out}/exp_ladder.csv", index=False)
def to_md(df):
    """Markdown table without the optional `tabulate` dependency."""
    df = df.round(3)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


open(f"{args.out}/exp_ladder.md", "w").write(to_md(ladder))
print("== Ladder (absolute metrics, delta vs Baseline 0) ==")
print(ladder.round(3).to_string(index=False))


def paired(sys_k, ref_k, metric="EM"):
    a, b = data[sys_k][metric].to_numpy(), data[ref_k][metric].to_numpy()
    diff = a - b
    lo, hi = ci(diff[idx].mean(axis=1))
    ea, eb = data[sys_k]["EM"].to_numpy() > 0, data[ref_k]["EM"].to_numpy() > 0
    b01, b10 = int((ea & ~eb).sum()), int((~ea & eb).sum())
    pval = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"diff": float(diff.mean()), "ci_lo": lo, "ci_hi": hi, "only_sys": b01, "only_ref": b10,
            "p_mcnemar": float(pval), "significant": bool(lo > 0 or hi < 0)}


# ---------------- experiment 1: ablations vs the full agent ----------------
if args.agent in data:
    ab = []
    for k in data:
        if k.startswith("agent_") and k != args.agent:
            for m in ["EM", "Contains", "ROUGE-1"]:
                r = paired(k, args.agent, m)
                ab.append({"config": labels[k], "metric": m, **r})
    ab = pd.DataFrame(ab)
    ab.to_csv(f"{args.out}/exp1_ablation.csv", index=False)
    print("\n== Experiment 1: each configuration minus the full agent (paired) ==")
    print(ab[ab.metric == "EM"].round(3).to_string(index=False) if len(ab) else "(no ablation traces yet)")

    # ---------------- experiment 2: dynamic vs fixed chains ----------------
    ex2 = []
    for k in data:
        if k.startswith("baseline_") and k != "baseline_none_v3":
            for m in ["EM", "Contains", "ROUGE-1"]:
                ex2.append({"fixed_chain": labels[k], "metric": m, **paired(args.agent, k, m)})
    ex2 = pd.DataFrame(ex2)
    ex2.to_csv(f"{args.out}/exp2_vs_static.csv", index=False)
    print(f"\n== Experiment 2: {labels[args.agent]} minus each fixed chain (paired) ==")
    print(ex2[ex2.metric == "EM"].round(3).to_string(index=False))

    cats = data[args.agent].cat_num
    cols = {labels[k]: d.groupby("cat_num")["EM"].mean() for k, d in data.items()
            if k == args.agent or (k.startswith("baseline_") and k != "baseline_none_v3")}
    pc = pd.DataFrame(cols)
    pc.insert(0, "n", cats.value_counts().sort_index())
    if args.best_static in data:
        pc["agent - best_static"] = pc[labels[args.agent]] - pc[labels[args.best_static]]
    pc.to_csv(f"{args.out}/exp2_per_category.csv")
    print("\n== Experiment 2: EM per category (small n: differences of 1-3 questions are noise) ==")
    print(pc.round(3).to_string())
else:
    print(f"\n[note] {args.agent} not available yet; experiments 1-2 tables skipped")


# ---------------- selected paired comparisons (guardrail effect, agents vs the strongest fixed chain) ----------------
PAIRS = [
    ("guardrail effect, MedGemma scheduler", "agent_full_guard", "agent_full"),
    ("guardrail effect, Granite scheduler", "agent_full_granite_guard", "agent_full_granite"),
    ("Granite: local-index guardrail variant vs original guardrail", "agent_full_granite_guardlocal", "agent_full_granite_guard"),
    ("Granite: local-index guardrail variant vs Vanilla RAG", "agent_full_granite_guardlocal", "baseline_vanilla_rag_v3"),
    ("MedGemma agent + guardrail vs Vanilla RAG", "agent_full_guard", "baseline_vanilla_rag_v3"),
    ("MedGemma agent + guardrail vs index-only agent", "agent_full_guard", "agent_index_only"),
    ("Granite agent + guardrail vs Vanilla RAG", "agent_full_granite_guard", "baseline_vanilla_rag_v3"),
    ("Granite vs MedGemma scheduler (both + guardrail)", "agent_full_granite_guard", "agent_full_guard"),
    ("index-only agent vs Vanilla RAG", "agent_index_only", "baseline_vanilla_rag_v3"),
]
rows = []
for label, a, b in PAIRS:
    if a in data and b in data:
        rows.append({"comparison": label, "EM_sys": data[a]["EM"].mean(), "EM_ref": data[b]["EM"].mean(), **paired(a, b, "EM")})
if rows:
    pairs = pd.DataFrame(rows)
    pairs.to_csv(f"{args.out}/exp_pairs.csv", index=False)
    print("\n== Selected paired comparisons (EM) ==")
    print(pairs.round(3).to_string(index=False))
