"""Coverage experiment: what happens when the gold paper is NOT in the local index?

The gold papers of a fixed random fraction of the held-out questions are removed from the index
(--withhold 0.5 / 1.0; nested sets, see common.withheld_qids). Only live PubMed can still supply them.
Compares, per coverage level: no tools, vanilla RAG, reformulate -> retrieve, two chains that always call PubMed
(pubmed_rag, hybrid_rag), the agent with the guardrail, and the same agent without fetch_pubmed_live.

Reads  ../traces/<system><tag>.jsonl   (tag = _holdout, _holdout_w50, _holdout_w100)
Writes ../results/coverage/*.csv and fig_coverage.png

Usage: python 16_coverage.py
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from common import load_data, withheld_qids

p = argparse.ArgumentParser()
p.add_argument("--sample", default="../data/holdout.jsonl")
p.add_argument("--out", default="../results/coverage")
p.add_argument("--n_boot", type=int, default=2000)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

LEVELS = {0: "_holdout", 50: "_holdout_w50", 100: "_holdout_w100"}          # percent withheld -> trace tag
SYSTEMS = {                                                                    # label -> trace file template
    "Baseline 0 (no tools)": "baseline_none_v3_holdout",                         # does not touch the index
    "Vanilla RAG": "baseline_vanilla_rag_v3{tag}",
    "Reformulate -> retrieve": "baseline_reformulate_rag_v3{tag}",
    "Reformulate -> PubMed (chain)": "baseline_pubmed_rag_v3{tag}",
    "Retrieve + PubMed (chain)": "baseline_hybrid_rag_v3{tag}",
    "Agent: full + guardrail": "agent_full_guard{tag}",
    "Agent: no PubMed + guardrail": "agent_no_pubmed_guard{tag}",
}
sample, _ = load_data(args.sample)
rng = np.random.default_rng(0)


def load(label, pct):
    path = f"../traces/{SYSTEMS[label].format(tag=LEVELS[pct])}.jsonl"
    if not os.path.exists(path):
        return None
    df = pd.DataFrame([json.loads(l) for l in open(path)]).set_index("qid").sort_index()
    return df if len(df) == len(sample) else print(f"[skip] {path}: {len(df)} rows") or None


T = {(s, pct): load(s, pct) for s in SYSTEMS for pct in LEVELS}
withheld = {pct: withheld_qids(sample, pct / 100) for pct in LEVELS}


def em(df, qids=None):
    d = df if qids is None else df.loc[sorted(qids)]
    return float(d.EM.mean()) if len(d) else np.nan


# ---- 1. EM per system and coverage level, split into withheld / retained questions ---------------------------
rows = []
for s in SYSTEMS:
    for pct in LEVELS:
        df = T[(s, pct)]
        if df is None:
            continue
        w = withheld[pct]
        kept = set(df.index) - w
        r = {"system": s, "withheld_%": pct, "EM": em(df), "EM_withheld_qs": em(df, w) if w else np.nan,
             "EM_retained_qs": em(df, kept) if kept else np.nan, "tool_calls": df.tool_calls.mean(),
             "latency_s": df.latency_s.mean()}
        if "gold_seen" in df:
            r["gold_seen_withheld_qs"] = float(df.loc[sorted(w)].gold_seen.mean()) if w else np.nan
            r["gold_seen_retained_qs"] = float(df.loc[sorted(kept)].gold_seen.mean()) if kept else np.nan
        rows.append(r)
tab = pd.DataFrame(rows)
tab.round(3).to_csv(f"{args.out}/coverage_em.csv", index=False)
print("== EM by coverage level (withheld_% = share of questions whose gold paper is not in the index) ==")
print(tab.round(3).to_string(index=False))

# ---- 2. does the agent switch to PubMed when (and only when) the local index cannot answer? -------------------
rows = []
for s in ("Agent: full + guardrail",):
    for pct in LEVELS:
        df = T[(s, pct)]
        if df is None:
            continue
        used = df.tool_usage.apply(lambda u: u.get("fetch_pubmed_live", 0) > 0)
        retr = df.tool_usage.apply(lambda u: u.get("retrieve_from_index", 0) > 0)
        w = withheld[pct]
        kept = set(df.index) - w
        rows.append({"withheld_%": pct,
                     "pubmed_used_overall": float(used.mean()),
                     "pubmed_used_withheld_qs": float(used.loc[sorted(w)].mean()) if w else np.nan,
                     "pubmed_used_retained_qs": float(used.loc[sorted(kept)].mean()) if kept else np.nan,
                     "index_used_overall": float(retr.mean()),
                     "gold_seen_withheld_qs": float(df.loc[sorted(w)].gold_seen.mean()) if w else np.nan,
                     "gold_seen_retained_qs": float(df.loc[sorted(kept)].gold_seen.mean()) if kept else np.nan})
sw = pd.DataFrame(rows)
sw.round(3).to_csv(f"{args.out}/coverage_switching.csv", index=False)
print("\n== Agent (full + guardrail): share of questions with >= 1 fetch_pubmed_live call ==")
print(sw.round(3).to_string(index=False))


# ---- 3. paired comparisons (EM) -----------------------------------------------------------------------------
def paired(a, b):
    x, y = a.sort_index().EM.to_numpy(), b.sort_index().EM.to_numpy()
    d = x - y
    boot = rng.choice(d, size=(args.n_boot, len(d))).mean(axis=1)
    so, sr = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    pv = binomtest(so, so + sr, 0.5).pvalue if so + sr else 1.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"EM_a": x.mean(), "EM_b": y.mean(), "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "only_a": so, "only_b": sr, "p_mcnemar": pv, "significant": bool(lo > 0 or hi < 0)}


PAIRS = [("Agent: full + guardrail", "Agent: no PubMed + guardrail", "PubMed tool effect (agent)"),
         ("Agent: full + guardrail", "Vanilla RAG", "agent vs Vanilla RAG"),
         ("Agent: full + guardrail", "Retrieve + PubMed (chain)", "agent vs always-PubMed hybrid chain"),
         ("Retrieve + PubMed (chain)", "Vanilla RAG", "hybrid chain vs Vanilla RAG"),
         ("Reformulate -> PubMed (chain)", "Baseline 0 (no tools)", "PubMed-only chain vs no tools")]
rows = []
for a, b, label in PAIRS:
    for pct in LEVELS:
        if T[(a, pct)] is not None and T[(b, pct)] is not None:
            rows.append({"comparison": label, "withheld_%": pct, **paired(T[(a, pct)], T[(b, pct)])})
pc = pd.DataFrame(rows)
pc.round(4).to_csv(f"{args.out}/coverage_pairs.csv", index=False)
print("\n== Paired differences in EM (a - b); bootstrap 95% CI, exact McNemar ==")
print(pc.round(3).to_string(index=False))

# ---- 4. figure ----------------------------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s in SYSTEMS:
        d = tab[tab.system == s]
        if len(d) > 1:
            ax.plot(d["withheld_%"], d.EM, marker="o", label=s)
    ax.set_xlabel("questions whose gold paper is withheld from the index (%)")
    ax.set_ylabel("EM (500 held-out questions)")
    ax.set_xticks(list(LEVELS))
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_coverage.png", dpi=150)
    print(f"\nsaved {args.out}/fig_coverage.png")
except Exception as e:
    print("figure skipped:", e)
