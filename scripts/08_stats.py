"""Paired statistics over per-question traces (CPU only).

For each system: EM / Contains / ROUGE-1 with bootstrap 95% CI.
Versus a reference system: paired difference (bootstrap 95% CI) and an exact McNemar test on EM.
Oracle upper bound: a question counts as solved if ANY of the chosen systems solves it
(what perfect per-question routing could reach). Also per-category EM and complementarity.

Usage: python 08_stats.py --systems none vanilla_rag reformulate_rag reasoning_rag fda_rag \
           --ref vanilla_rag --prefix baseline_ --suffix _v3 --out stats_baselines
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import binomtest

p = argparse.ArgumentParser()
p.add_argument("--systems", nargs="+", required=True)
p.add_argument("--ref", required=True)
p.add_argument("--prefix", default="baseline_")
p.add_argument("--suffix", default="_v3")
p.add_argument("--oracle_over", nargs="*", default=None,
               help="systems that take part in the oracle (default: all except 'none')")
p.add_argument("--metric", default="EM")
p.add_argument("--n_boot", type=int, default=10000)
p.add_argument("--out", default="stats")
args = p.parse_args()
rng = np.random.default_rng(0)


def load(name):
    rows = [json.loads(l) for l in open(f"../traces/{args.prefix}{name}{args.suffix}.jsonl")]
    return pd.DataFrame(rows).set_index("qid").sort_index()


data = {s: load(s) for s in args.systems}
qids = sorted(set.intersection(*[set(d.index) for d in data.values()]))
data = {s: d.loc[qids] for s, d in data.items()}
n = len(qids)
cat = data[args.ref].cat_num.to_numpy()
idx = rng.integers(0, n, size=(args.n_boot, n))   # shared bootstrap resamples -> paired CIs


def ci(x):
    return np.percentile(x, [2.5, 97.5])


out = {"n": n, "systems": {}, "vs_ref": {}}
print(f"n = {n} questions; reference = {args.ref}; metric for tests = {args.metric}\n")
print(f"{'system':18s}" + "".join(f"{m:>26s}" for m in ["EM", "Contains", "ROUGE-1"]))
for s, d in data.items():
    row, out["systems"][s] = f"{s:18s}", {}
    for m in ["EM", "Contains", "ROUGE-1"]:
        v = d[m].to_numpy()
        lo, hi = ci(v[idx].mean(axis=1))
        out["systems"][s][m] = {"mean": float(v.mean()), "ci95": [float(lo), float(hi)]}
        row += f"{v.mean():10.3f} [{lo:.3f},{hi:.3f}]"
    print(row)

print(f"\nPaired difference vs {args.ref} ({args.metric}); McNemar exact test on EM")
print(f"{'system':18s}{'diff':>8s}{'95% CI':>20s}{'only sys':>10s}{'only ref':>10s}{'p (McNemar)':>14s}")
ref = data[args.ref][args.metric].to_numpy()
refEM = data[args.ref]["EM"].to_numpy() > 0
for s, d in data.items():
    if s == args.ref:
        continue
    v = d[args.metric].to_numpy()
    diff = v - ref
    lo, hi = ci(diff[idx].mean(axis=1))
    em = d["EM"].to_numpy() > 0
    b, c = int((em & ~refEM).sum()), int((~em & refEM).sum())   # discordant pairs
    pval = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    out["vs_ref"][s] = {"diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
                        "only_system": b, "only_ref": c, "p_mcnemar": float(pval)}
    sig = "*" if lo > 0 or hi < 0 else ""
    print(f"{s:18s}{diff.mean():+8.3f}  [{lo:+.3f},{hi:+.3f}]{sig:1s}{b:10d}{c:10d}{pval:14.4f}")
print("(* = 95% CI excludes 0)")

# oracle upper bound and complementarity
pool = args.oracle_over if args.oracle_over else [s for s in data if s != "none"]
M = np.stack([data[s]["EM"].to_numpy() > 0 for s in pool])
oracle = M.any(axis=0)
best_single = max(M.mean(axis=1))
out["oracle"] = {"over": pool, "oracle_EM": float(oracle.mean()), "best_single_EM": float(best_single),
                 "headroom": float(oracle.mean() - best_single),
                 "solved_by_all": float(M.all(axis=0).mean()),
                 "solved_by_none": float((~oracle).mean())}
print(f"\nOracle over {pool}: EM {oracle.mean():.3f} (best single {best_single:.3f}, headroom "
      f"{oracle.mean() - best_single:+.3f}); solved by all {M.all(axis=0).mean():.3f}, by none {(~oracle).mean():.3f}")

# per-category EM
tab = pd.DataFrame({s: d.groupby("cat_num")["EM"].mean() for s, d in data.items()})
tab["oracle"] = pd.Series(oracle, index=data[args.ref].index).groupby(cat).mean()
tab.insert(0, "n", pd.Series(cat).value_counts().sort_index())
print("\nEM by category:")
print(tab.round(3).to_string())
out["per_category_EM"] = {int(k): {c: float(v) for c, v in r.items()} for k, r in tab.iterrows()}
json.dump(out, open(f"../results/{args.out}.json", "w"), indent=1)
print(f"\nsaved ../results/{args.out}.json")
