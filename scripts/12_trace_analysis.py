"""Experiment 3: behavioural analysis of ReAct traces (CPU only).

Outputs (in ../results):
  fig_tool_heatmap_<agent>.png / tool_heatmap_<agent>.csv   category x tool, cell = mean calls per question
  fig_steps_vs_ndcg_<agent>.png / steps_vs_ndcg_<agent>.csv  nDCG@10 (and EM) against number of reasoning steps
  failure_rates_<agent>.csv / fig_failures_<agent>.png      failure-mode rate per category
  trace_examples_<agent>.md                                 side-by-side fixed chain vs dynamic agent traces

Failure modes (a question can have several):
  looping        repeated an identical action, or the loop breaker had to stop the agent
  format         at least one reply that could not be parsed as Action / Final Answer
  stopped_early  decided to answer without calling any tool
  bad_args       unknown tool, tool error, or a PubMed call that returned nothing (non-keyword input)
  retrieval_miss the gold paper never appeared in any observation
  reading_error  the gold paper was seen but the final answer is still wrong
Hallucinated documents cannot be detected reliably from truncated observations; this is a limitation.

Usage: python 12_trace_analysis.py --agent agent_full --static baseline_vanilla_rag_v3
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

TOOLS = ["retrieve_from_index", "query_reformulator", "generate_reasoning_query", "fetch_pubmed_live", "query_openfda"]
p = argparse.ArgumentParser()
p.add_argument("--agent", default="agent_full")
p.add_argument("--static", default="baseline_vanilla_rag_v3")
p.add_argument("--examples", type=int, default=3, help="examples per direction (agent better / worse)")
p.add_argument("--traces", default="../traces")
p.add_argument("--out", default="../results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
tag = args.agent


def load(name):
    return pd.DataFrame([json.loads(l) for l in open(f"{args.traces}/{name}.jsonl")]).drop_duplicates("qid").set_index("qid").sort_index()


A = load(args.agent)
S = load(args.static) if os.path.exists(f"{args.traces}/{args.static}.jsonl") else None
print(f"{args.agent}: {len(A)} questions")

# ---------------- 1. tool usage heatmap (category x tool) ----------------
use = pd.DataFrame(list(A.tool_usage), index=A.index).reindex(columns=TOOLS).fillna(0)
heat = use.groupby(A.cat_num).mean()
heat.to_csv(f"{args.out}/tool_heatmap_{tag}.csv")
print("\n== mean tool calls per question, by category ==")
print(heat.round(2).to_string())
print("overall:", use.mean().round(3).to_dict())
fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(TOOLS))); ax.set_xticklabels([t.replace("_", "\n", 1) for t in TOOLS], fontsize=8)
ax.set_yticks(range(len(heat))); ax.set_yticklabels([f"cat {c} (n={int((A.cat_num == c).sum())})" for c in heat.index], fontsize=8)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", color="white" if heat.values[i, j] < heat.values.max() / 2 else "black", fontsize=7)
fig.colorbar(im, label="mean calls per question"); ax.set_title(f"Tool usage by question category ({tag})")
fig.tight_layout(); fig.savefig(f"{args.out}/fig_tool_heatmap_{tag}.png", dpi=150); plt.close(fig)

# ---------------- 2. reasoning steps vs retrieval quality ----------------
A["ndcg10"] = np.where(A.gold_rank <= 10, 1.0 / np.log2(A.gold_rank.clip(lower=1) + 1), 0.0)
by = A.groupby("steps").agg(n=("ndcg10", "size"), ndcg10=("ndcg10", "mean"), EM=("EM", "mean"), gold_seen=("gold_seen", "mean"))
by.to_csv(f"{args.out}/steps_vs_ndcg_{tag}.csv")
rho, pv = spearmanr(A.steps, A.ndcg10) if A.steps.nunique() > 1 else (float("nan"), float("nan"))
print(f"\n== reasoning steps vs nDCG@10 ==  Spearman rho = {rho:.3f} (p = {pv:.3g})")
print(by.round(3).to_string())
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(by.index, by.ndcg10, "o-", label="nDCG@10"); ax.plot(by.index, by.EM, "s--", label="EM")
for x, y, c in zip(by.index, by.ndcg10, by.n):
    ax.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
ax.set_xlabel("number of reasoning steps (agent iterations)"); ax.set_ylabel("mean score"); ax.set_ylim(0, 1.05)
ax.set_title(f"Longer reasoning chains vs quality ({tag}); Spearman rho={rho:.2f}"); ax.legend()
fig.tight_layout(); fig.savefig(f"{args.out}/fig_steps_vs_ndcg_{tag}.png", dpi=150); plt.close(fig)


# ---------------- 3. failure modes ----------------
def empty_pubmed(trace):
    return any(s.get("tool") == "fetch_pubmed_live" and str(s.get("observation", "")).startswith("No PubMed") for s in trace)


F = pd.DataFrame(index=A.index)
F["looping"] = ((A.repeat_call > 0) | (A.loop_break > 0)).astype(int)
F["format"] = (A.parse_error > 0).astype(int)
F["stopped_early"] = (A.no_tool_answer > 0).astype(int)
F["bad_args"] = ((A.unknown_tool > 0) | (A.tool_error > 0) | A.trace.apply(empty_pubmed)).astype(int)
F["retrieval_miss"] = (~A.gold_seen.astype(bool)).astype(int)
F["reading_error"] = (A.gold_seen.astype(bool) & (A.EM == 0)).astype(int)
fail = F.groupby(A.cat_num).mean()
fail.loc["all"] = F.mean()
fail.insert(0, "n", list(A.cat_num.value_counts().sort_index()) + [len(A)])
fail.to_csv(f"{args.out}/failure_rates_{tag}.csv")
print("\n== failure-mode rate per category (share of questions) ==")
print(fail.round(3).to_string())
fig, ax = plt.subplots(figsize=(8, 5.5))
vals = fail.drop(columns="n").values
im = ax.imshow(vals, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
ax.set_xticks(range(vals.shape[1])); ax.set_xticklabels(fail.columns[1:], rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(fail))); ax.set_yticklabels([f"cat {c}" if c != "all" else "ALL" for c in fail.index], fontsize=8)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", color="black" if vals[i, j] < .6 else "white", fontsize=7)
fig.colorbar(im, label="share of questions"); ax.set_title(f"Failure modes by category ({tag})")
fig.tight_layout(); fig.savefig(f"{args.out}/fig_failures_{tag}.png", dpi=150); plt.close(fig)

# ---------------- 4. side-by-side examples: fixed chain vs dynamic agent ----------------
if S is not None:
    both = A.index.intersection(S.index)
    better = [q for q in both if A.loc[q, "EM"] > S.loc[q, "EM"]]
    worse = [q for q in both if A.loc[q, "EM"] < S.loc[q, "EM"]]
    rng = np.random.default_rng(0)
    pick = [("agent BETTER than the fixed chain", list(rng.permutation(better)[:args.examples])),
            ("agent WORSE than the fixed chain", list(rng.permutation(worse)[:args.examples]))]
    md = [f"# Fixed chain ({args.static}) vs dynamic agent ({args.agent})\n",
          f"Questions where EM differs: agent better {len(better)}, agent worse {len(worse)} (of {len(both)}).\n"]
    for title, qs in pick:
        md.append(f"\n## {title}\n")
        for q in qs:
            a, s = A.loc[q], S.loc[q]
            md.append(f"### qid {q}  (category {int(a.cat_num)})\n**Question:** {a.question}\n\n**Gold:** {a.gold}\n")
            md.append(f"**Fixed chain:** pred = `{s.pred}`  (EM {s.EM:.0f})" + (f", gold paper rank {int(s.gold_rank)}" if "gold_rank" in s and s.gold_rank < 10 ** 6 else "") + "\n")
            md.append(f"**Agent:** pred = `{a.pred}`  (EM {a.EM:.0f}); stop = {a.get('stop')}; steps = {a.steps}; tools = {a.tool_usage}; gold paper seen = {bool(a.gold_seen)}\n")
            for st in a.trace:
                line = f"- step {st['step']} [{st['kind']}]"
                if st.get("tool"): line += f" `{st['tool']}({st.get('input', '')[:80]})`"
                if st.get("reason"): line += f" reason={st['reason']}"
                if st.get("thought"): line += f"  — thought: {st['thought'][:120]}"
                if st.get("observation"): line += f"  → obs: {st['observation'][:90]!r}"
                md.append(line)
            notes = []
            if a.get("repeat_call", 0): notes.append("repeated an identical action")
            if a.get("loop_break", 0): notes.append("loop breaker stopped it")
            if a.get("parse_error", 0): notes.append("format error")
            if not a.gold_seen: notes.append("gold paper never retrieved")
            if a.tool_calls == 0: notes.append("answered without any tool")
            md.append(f"\n*Auto notes:* {'; '.join(notes) or 'none'}\n\n*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?\n")
    open(f"{args.out}/trace_examples_{tag}.md", "w").write("\n".join(md))
    print(f"\nwrote {args.out}/trace_examples_{tag}.md  ({len(better)} better / {len(worse)} worse questions)")
print("\nfigures + csv written to", args.out)
