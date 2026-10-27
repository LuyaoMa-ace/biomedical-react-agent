"""Quick diagnostics for an agent run vs. Vanilla RAG on the same questions (CPU only).

Usage: python 10_agent_diag.py agent_full_smoke3 [--ref baseline_vanilla_rag_v3] [--show 4]
"""
import argparse, json
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("--ref", default="baseline_vanilla_rag_v3")
p.add_argument("--show", type=int, default=4)
a = p.parse_args()
A = {r["qid"]: r for r in map(json.loads, open(f"../traces/{a.name}.jsonl"))}
V = {r["qid"]: r for r in map(json.loads, open(f"../traces/{a.ref}.jsonl")) if r["qid"] in A}
df = pd.DataFrame([{"qid": q, "stop": A[q].get("stop"), "agent_EM": A[q]["EM"], "ref_EM": V[q]["EM"],
                    "steps": A[q]["steps"], "tools": A[q]["tool_calls"], "gold_rank": A[q]["gold_rank"],
                    "gold_seen": A[q]["gold_seen"]} for q in A])
print(f"{a.name}: n={len(df)}  agent EM {df.agent_EM.mean():.3f} | {a.ref} EM {df.ref_EM.mean():.3f} | "
      f"agent right/ref wrong {((df.agent_EM > 0) & (df.ref_EM == 0)).sum()} | ref right/agent wrong {((df.agent_EM == 0) & (df.ref_EM > 0)).sum()}")
print("\nby stop reason:")
print(df.groupby("stop").agg(n=("qid", "count"), agent_EM=("agent_EM", "mean"), ref_EM=("ref_EM", "mean"),
                             steps=("steps", "mean"), tools=("tools", "mean")).round(2))
ag = [(r.get("agent_answer"), r["pred"]) for r in A.values() if r.get("agent_answer")]
if ag:
    same = sum(x.lower().strip(" .") == y.lower().strip(" .") for x, y in ag)
    print(f"\nscheduler's own Final Answer equals the generator's answer in {same}/{len(ag)} cases")
print("\ngold in first retrieval top-1: %.2f | gold seen in any observation: %.2f" % ((df.gold_rank == 1).mean(), df.gold_seen.mean()))
print("steps distribution:", df.steps.value_counts().sort_index().to_dict())
use = pd.DataFrame([r["tool_usage"] for r in A.values()]).fillna(0)
print("tool usage per question:", use.mean().round(2).to_dict())
print("\n--- first %d traces ---" % a.show)
for r in list(A.values())[:a.show]:
    print("=" * 80); print("Q:", r["question"][:130]); print("GOLD:", r["gold"], "| PRED:", r["pred"], "| EM:", r["EM"], "| stop:", r.get("stop"), "| scheduler said:", r.get("agent_answer"))
    for st in r["trace"]:
        print(f"  [{st['step']}] {st['kind']}", st.get("tool", ""), "|", (st.get("input") or st.get("answer") or st.get("reason") or "")[:90])
        if st.get("thought"): print("      thought:", st["thought"][:150])
        if st.get("observation"): print("      obs:", st["observation"][:100].replace("\n", " "))
