"""Run the dynamic ReAct agent (LangGraph) on the sample and log full traces.

Configs (experiment 1): full | no_fda | no_pubmed | no_reasoning | no_reformulator | index_only
Resumable: records are appended to ../traces/agent_<config><tag>.jsonl; finished qids are skipped.
Latency is wall time of a chunk / chunk size (amortised, like the baselines).

Usage: python 09_run_agent.py --config full
       python 09_run_agent.py --config full --n 32 --tag _smoke
"""
import argparse, json, os, threading, time
import pandas as pd
from common import load_data, retrieval_metrics
from qa_utils import score_prediction, aggregate

ALL_TOOLS = ["retrieve_from_index", "query_reformulator", "generate_reasoning_query",
             "fetch_pubmed_live", "query_openfda"]
CONFIGS = {
    "full": ALL_TOOLS,
    "no_fda": [t for t in ALL_TOOLS if t != "query_openfda"],
    "no_pubmed": [t for t in ALL_TOOLS if t != "fetch_pubmed_live"],
    "no_reasoning": [t for t in ALL_TOOLS if t != "generate_reasoning_query"],
    "no_reformulator": [t for t in ALL_TOOLS if t != "query_reformulator"],
    "index_only": ["retrieve_from_index"],
}
FAIL_KEYS = ["parse_error", "unknown_tool", "repeat_call", "tool_error", "forced_final", "loop_break", "no_tool_answer", "guardrail_retrieval"]

p = argparse.ArgumentParser()
p.add_argument("--config", choices=list(CONFIGS), default="full")
p.add_argument("--orchestrator", choices=["medgemma", "granite"], default="medgemma",
               help="model that schedules tools; MedGemma always writes answers and runs the LLM tools")
p.add_argument("--n", type=int, default=None)
p.add_argument("--chunk", type=int, default=64, help="questions per resumable chunk")
p.add_argument("--concurrency", type=int, default=16, help="graph runs in parallel = LLM micro-batch size")
p.add_argument("--max_steps", type=int, default=6)
p.add_argument("--k_show", type=int, default=3, help="abstracts shown per retrieval observation")
p.add_argument("--k_eval", type=int, default=100)
p.add_argument("--dtype", default="bfloat16")
p.add_argument("--index_tag", default="_5k")
p.add_argument("--retr_device", default=None, choices=["cpu", "cuda"])
p.add_argument("--guardrail", action="store_true",
               help="auto-retrieve once if the agent stops without any document evidence")
p.add_argument("--guardrail_local", action="store_true",
               help="variant: also retrieve when the evidence contains no local-index result (e.g. PubMed only)")
p.add_argument("--tag", default="")
args = p.parse_args()
if args.guardrail_local:
    args.guardrail = True

sample, _ = load_data()
if args.n:
    sample = sample.head(args.n)
name = f"agent_{args.config}{'' if args.orchestrator == 'medgemma' else '_granite'}{'_guardlocal' if args.guardrail_local else '_guard' if args.guardrail else ''}{args.tag}"
os.makedirs("../traces", exist_ok=True); os.makedirs("../results", exist_ok=True)
trace_path = f"../traces/{name}.jsonl"
done = set()
if os.path.exists(trace_path):
    done = {json.loads(l)["qid"] for l in open(trace_path)}
todo = sample[~sample.qid.isin(done)]
print(f"{name}: {len(sample)} questions, {len(done)} done, {len(todo)} to go; orchestrator = {args.orchestrator}; tools = {CONFIGS[args.config]}")

if len(todo):
    from llm import MedGemma, CausalLM, BatchingLLM
    from tools import (LocalRetriever, PubMedLive, OpenFDA, format_docs, format_fda,
                       query_reformulator, generate_reasoning_query)
    from agent import ReActAgent

    llm = BatchingLLM(MedGemma(dtype=args.dtype), max_batch=args.concurrency)   # answers + LLM tools
    sched = llm if args.orchestrator == "medgemma" else BatchingLLM(
        CausalLM("ibm-granite/granite-4.0-micro", dtype=args.dtype), max_batch=args.concurrency)
    enabled = CONFIGS[args.config]
    retriever = LocalRetriever(tag=args.index_tag, device=args.retr_device)
    retr_lock = threading.Lock()                     # one encoder call at a time
    pubmed = PubMedLive() if "fetch_pubmed_live" in enabled else None
    fda = OpenFDA() if "query_openfda" in enabled else None

    def t_retrieve(x):
        with retr_lock:
            hits = retriever.search(x, k=args.k_eval)
        return format_docs(hits[:args.k_show], max_chars=1500), \
            {"ranked": [h["pmid"] for h in hits], "shown": [h["pmid"] for h in hits[:args.k_show]]}

    def t_reformulate(x):
        return "Search query: " + query_reformulator(llm, [x])[0], {}

    def t_reasoning(x):
        return "Reasoning (use it as the query for retrieve_from_index): " + generate_reasoning_query(llm, [x])[0], {}

    def t_pubmed(x):
        docs = pubmed(x)
        if not docs:
            return ("No PubMed results. PubMed needs 3-6 specific keywords, not a full question; "
                    "try different keywords."), {"shown": []}
        txt = "\n\n".join(f"[{i + 1}] {d['title']}\n{(d['abstract'] or '')[:1500]}" for i, d in enumerate(docs))
        return txt, {"shown": [d["pmid"] for d in docs]}

    def t_fda(x):
        lab = fda(x)
        return (format_fda(lab) if lab else f"No FDA drug label found for '{x}'."), {}

    impl = {"retrieve_from_index": t_retrieve, "query_reformulator": t_reformulate,
            "generate_reasoning_query": t_reasoning, "fetch_pubmed_live": t_pubmed,
            "query_openfda": t_fda}
    agent = ReActAgent(sched, {t: impl[t] for t in enabled}, max_steps=args.max_steps, answer_llm=llm,
                       guardrail=args.guardrail, guardrail_local=args.guardrail_local)

    with open(trace_path, "a") as f:
        for i in range(0, len(todo), args.chunk):
            chunk = todo.iloc[i:i + args.chunk]
            t0 = time.time()
            states = agent.run_many(chunk.question.tolist(), max_concurrency=args.concurrency)
            lat = (time.time() - t0) / len(chunk)
            for (_, r), s in zip(chunk.iterrows(), states):
                golds = list(dict.fromkeys(list(r.golden_answers) + [r.answer]))
                steps, ranks, seen = s["steps"], [], False
                usage = {}
                for st in steps:
                    m = st.get("meta") or {}
                    if st.get("executed"):
                        usage[st["tool"]] = usage.get(st["tool"], 0) + 1
                    if r.pmid in m.get("ranked", []):
                        ranks.append(m["ranked"].index(r.pmid) + 1)
                    seen = seen or r.pmid in m.get("shown", [])
                    st["meta"] = {k: v for k, v in m.items() if k != "ranked"}   # keep traces small
                rec = {"qid": int(r.qid), "cat_num": int(r.cat_num), "question": r.question,
                       "gold": r.answer, "raw": s["final"] or "", "stop": s["stop"], "agent_answer": s["agent_answer"],
                       **score_prediction(s["final"] or "", golds),
                       "tool_calls": sum(usage.values()), "tool_usage": usage, "steps": int(s["step"]),
                       "latency_s": lat, "gold_rank": min(ranks) if ranks else 10 ** 6,
                       "gold_seen": bool(seen), **{k: int(s["flags"].get(k, 0)) for k in FAIL_KEYS},
                       "trace": steps}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{min(i + args.chunk, len(todo))}/{len(todo)}  {lat:.2f}s/q  "
                  f"(answer-LLM calls {llm.n_calls}/{llm.n_requests} requests; scheduler calls {sched.n_calls}/{sched.n_requests})")

df = pd.DataFrame([json.loads(l) for l in open(trace_path)])
res = aggregate(df)
res["retrieval"] = retrieval_metrics(df.gold_rank.to_numpy())
res["failure_rates"] = {k: float((df[k] > 0).mean()) for k in FAIL_KEYS}
res["gold_seen"] = float(df.gold_seen.mean())
use = pd.DataFrame(list(df.tool_usage)).fillna(0)
res["tool_usage_per_question"] = {c: float(use[c].mean()) for c in use.columns}
res["tool_usage_by_category"] = {int(c): {t: float(v) for t, v in use[df.cat_num == c].mean().items()}
                                 for c in sorted(df.cat_num.unique())}
json.dump(res, open(f"../results/{name}.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("overall", "retrieval", "failure_rates", "gold_seen",
                                      "tool_usage_per_question")}, indent=1))
