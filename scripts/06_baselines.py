"""Baseline 0 (no tools) and the four fixed-chain pipelines (static RAG).

  none               : question -> answer
  vanilla_rag        : retrieve_from_index -> answer
  reformulate_rag    : query_reformulator -> retrieve_from_index -> answer
  reasoning_rag      : generate_reasoning_query -> retrieve_from_index -> answer
  fda_rag            : retrieve_from_index + query_openfda (drug extracted from the question) -> answer

Every question goes through exactly the same steps (that is what "static" means).
Counting convention: tool_calls = tools in the chain (fda_rag always 2: the OpenFDA call is
made even when no drug is found); steps = LLM generation calls (incl. the final answer).
Retrieval metrics rank the gold paper for the query the chain actually used.

Resumable: records are appended to ../traces/baseline_<mode>_<prompt>.jsonl; finished qids are skipped.
Latency is amortised over the batch (batch_time / batch_size).

Usage: python 06_baselines.py --mode none
       python 06_baselines.py --mode reformulate_rag
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from common import load_data, retrieval_metrics
from qa_utils import score_prediction, aggregate

RAG_MODES = ["vanilla_rag", "reformulate_rag", "reasoning_rag", "fda_rag"]
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["none"] + RAG_MODES, required=True)
p.add_argument("--n", type=int, default=None, help="only the first n questions (smoke test)")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--k", type=int, default=3, help="documents placed in the prompt")
p.add_argument("--k_eval", type=int, default=100, help="depth used to rank the gold paper")
p.add_argument("--dtype", default="bfloat16")
p.add_argument("--index_tag", default="_5k")
p.add_argument("--retr_device", default=None, choices=["cpu", "cuda"],
               help="device for the jina query encoder; use cpu on 11GB GPUs")
p.add_argument("--sample", default="../data/sample.jsonl", help="question file (held-out run: ../data/holdout.jsonl with --tag _holdout)")
p.add_argument("--tag", default="")
p.add_argument("--prompt", default="v3", choices=["v1", "v2", "v3", "v4", "v5", "v6"])
args = p.parse_args()

assert args.sample == "../data/sample.jsonl" or args.tag, "use --tag when running another question file (traces would collide)"
sample, _ = load_data(args.sample)
if args.n:
    sample = sample.head(args.n)
name = f"baseline_{args.mode}{'' if args.prompt == 'v1' else '_' + args.prompt}{args.tag}"
os.makedirs("../traces", exist_ok=True); os.makedirs("../results", exist_ok=True)
trace_path = f"../traces/{name}.jsonl"
done = set()
if os.path.exists(trace_path):
    done = {json.loads(l)["qid"] for l in open(trace_path)}
todo = sample[~sample.qid.isin(done)]
print(f"{name}: {len(sample)} questions, {len(done)} done, {len(todo)} to go")

PROMPTS = {
    # v1: first version (its example answer leaked into outputs: the model copied it)
    "v1": dict(
        head=("You are a biomedical expert. Answer the question with a short phrase only "
              "(a few words). Do not explain.\n\n"),
        example=("Example of the answer format:\nQ: Which enzyme converts angiotensin I to angiotensin II?\n"
                 "A: Angiotensin-converting enzyme\n\n")),
    # v2: neutral, non-biomedical examples; asks for the exact term used in the abstracts
    "v2": dict(
        head=("You are a biomedical expert. Reply with ONLY the answer: a short noun phrase "
              "(at most 6 words), no explanation, no full sentence. When PubMed abstracts are "
              "provided, copy the exact term they use, including its modifiers.\n\n"),
        example=("Examples of the answer format:\nQ: What is the capital of France?\nA: Paris\n\n"
                 "Q: Which gas do plants absorb during photosynthesis?\nA: Carbon dioxide\n\n")),
}
# v3: v2 wording + the model turn is pre-filled with "A:" (stops the "thought ..." reasoning leak)
PROMPTS["v3"] = {**PROMPTS["v2"], "prefill": "A:"}
# prompt-sensitivity variants (same prefill as v3; only the wording changes) -> 14_prompt_sensitivity.py
PROMPTS["v4"] = {**PROMPTS["v3"],  # paraphrase of the v3 instruction, same examples
    "head": ("You answer biomedical questions. Give just the answer as a brief noun phrase of no more than six words, "
             "without any explanation or complete sentence. If PubMed abstracts are given, reuse the wording they use, "
             "modifiers included.\n\n")}
PROMPTS["v5"] = {**PROMPTS["v3"], "example": ""}  # same instruction, no format examples
PROMPTS["v6"] = {**PROMPTS["v3"],  # minimal one-line instruction, no examples
    "head": "Answer with a short phrase only.\n\n", "example": ""}
HEAD, EXAMPLE = PROMPTS[args.prompt]["head"], PROMPTS[args.prompt]["example"]
PREFILL = PROMPTS[args.prompt].get("prefill")

if len(todo):
    from llm import MedGemma
    llm = MedGemma(dtype=args.dtype)
    retriever = openfda = None
    if args.mode in RAG_MODES:
        from tools import (LocalRetriever, format_docs, query_reformulator,
                           generate_reasoning_query, extract_drug, OpenFDA, format_fda)
        retriever = LocalRetriever(tag=args.index_tag, device=args.retr_device)
    if args.mode == "fda_rag":
        openfda = OpenFDA()

    with open(trace_path, "a") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo.iloc[i:i + args.batch]
            qs, m = chunk.question.tolist(), len(chunk)
            t0 = time.time()
            queries, hits, drugs, labels = qs, [None] * m, [""] * m, [{}] * m
            tool_calls, steps = 0, 1
            if args.mode == "reformulate_rag":
                queries = query_reformulator(llm, qs); tool_calls, steps = 1, 2
            elif args.mode == "reasoning_rag":
                queries = generate_reasoning_query(llm, qs); tool_calls, steps = 1, 2
            elif args.mode == "fda_rag":
                drugs = extract_drug(llm, qs); steps = 2
                labels = [openfda(d) for d in drugs]; tool_calls = 1
            if retriever:
                hits = retriever.search_batch(queries, k=args.k_eval); tool_calls += 1

            prompts = []
            for q, h, lab, d in zip(qs, hits, labels, drugs):
                ctx = ""
                if h is not None:
                    ctx += ("Use the PubMed abstracts below if they help.\n\nContext:\n"
                            + format_docs(h[:args.k]) + "\n\n")
                if lab:
                    ctx += f"Drug label information (OpenFDA) for {d}:\n{format_fda(lab)}\n\n"
                prompts.append(HEAD + ctx + EXAMPLE + f"Q: {q}\nA:")
            raws = llm.generate(prompts, max_new_tokens=48, prefill=PREFILL)
            lat = (time.time() - t0) / m
            for (_, r), raw, h, qy, d, lab in zip(chunk.iterrows(), raws, hits, queries, drugs, labels):
                golds = list(dict.fromkeys(list(r.golden_answers) + [r.answer]))
                rec = {"qid": int(r.qid), "cat_num": int(r.cat_num), "question": r.question,
                       "gold": r.answer, "raw": raw, **score_prediction(raw, golds),
                       "tool_calls": tool_calls, "steps": steps, "latency_s": lat}
                if qy != r.question:
                    rec["query"] = qy
                if args.mode == "fda_rag":
                    rec["drug"], rec["fda_hit"] = d, bool(lab)
                if h is not None:
                    pm = [x["pmid"] for x in h]
                    rec["gold_rank"] = pm.index(r.pmid) + 1 if r.pmid in pm else 10 ** 6
                    rec["retrieved_pmids"] = pm[:args.k]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{min(i + args.batch, len(todo))}/{len(todo)}  {lat:.2f}s/q")

df = pd.DataFrame([json.loads(l) for l in open(trace_path)])
res = aggregate(df)
if "gold_rank" in df:
    res["retrieval"] = retrieval_metrics(df.gold_rank.to_numpy())
    res["retrieval_per_category"] = {int(c): retrieval_metrics(g.gold_rank.to_numpy())
                                     for c, g in df.groupby("cat_num")}
if "fda_hit" in df:
    res["fda"] = {"drug_extracted": float((df.drug != "").mean()), "label_found": float(df.fda_hit.mean())}
json.dump(res, open(f"../results/{name}.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("overall", "retrieval", "fda") if k in res}, indent=1))
