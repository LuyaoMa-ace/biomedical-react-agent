"""Dense (jina) top-100 PMIDs for every question of both samples, saved as JSON (notebook section 3b).

Only this step needs FAISS / torch (run on giano). 17_bm25_eval.py --hybrid then reads the JSON on any CPU.
Usage: python 17b_dense_top100.py --data_dir ../data --out ../results/bm25
"""
import argparse, json, os, time
from common import load_data

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="../data")
ap.add_argument("--out", default="../results/bm25")
ap.add_argument("--index_dir", default="../index")
ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from tools import LocalRetriever

# 16 queries per FAISS call, as in 06_baselines.py. One call with all 500 questions crashed with "Illegal instruction" inside
# faiss index.search on the cluster (batches of 20+ queries presumably take FAISS's BLAS code path, which these nodes cannot run).
BATCH = 16
dense = LocalRetriever(index_dir=args.index_dir, tag="_5k", device=args.device)
for name, fname in (("development", "sample.jsonl"), ("held-out", "holdout.jsonl")):
    s, _ = load_data(f"{args.data_dir}/{fname}", f"{args.data_dir}/corpus.jsonl")
    t0 = time.time()
    qs, hits = s.question.tolist(), []
    for i in range(0, len(qs), BATCH):                      # small batches: see BATCH
        hits += dense.search_batch(qs[i:i + BATCH], k=100)
    sec = time.time() - t0
    json.dump({"qids": [int(q) for q in s.qid], "pmids": [[str(h["pmid"]) for h in hh] for hh in hits]},
              open(f"{args.out}/dense_top100_{name}.json", "w"))
    print(f"{name}: {len(s)} questions, {sec:.1f} s ({sec / len(s) * 1000:.0f} ms/question incl. batching)")
