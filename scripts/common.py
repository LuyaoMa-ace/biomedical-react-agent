"""Shared helpers: data loading and retrieval metrics.

With one gold paper per question, every ranking has exactly one relevant
document, so AP == RR (MAP == MRR) and nDCG@k = 1/log2(rank+1) if rank <= k.
"""
import json, math
import numpy as np
import pandas as pd


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_data(sample_path="../data/sample.jsonl", corpus_path="../data/corpus.jsonl",
              distractor_path=None):
    """Gold corpus (+ optional distractor abstracts). Docs without abstract are dropped."""
    sample = pd.DataFrame(read_jsonl(sample_path))
    rows = read_jsonl(corpus_path)
    if distractor_path:
        rows += read_jsonl(distractor_path)
    corpus = pd.DataFrame(rows).drop_duplicates("pmid")
    corpus = corpus[corpus.abstract.notna() & (corpus.abstract.str.len() > 0)]
    corpus["text"] = corpus["title"].fillna("") + ". " + corpus["abstract"].fillna("")
    return sample, corpus.reset_index(drop=True)


def rank_of_gold(scores, gold_idx):
    """1-based rank of the gold doc for each query. scores: (n_q, n_docs)."""
    gold = scores[np.arange(len(gold_idx)), gold_idx][:, None]
    return (scores > gold).sum(axis=1) + 1


def retrieval_metrics(ranks, ks=(1, 5, 10)):
    ranks = np.asarray(ranks, dtype=float)
    out = {f"Recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / ranks).mean())
    out["MAP"] = out["MRR"]  # single relevant doc per query
    for k in ks:
        out[f"nDCG@{k}"] = float(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).mean())
    return out
