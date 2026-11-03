"""Classical retrieval baselines: BM25 and a BM25 + dense (jina) hybrid (notebook section 3b).

Protocol (fixed before running, no tuning): same 5,000 documents (title + abstract), the raw question as the query.
  BM25   : Okapi BM25, k1 = 1.5, b = 0.75, Lucene idf = ln(1 + (N - n + 0.5) / (n + 0.5)); lower-case, tokens [a-z0-9]+,
           English stop words removed, no stemming; every distinct query term counted once.
  hybrid : reciprocal rank fusion of the BM25 and jina rankings (top 100 of each, k = 60, equal weights).
Ranks beyond 100 count as misses for every method, so Recall@k, MRR and nDCG@10 are computed the same way everywhere.

  python 17_bm25_eval.py --data_dir ../data --out ../results/bm25                 BM25 only (CPU, no index needed)
  python 17b_dense_top100.py                                                      (on giano, GPU) jina top-100 per question -> JSON
  python 17_bm25_eval.py --hybrid                                                 + jina and hybrid, reading those JSON files (CPU)

Writes <out>/retrieval_bm25_only.csv (BM25 only) or <out>/retrieval_bm25.csv (with --hybrid) and <out>/timing_<tag>.json.
"""
import argparse, json, os, re, time, tracemalloc
import numpy as np, pandas as pd
from scipy import sparse
from common import load_data, retrieval_metrics

STOP = set("""i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself
it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being
have has had having do does did doing a an the and but if or because as until while of at by for with about against between
into through during before after above below to from up down in out on off over under again further then once here there
when where why how all any both each few more most other some such no nor not only own same so than too very s t can will
just don should now""".split())
TOKEN = re.compile(r"[a-z0-9]+")
MISS = 10 ** 6
TOP = 100


def tokenize(text):
    return [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        vocab, rows, cols, vals = {}, [], [], []
        dl = np.zeros(len(texts))
        for i, t in enumerate(texts):
            counts = {}
            for w in tokenize(t):
                counts[w] = counts.get(w, 0) + 1
            dl[i] = sum(counts.values())
            for w, c in counts.items():
                rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); vals.append(c)
        tf = sparse.coo_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float64)
        n_docs = len(texts)
        df = np.bincount(tf.col, minlength=len(vocab))
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean())
        w = self.idf[tf.col] * tf.data * (k1 + 1) / denom
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).tocsc()      # docs x vocabulary
        self.vocab = vocab

    def score(self, question):
        ids = sorted({self.vocab[w] for w in tokenize(question) if w in self.vocab})
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel() if ids else np.zeros(self.W.shape[0])

    def size_mb(self):
        return (self.W.data.nbytes + self.W.indices.nbytes + self.W.indptr.nbytes) / 1e6


def top_pmids(scores, pmids, k=TOP):
    order = np.argsort(-scores, kind="stable")[:k]
    return [pmids[i] for i in order]


def rrf(rankings, k=60):
    """rankings: lists of pmids, best first. Reciprocal rank fusion, equal weights; ties broken by the first list."""
    score = {}
    for lst in rankings:
        for r, p in enumerate(lst, 1):
            score[p] = score.get(p, 0.0) + 1.0 / (k + r)
    first = {p: r for r, p in enumerate(rankings[-1])}              # the dense ranking
    return sorted(score, key=lambda p: (-score[p], first.get(p, MISS)))[:TOP]


def rank_in(lst, gold):
    return lst.index(gold) + 1 if gold in lst else MISS


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--out", default="../results/bm25")
    ap.add_argument("--hybrid", action="store_true", help="also evaluate jina and the BM25 + jina hybrid (reads <out>/dense_top100_*.json from 17b)")
    ap.add_argument("--tag", default="local", help="label for the timing file (the machine the timing was measured on)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = args.data_dir
    samples = {"development": f"{D}/sample.jsonl", "held-out": f"{D}/holdout.jsonl"}

    _, corpus = load_data(f"{D}/sample.jsonl", f"{D}/corpus.jsonl", f"{D}/corpus_distractors.jsonl")
    pmids = corpus.pmid.astype(str).tolist()
    print(f"{len(corpus)} documents")

    tracemalloc.start()
    t0 = time.time()
    bm25 = BM25(corpus.text.tolist())
    build_s = time.time() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    print(f"BM25 index: {build_s:.2f} s, index {bm25.size_mb():.1f} MB, peak Python memory while building {peak_mb:.1f} MB, vocabulary {len(bm25.vocab)}")

    rows, latency = [], {}
    for name, path in samples.items():
        s, _ = load_data(path, f"{D}/corpus.jsonl")
        qs, gold, para = s.question.tolist(), s.pmid.astype(str).tolist(), s.is_paraphrased.to_numpy()
        t0 = time.time()
        lists = {"BM25": [top_pmids(bm25.score(q), pmids) for q in qs]}
        latency[name] = (time.time() - t0) / len(qs) * 1000
        if args.hybrid:
            d = json.load(open(f"{args.out}/dense_top100_{name}.json"))
            assert d["qids"] == [int(q) for q in s.qid], "dense lists were computed for different questions"
            lists["jina (dense)"] = d["pmids"]
            lists["hybrid (RRF)"] = [rrf([b, d]) for b, d in zip(lists["BM25"], lists["jina (dense)"])]
        for method, ll in lists.items():
            ranks = np.array([rank_in(l, g) for l, g in zip(ll, gold)])
            for subset, mask in (("all", np.ones(len(qs), bool)), ("original", ~para), ("paraphrased", para)):
                m = retrieval_metrics(ranks[mask])
                rows.append({"sample": name, "subset": subset, "method": method, "n": int(mask.sum()),
                             "Recall@1": m["Recall@1"], "Recall@5": m["Recall@5"], "Recall@10": m["Recall@10"],
                             "MRR": m["MRR"], "nDCG@10": m["nDCG@10"]})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(f"{args.out}/{'retrieval_bm25' if args.hybrid else 'retrieval_bm25_only'}.csv", index=False)
    json.dump({"machine": args.tag, "documents": len(corpus), "build_seconds": build_s, "index_mb": bm25.size_mb(),
               "peak_python_mb_while_building": peak_mb, "latency_ms_per_query": latency},
              open(f"{args.out}/timing_{args.tag}.json", "w"), indent=1)
    show = res[res.subset == "all"].drop(columns="subset")
    print(show.round(3).to_string(index=False))
    print("per-subset Recall@1:\n", res.pivot_table(index=["sample", "method"], columns="subset", values="Recall@1").round(3).to_string())
    print("latency ms/query:", {k: round(v, 2) for k, v in latency.items()})
