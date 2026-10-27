"""Compare dense retrievers on the 500-question sample (raw question as query).

Builds a FAISS index per embedding model and reports Recall@k / MRR / MAP /
nDCG@k overall and per category. Saves indices, embeddings and per-query ranks.

Usage: python 03_retrieval_eval.py --models pubmedbert jina bge-m3
"""
import argparse, json, os, time
import faiss, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from common import load_data, rank_of_gold, retrieval_metrics

MODELS = {
    "pubmedbert": dict(name="NeuML/pubmedbert-base-embeddings", kw={}, q={}, d={}),
    "jina": dict(name="jinaai/jina-embeddings-v5-text-small",
                 kw=dict(trust_remote_code=True),
                 q=dict(task="retrieval", prompt_name="query"),
                 d=dict(task="retrieval", prompt_name="document")),
    "bge-m3": dict(name="BAAI/bge-m3", kw={}, q={}, d={}),
}

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["pubmedbert", "jina"])
p.add_argument("--out", default="../results")
p.add_argument("--index_dir", default="../index")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--distractors", default=None,
               help="jsonl of extra abstracts to add to the index, e.g. ../data/corpus_distractors.jsonl")
p.add_argument("--tag", default="", help="suffix for output files, e.g. _5k")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs(args.index_dir, exist_ok=True)

sample, corpus = load_data(distractor_path=args.distractors)
corpus.to_json(f"{args.index_dir}/corpus{args.tag}.jsonl", orient="records", lines=True, force_ascii=False)
pmid2idx = {p_: i for i, p_ in enumerate(corpus.pmid)}
gold_idx = sample.pmid.map(pmid2idx).to_numpy()
assert not np.isnan(gold_idx.astype(float)).any(), "some gold PMIDs missing from corpus"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(sample)} queries, {len(corpus)} docs, device={dev}")

summary = {}
for key in args.models:
    cfg = MODELS[key]
    t0 = time.time()
    # fp16 on GPU for the big model (Turing GPUs have no native bf16)
    mkw = {"torch_dtype": torch.float16} if (key == "jina" and dev == "cuda") else {}
    model = SentenceTransformer(cfg["name"], device=dev, model_kwargs=mkw, **cfg["kw"])
    enc = lambda texts, extra: model.encode(
        texts, batch_size=args.batch, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False, **extra).astype("float32")
    D = enc(corpus.text.tolist(), cfg["d"])
    Q = enc(sample.question.tolist(), cfg["q"])

    index = faiss.IndexFlatIP(D.shape[1])
    index.add(D)
    faiss.write_index(index, f"{args.index_dir}/{key}{args.tag}.faiss")
    np.save(f"{args.index_dir}/{key}{args.tag}_docs.npy", D)

    scores = Q @ D.T  # exact; identical to index.search over the full corpus
    ranks = rank_of_gold(scores, gold_idx)
    m = retrieval_metrics(ranks)
    m["encode_seconds"] = round(time.time() - t0, 1)
    summary[key] = m
    print(key, json.dumps(m, indent=1))

    df = pd.DataFrame({"qid": sample.qid, "cat_num": sample.cat_num, "rank": ranks})
    df.to_csv(f"{args.out}/retrieval_ranks_{key}{args.tag}.csv", index=False)
    per_cat = {int(c): retrieval_metrics(g["rank"].to_numpy())
               for c, g in df.groupby("cat_num")}
    json.dump(per_cat, open(f"{args.out}/retrieval_percat_{key}{args.tag}.json", "w"), indent=1)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

json.dump(summary, open(f"{args.out}/retrieval_summary{args.tag}.json", "w"), indent=1)
print(pd.DataFrame(summary).T.round(3).to_string())
