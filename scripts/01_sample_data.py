"""Stratified sample of PaperSearchQA, one question per PMID.

Usage: python 01_sample_data.py --n 500 --seed 42 --out ../data/sample.jsonl
"""
import argparse, json, os
import pandas as pd
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=500)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--min_per_cat", type=int, default=0,
               help="minimum questions per category (capped by availability)")
p.add_argument("--out", default="../data/sample.jsonl")
args = p.parse_args()

df = load_dataset("jmhb/PaperSearchQA", split="train").to_pandas()
print("full dataset:", len(df), "rows,", df.pmid.nunique(), "unique PMIDs")

# one question per PMID, so the index has no duplicate documents
df = df.sample(frac=1, random_state=args.seed).drop_duplicates("pmid")
df["cat_num"] = df["cat_num"].astype(int)

avail = df.cat_num.value_counts()
print("available per category (one question per PMID):")
print(avail.sort_index().to_string())

# proportional allocation per category (largest-remainder rounding)
share = df.cat_num.value_counts(normalize=True) * args.n
alloc = share.astype(int)
rest = args.n - alloc.sum()
alloc[(share - alloc).sort_values(ascending=False).index[:rest]] += 1

# floor for small categories, paid for by trimming the largest ones
if args.min_per_cat:
    alloc = alloc.clip(lower=0)
    for c in alloc.index:
        alloc[c] = min(max(alloc[c], args.min_per_cat), avail[c])
    while alloc.sum() > args.n:
        big = alloc.idxmax()
        alloc[big] -= 1
    print("allocation with floor:", alloc.sort_index().to_dict())

parts = [g.head(alloc[c]) for c, g in df.groupby("cat_num")]
sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
sample.insert(0, "qid", range(len(sample)))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
sample.to_json(args.out, orient="records", lines=True, force_ascii=False)
print("saved", len(sample), "->", args.out)
print(sample.groupby(["cat_num", "cat"]).size())
