"""Held-out question set (never used for prompt, agent or guardrail development).

The 4,500 distractor abstracts of the 5,000-paper index come from PaperSearchQA itself, so each of them has questions.
We draw 500 of those questions (one per PMID, same per-category counts as the development sample, different seed).
Their gold abstracts are ALREADY in the index, so no new fetching or indexing is needed: the held-out systems search the
same 5,000-paper index, and the 500 development papers now act as distractors.

Run on giano after the data exists:  python 15_sample_holdout.py        -> ../data/holdout.jsonl
"""
import argparse, json
import pandas as pd
from datasets import load_dataset
from common import read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--out", default="../data/holdout.jsonl")
args = p.parse_args()

dev = pd.DataFrame(read_jsonl("../data/sample.jsonl"))
dist = pd.DataFrame(read_jsonl("../data/corpus_distractors.jsonl"))
dist = dist[dist.abstract.notna() & (dist.abstract.str.len() > 0)]
in_index = set(dist.pmid) - set(dev.pmid)

df = load_dataset("jmhb/PaperSearchQA", split="train").to_pandas()
df = df[df.pmid.isin(in_index)].copy()
df["cat_num"] = df["cat_num"].astype(int)
df = df.sample(frac=1, random_state=args.seed).drop_duplicates("pmid")     # one question per PMID
avail = df.cat_num.value_counts()
want = dev.cat_num.value_counts()                                          # same category mix as the dev sample
alloc = {c: int(min(want.get(c, 0), avail.get(c, 0))) for c in want.index}
short = len(dev) - sum(alloc.values())                                     # top up from categories with spare questions
while short > 0:
    spare = {c: avail.get(c, 0) - alloc[c] for c in alloc if avail.get(c, 0) > alloc[c]}
    if not spare:
        break
    c = max(spare, key=spare.get); alloc[c] += 1; short -= 1

parts = [g.head(alloc[c]) for c, g in df.groupby("cat_num") if alloc.get(c, 0)]
hold = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
hold.insert(0, "qid", range(len(hold)))
assert not set(hold.pmid) & set(dev.pmid), "held-out questions overlap the development sample"
assert hold.pmid.is_unique
hold.to_json(args.out, orient="records", lines=True, force_ascii=False)
print(f"held-out: {len(hold)} questions, {hold.pmid.nunique()} papers (all already in the 5k index) -> {args.out}")
print(pd.DataFrame({"dev": dev.cat_num.value_counts().sort_index(), "holdout": hold.cat_num.value_counts().sort_index()}).to_string())
