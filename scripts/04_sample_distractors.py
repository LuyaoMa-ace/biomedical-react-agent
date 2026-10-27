"""Pick distractor PMIDs (not in the 500-question sample) from the full dataset.

Writes a jsonl of {"pmid": ...} that 02_fetch_abstracts.py can consume:
  python 04_sample_distractors.py --n 4500
  python 02_fetch_abstracts.py --sample ../data/distractor_pmids.jsonl --out ../data/corpus_distractors.jsonl
"""
import argparse, json
from datasets import load_dataset
from common import read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=4500)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="../data/distractor_pmids.jsonl")
args = p.parse_args()

gold = {r["pmid"] for r in read_jsonl("../data/sample.jsonl")}
pmids = sorted(set(load_dataset("jmhb/PaperSearchQA", split="train").to_pandas().pmid) - gold)
import random
random.Random(args.seed).shuffle(pmids)
pick = pmids[:args.n]
with open(args.out, "w") as f:
    for x in pick:
        f.write(json.dumps({"pmid": x}) + "\n")
print(f"{len(pmids)} candidate PMIDs outside the sample; wrote {len(pick)} -> {args.out}")
