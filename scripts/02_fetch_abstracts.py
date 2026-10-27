"""Fetch title/abstract/metadata for every PMID in the sample via metapub.

Resumable: PMIDs already present in the output file are skipped.
Needs NCBI_API_KEY in the environment.

Usage: python 02_fetch_abstracts.py --sample ../data/sample.jsonl --out ../data/corpus.jsonl
"""
import argparse, json, os, time
from metapub import PubMedFetcher

p = argparse.ArgumentParser()
p.add_argument("--sample", default="../data/sample.jsonl")
p.add_argument("--out", default="../data/corpus.jsonl")
p.add_argument("--retries", type=int, default=3)
args = p.parse_args()

pmids = [json.loads(l)["pmid"] for l in open(args.sample)]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["pmid"] for l in open(args.out)}
todo = [x for x in pmids if x not in done]
print(f"{len(pmids)} PMIDs, {len(done)} already fetched, {len(todo)} to go")

fetch = PubMedFetcher()
missing = 0
with open(args.out, "a") as f:
    for i, pmid in enumerate(todo, 1):
        rec = None
        for attempt in range(args.retries):
            try:
                a = fetch.article_by_pmid(pmid)
                rec = {"pmid": pmid, "title": a.title, "abstract": a.abstract,
                       "journal": a.journal, "year": a.year,
                       "authors": list(a.authors or [])}
                break
            except Exception as e:
                print(f"  {pmid} attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        if rec is None or not rec["abstract"]:
            missing += 1
            rec = rec or {"pmid": pmid, "title": None, "abstract": None,
                          "journal": None, "year": None, "authors": []}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        if i % 25 == 0:
            print(f"{i}/{len(todo)} fetched, {missing} without abstract")
print("done; PMIDs without abstract:", missing)
