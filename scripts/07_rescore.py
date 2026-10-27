"""Re-score saved generations with the current cleaning rules (no GPU, no regeneration).

Reads ../traces/<name>.jsonl (needs the `raw` field), re-cleans and re-scores against the
gold answers from the sample, prints before/after, writes ../results/<name>_rescored.json
and overwrites the cleaned fields in ../traces/<name>.jsonl (raw output is kept).

Usage: python 07_rescore.py baseline_none baseline_vanilla_rag
"""
import json, sys
import pandas as pd
from common import load_data, retrieval_metrics
from qa_utils import score_prediction, aggregate

sample, _ = load_data()
golds = {int(r.qid): list(dict.fromkeys(list(r.golden_answers) + [r.answer]))
         for r in sample.itertuples()}
for name in sys.argv[1:]:
    path = f"../traces/{name}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    before = pd.DataFrame(rows)
    for r in rows:
        r.update(score_prediction(r["raw"], golds[r["qid"]]))
    after = pd.DataFrame(rows)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = aggregate(after)
    if "gold_rank" in after:
        res["retrieval"] = retrieval_metrics(after.gold_rank.to_numpy())
    json.dump(res, open(f"../results/{name}_rescored.json", "w"), indent=1)
    cols = ["EM", "Contains", "ROUGE-1", "ROUGE-L"]
    print(f"\n{name}  (n={len(after)}, reasoning-leak outputs: {int(after.leak.sum())})")
    print(pd.DataFrame({"before": before[cols].mean(), "after": after[cols].mean()}).round(3).T.to_string())
