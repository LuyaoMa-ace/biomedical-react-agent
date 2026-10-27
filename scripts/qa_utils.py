"""Answer cleaning and answer-quality metrics (EM, ROUGE-1, ROUGE-L, contains).

Raw LLM output is cleaned first (strip "Answer:", markdown, keep one line), then both
prediction and gold answers go through SQuAD-style normalisation before comparison.
Scores are the max over all gold answers.
"""
import re, string
import numpy as np
import pandas as pd

_MARKER = re.compile(r"(?i)\b(?:final\s+)?answer(?:\s+is)?\s*[:\-]\s*(.*)$")
_LEAK = re.compile(r"(?i)^\s*thought\b")


def has_leak(raw):
    """MedGemma sometimes starts a chain of thought ("thought\\n1. ...") instead of answering."""
    return bool(_LEAK.match(raw or ""))


def _strip(a):
    a = re.sub(r"(?i)^(a|answer)\s*[:\-]\s*", "", a.strip()).strip(" \t\"'`")
    return a.rstrip(".").strip()


def clean_answer(text):
    """Turn a raw generation into a short answer phrase.

    - cut at a self-generated follow-up question ("\\nQ:")
    - chain-of-thought leaks ("thought ...") -> empty answer
    - "...the answer is: X" / "Answer: X" markers (X may be on the next line)
    - otherwise the first non-empty line
    """
    if not text or has_leak(text):
        return ""
    text = re.split(r"\n\s*Q\s*:", text.replace("**", ""))[0]
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    for i, l in enumerate(lines):
        m = _MARKER.search(l)
        if m:
            rest = m.group(1).strip()
            if rest:
                return _strip(rest)
            if i + 1 < len(lines):
                return _strip(lines[i + 1])
    return _strip(lines[0])


def normalize(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _variants(s):
    """Normalised string plus the same string with parenthetical text removed
    (so "Short-chain fatty acids (SCFAs)" also matches "short-chain fatty acids")."""
    out = [normalize(s), normalize(re.sub(r"\([^)]*\)", " ", s or ""))]
    return [v for v in dict.fromkeys(out) if v]


def _golds(golds):
    return [g for g in dict.fromkeys(v for x in golds for v in _variants(x)) if g]


def exact_match(pred, golds):
    gs = set(_golds(golds))
    return float(any(v in gs for v in _variants(pred)))


def contains(pred, golds):
    """Supplementary: a gold answer appears as a whole phrase inside the prediction."""
    p = f" {normalize(pred)} "
    return float(any(f" {g} " in p for g in _golds(golds)))


_scorer = None


def rouge(pred, golds):
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer
        _scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    p = normalize(pred)
    best1 = bestL = 0.0
    for g in _golds(golds):
        if not p:
            break
        s = _scorer.score(g, p)
        best1 = max(best1, s["rouge1"].fmeasure)
        bestL = max(bestL, s["rougeL"].fmeasure)
    return best1, bestL


def score_prediction(raw, golds):
    pred = clean_answer(raw)
    r1, rl = rouge(pred, golds)
    return {"pred": pred, "EM": exact_match(pred, golds), "Contains": contains(pred, golds),
            "ROUGE-1": r1, "ROUGE-L": rl, "leak": float(has_leak(raw))}


ANSWER_METRICS = ["EM", "Contains", "ROUGE-1", "ROUGE-L"]
DIAGNOSTICS = ["leak"]
EFFICIENCY_METRICS = ["tool_calls", "steps", "latency_s"]


def aggregate(df):
    """Overall and per-category means. df needs cat_num + metric columns."""
    cols = [c for c in ANSWER_METRICS + EFFICIENCY_METRICS + DIAGNOSTICS if c in df.columns]
    overall = {c: float(df[c].mean()) for c in cols}
    overall["n"] = int(len(df))
    per_cat = {int(c): {**{m: float(g[m].mean()) for m in cols}, "n": int(len(g))}
               for c, g in df.groupby("cat_num")}
    return {"overall": overall, "per_category": per_cat}
