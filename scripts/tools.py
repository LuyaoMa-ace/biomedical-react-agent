"""Agent tools. Currently: retrieve_from_index (local dense retrieval).

Other tools (query_reformulator, generate_reasoning_query, fetch_pubmed_live,
query_openfda) will be added to this module.
"""
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer


class LocalRetriever:
    """FAISS index over the corpus, queried with jina-embeddings-v5 (query prompt)."""

    def __init__(self, index_dir="../index", tag="_5k", key="jina", device=None):
        """device: "cuda" / "cpu" / None (auto). Use "cpu" on 11GB GPUs so MedGemma has the whole card."""
        self.docs = [json.loads(l) for l in open(f"{index_dir}/corpus{tag}.jsonl")]
        self.index = faiss.read_index(f"{index_dir}/{key}{tag}.faiss")
        assert self.index.ntotal == len(self.docs), "index and corpus are out of sync"
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"torch_dtype": torch.float16} if dev == "cuda" else {}
        self.enc = SentenceTransformer("jinaai/jina-embeddings-v5-text-small", device=dev,
                                       trust_remote_code=True, model_kwargs=kw)

    def search_batch(self, queries, k=5):
        q = self.enc.encode(queries, task="retrieval", prompt_name="query",
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype("float32")
        scores, ids = self.index.search(q, k)
        return [[{"rank": r + 1, "score": float(s), **{f: self.docs[i][f] for f in
                  ("pmid", "title", "abstract")}}
                 for r, (s, i) in enumerate(zip(ss, ii))] for ss, ii in zip(scores, ids)]

    def search(self, query, k=5):
        return self.search_batch([query], k)[0]


def format_docs(hits, max_chars=2000):
    """Render retrieved documents as a context block for the prompt."""
    return "\n\n".join(f"[{h['rank']}] {h['title']}\n{(h['abstract'] or '')[:max_chars]}"
                       for h in hits)


# ---------------------------------------------------------------------------
# LLM-prompt tools (no retrieval inside): they only rewrite the query.
# ---------------------------------------------------------------------------
import os, re, time

REFORMULATE_PROMPT = (
    "Rewrite the biomedical question below as a concise keyword-style search query for "
    "retrieving the PubMed abstract that answers it. Remove the question wording, expand "
    "acronyms, and make implicit concepts explicit. Output ONLY the query, on one line.\n\n"
    "Question: {q}")

REASONING_PROMPT = (
    "Think briefly about the biomedical question below: which concepts, mechanisms, "
    "populations or methods it involves, and what an abstract that answers it would say. "
    "Write 3-4 plain sentences of reasoning. Do NOT restate or paraphrase the question "
    "(no \"The question asks...\"); write the domain concepts and terminology the relevant "
    "abstract would contain. Do NOT state a final answer.\n\n"
    "Question: {q}")


def _one_line(text, fallback):
    t = re.sub(r"\s+", " ", (text or "").replace("**", "")).strip(" \"'`")
    t = re.sub(r"(?i)^(search\s+)?query\s*:\s*", "", t).strip()
    return t or fallback


def query_reformulator(llm, questions):
    """Tool 1. LLM prompt -> retrieval-friendly query per question (list[str])."""
    outs = llm.generate([REFORMULATE_PROMPT.format(q=q) for q in questions],
                        max_new_tokens=48, prefill="Query:")
    return [_clean_keywords(_one_line(o.split("\n")[0], q)) or q for o, q in zip(outs, questions)]


def _clean_keywords(s):
    """Drop quotes and boolean operators the LLM sometimes writes (PubMed ANDs plain terms)."""
    s = re.sub(r"\s+(AND|OR|NOT)\s+", " ", s.replace('"', " "))
    return re.sub(r"\s+", " ", s).strip()


def generate_reasoning_query(llm, questions):
    """Tool 2. LLM writes a short reasoning trace; the trace (not the question) is the
    retrieval query (Reason-ModernColBERT / BRIGHT idea). Falls back to the question."""
    outs = llm.generate([REASONING_PROMPT.format(q=q) for q in questions],
                        max_new_tokens=160, prefill="Reasoning:")
    return [_one_line(o, q) for o, q in zip(outs, questions)]


# ---------------------------------------------------------------------------
# External tools (cached on disk so reruns are fast, reproducible and rate-limit friendly)
# ---------------------------------------------------------------------------
class _Cache:
    def __init__(self, path):
        import threading
        self.lock = threading.Lock()
        self.path, self.data = path, {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            for l in open(path):
                r = json.loads(l)
                self.data[r["key"]] = r["value"]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")


_STOP = set("a an the of in on at to for and or with by from is are was were which what who whom how "
            "when where why that this these those as its their be been into than then also using "
            "used use does do did has have had after before during between within".split())


def _keywords(text, n=8):
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text) if w.lower() not in _STOP]
    return " ".join(words[:n])


class PubMedLive:
    """Tool 4. fetch_pubmed_live: search PubMed via NCBI E-utilities (metapub), return top papers.

    Needs NCBI_API_KEY in the environment (10 req/s instead of 3). Keyword search, so pass
    a keyword-style query (that is what query_reformulator produces)."""

    def __init__(self, cache_path="../cache/pubmed_live.jsonl", retmax=3):
        from metapub import PubMedFetcher
        self.fetch, self.retmax = PubMedFetcher(), retmax
        self.cache = _Cache(cache_path)

    def _search(self, query):
        pmids = self.fetch.pmids_for_query(query, retmax=self.retmax)
        docs = []
        for pmid in pmids:
            try:
                a = self.fetch.article_by_pmid(pmid)
                if a.abstract:
                    docs.append({"pmid": str(pmid), "title": a.title, "abstract": a.abstract})
            except Exception:
                continue
        return docs

    def __call__(self, query):
        key = f"{self.retmax}|{query}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        docs, err = [], None
        # No keyword-trimming fallback: PubMed ANDs all terms, so a long natural-language query
        # returns nothing, and trimmed queries returned unrelated papers. An honest empty result
        # lets the agent retry with better keywords.
        for q in (query,):
            for attempt in range(3):
                try:
                    docs = self._search(q)
                    err = None
                    break
                except Exception as e:
                    err = repr(e)[:120]
                    time.sleep(2 ** attempt)
            if docs:
                break
        if err is None:  # do not cache transient failures
            self.cache.put(key, docs)
        return docs


class OpenFDA:
    """Tool 5. query_openfda: drug label facts (indications, mechanism, adverse reactions, ...).

    Input is a drug name (generic or brand). Unknown drugs return an empty dict (HTTP 404)."""

    FIELDS = ("indications_and_usage", "mechanism_of_action", "adverse_reactions",
              "warnings", "description")

    def __init__(self, cache_path="../cache/openfda.jsonl", max_chars=700):
        import requests
        self.http, self.max_chars = requests, max_chars
        self.cache = _Cache(cache_path)

    def _label(self, field, drug):
        r = self.http.get("https://api.fda.gov/drug/label.json",
                          params={"search": f'openfda.{field}:"{drug}"', "limit": 1}, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()["results"][0]

    def __call__(self, drug):
        drug = (drug or "").strip().strip(".")
        if not drug or drug.lower() in ("none", "n/a", "no drug"):
            return {}
        key = drug.lower()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        res, err = None, None
        for field in ("generic_name", "brand_name", "substance_name"):
            try:
                res = self._label(field, drug)
            except Exception as e:
                err = repr(e)[:120]
                break
            if res:
                break
        out = {}
        if res:
            for f in self.FIELDS:
                v = res.get(f)
                if v:
                    out[f] = re.sub(r"\s+", " ", v[0])[:self.max_chars]
        if err is None:
            self.cache.put(key, out)
        return out


EXTRACT_DRUG_PROMPT = (
    "Does the biomedical question below mention a specific medicine used in humans (an approved "
    "or investigational drug)? Toxins, pesticides, laboratory reagents and chemical classes "
    "are NOT drugs. Reply with ONLY the generic drug name (one name), or NONE.\n\n"
    "Question: {q}")


def extract_drug(llm, questions):
    outs = llm.generate([EXTRACT_DRUG_PROMPT.format(q=q) for q in questions],
                        max_new_tokens=12, prefill="Drug:")
    res = []
    for o in outs:
        d = _one_line(o.split("\n")[0], "NONE")
        res.append("" if d.upper().startswith("NONE") or len(d) > 60 else d)
    return res


def format_fda(label):
    return "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in label.items())
