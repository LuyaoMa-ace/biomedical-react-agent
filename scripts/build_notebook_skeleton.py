"""Generate notebook/PaperSearchQA_ReAct_Agent.ipynb (skeleton) in the style of the lecture notebooks.

Run from the repo root:  python scripts/build_notebook_skeleton.py
Design: live demos use a few questions and are guarded by LIVE (needs a GPU); every table/figure reads the
precomputed files in results/ and traces/, so the notebook also runs on a laptop. Cells marked TODO are
filled once the final numbers are in.
"""
import json, textwrap, pathlib, uuid

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": textwrap.dedent(s).strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [],
                  "source": textwrap.dedent(s).strip("\n").splitlines(True)})


# --------------------------------------------------------------------------------------------- title
md("""
    # **Multi-tool ReAct agents for biomedical question answering**

    In this notebook we build and study an LLM agent that answers biomedical questions by deciding **which tool to call and when**,
    and we ask a simple question: *does dynamic orchestration really beat a fixed retrieval pipeline?*

    We will tackle the following tasks:
    > 1. Prepare **PaperSearchQA**, a QA dataset built on PubMed abstracts, and a local search index with several embedding models.
    > 2. Build a no-tool baseline and four **fixed chains** (static RAG).
    > 3. Implement **five tools** (local retrieval, live PubMed, OpenFDA, query reformulator, reasoning query).
    > 4. Build a **ReAct agent with LangGraph** and compare it with the baselines (experiments 1 and 2).
    > 5. **Read the traces**: tool choices per question category, failure modes, and why the agent behaves as it does (experiment 3).
""")
md("""
    📝 Luyao Ma  ·  project proposal by Lorenzo Molfetta  ·  *Big Data Analytics and Text Mining*, University of Bologna
""")
md("""
    ## 📜 Outline
    1. Set up: packages, secrets, GPU
    2. Dataset: PaperSearchQA
    3. Local retrieval and the embedding models
    3b. Two classical baselines: BM25 and a BM25 + dense hybrid
    4. The LLM, the answer prompt and the metrics
    5. Baselines: no tools and fixed chains
    6. Tool definition: five tools
    7. The ReAct agent in LangGraph
    8. Experiment 1: which tool contributes?
    9. Experiment 2: fixed chain vs dynamic orchestrator
    10. Experiment 3: what does the agent actually do? (trace analysis)
    11. Granite vs MedGemma as the scheduler
    12. An extra look: paraphrased questions (12b prompt sensitivity, 12c fixed-chain ranking, 12d held-out validation, 12e coverage experiment)
    13. Discussion, limitations and related work
    14. Reproducibility
""")

# --------------------------------------------------------------------------------------------- 1 setup
md("""
    ## ⚙️ 1. Set Up: Packages, Secrets, GPU

    > **Where is the data?** This notebook expects the repository layout (`scripts/`, `data/`, `results/`, `traces/`).
    > On Colab, upload `bigdata_project.zip` and run the unzip cell below. All **tables and figures are read from `results/`**,
    > so only the small *live demos* need a GPU (switch `LIVE`).
""")
code("""
    # Unzip the submission archive if we are on Colab; go to the project root if opened from notebook/
    import os, zipfile, pathlib
    if pathlib.Path("bigdata_project.zip").exists() and not pathlib.Path("scripts").exists():
        zipfile.ZipFile("bigdata_project.zip").extractall(".")
        os.chdir("bigdata_project")
    if not pathlib.Path("scripts").exists() and pathlib.Path("../scripts").exists():   # opened from notebook/
        os.chdir("..")
    print("project folder:", pathlib.Path.cwd().name, "|", sorted(os.listdir("."))[:8])
""")
code("""
    # 📦 Packages. Pinned: the ones Colab does not ship (exact versions of everything, torch 2.11.0 and numpy/pandas/scipy/matplotlib included, are in env/requirements.txt).
    # numpy / pandas / scipy / matplotlib / torch are left at the preinstalled versions to avoid conflicts; the notebook was NOT tested on Colab, only on a laptop (results-only mode).
    %pip install -q transformers==5.20.2 sentence-transformers==6.2.0 faiss-cpu==1.15.1 langgraph==1.2.16 langchain-core==1.7.0 metapub==0.7.5 datasets==5.1.0 rouge-score==0.1.2
""")
md("""
    > **RESTART** the runtime after the installation, then continue from the next cell.
""")
code("""
    import json, gzip, re, sys, warnings
    from pathlib import Path
    import numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Image, Markdown, display
    warnings.filterwarnings("ignore")

    ROOT = Path(".")
    sys.path.insert(0, str(ROOT / "scripts"))          # our modules: qa_utils, prompts, tools, agent, ...
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    def have(p):
        ok = (ROOT / p).exists()
        if not ok: print(f"(not found: {p}) -> run the step described in README.md")
        return ok
""")
md("""
    We use **open-source models from Hugging Face**. MedGemma is a *gated* model: accept its license on the model page and paste your key
    into the Colab `Secrets` 🔑 (`HF_TOKEN`). `NCBI_API_KEY` (free) makes PubMed 3× faster. Never write keys in the notebook.
""")
code("""
    import os
    for name in ("HF_TOKEN", "NCBI_API_KEY"):
        try:
            from google.colab import userdata
            os.environ[name] = userdata.get(name)         # Colab Secrets
        except Exception:
            pass                                          # locally / on the cluster: use environment variables
    print({k: ("set" if os.environ.get(k) else "missing") for k in ("HF_TOKEN", "NCBI_API_KEY")})
""")
md("""
    ### 🖥️ GPU and numeric precision

    MedGemma must run in **bfloat16**: in float16 it overflows and returns *empty* answers (we measured this).
    Turing GPUs (Colab **T4**, RTX 2080 Ti) have no native bf16 but run it through emulation: slower (about half the speed), yet correct.
""")
code("""
    LIVE = False
    try:
        import torch
    except ImportError:
        torch = None
        print("torch is not installed: live demos disabled")
    if torch is not None and torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(torch.cuda.get_device_name(0), f"| {mem:.0f} GB | bf16 native:", torch.cuda.get_device_capability(0)[0] >= 8)
        LIVE = mem >= 15          # MedGemma-4B needs ~9 GB of weights + cache; the retriever model fits next to it
    print("LIVE demos enabled:", LIVE)
""")

# --------------------------------------------------------------------------------------------- 2 dataset
md("""
    # 📚 2. Dataset: PaperSearchQA

    **PaperSearchQA** (Burgess et al., EACL 2026) contains factoid questions generated by GPT-4.1 from PubMed abstracts
    (three questions per abstract, ten expert-defined categories in the paper (our HuggingFace copy has eleven, `cat_num` 0-10, where 0 = "Other"), 50 % of the questions paraphrased to reduce keyword overlap).
    The dataset gives the *question, answer, synonyms, PMID, category* but **not the paper text**: we fetch the abstracts with `metapub`.

    Our sample (`scripts/01_sample_data.py`): **500 questions**, seed 42, one question per PMID, proportional per category with a floor of 20.
""")
code("""
    if have("data/sample.jsonl"):
        sample = pd.read_json(ROOT / "data" / "sample.jsonl", lines=True)
        print(len(sample), "questions |", sample.pmid.nunique(), "distinct papers |", f"{sample.is_paraphrased.mean():.0%} paraphrased")
        display(sample.groupby(["cat_num", "cat"]).size().rename("questions").to_frame())
        display(sample[["question", "answer", "cat", "is_paraphrased"]].head(5))
""")
md("""
    ### 🧾 The corpus: gold abstracts and distractors

    The local index holds the 500 gold abstracts **plus 4,500 distractor abstracts** (other papers of the same dataset).
    Why? With only the 500 gold papers, retrieval is almost perfect (Recall@10 = 0.99) and no tool can show an effect.
    We therefore measure everything on the 5,000-document index.
""")
code("""
    if have("data/corpus.jsonl") and have("data/corpus_distractors.jsonl"):
        gold = pd.read_json(ROOT / "data" / "corpus.jsonl", lines=True)
        dist = pd.read_json(ROOT / "data" / "corpus_distractors.jsonl", lines=True)
        print(len(gold), "gold abstracts +", len(dist), "distractors")
        display(gold[["pmid", "title", "journal", "year"]].head(3))
""")

# --------------------------------------------------------------------------------------------- 3 retrieval
md("""
    # 🔎 3. Local Retrieval and the Embedding Models

    An **embedding model** maps text to a vector so that texts with a similar meaning are close; a **FAISS index** stores the document vectors and
    finds the nearest ones to a query vector. We compare three models: `pubmedbert` (biomedical), `jina-embeddings-v5` (general, separate query/document prompts)
    and `bge-m3` (general). Each question has exactly **one** relevant paper, hence **MAP = MRR**.
""")
code("""
    rows = {}
    for tag, label in (("", "500 docs"), ("_5k", "5,000 docs")):
        if have(f"results/retrieval_summary{tag}.json"):
            for model, m in json.load(open(ROOT / "results" / f"retrieval_summary{tag}.json")).items():
                rows[(label, model)] = {k: v for k, v in m.items() if k != "encode_seconds"}
    if rows:
        display(pd.DataFrame(rows).T.round(3))
""")
md("""
    **Reading the table.** With 500 documents retrieval is saturated; adding 4,500 distractors separates the models:
    `jina` is the best (Recall@1 0.83 vs 0.76 for `bge-m3` and 0.69 for `pubmedbert`), so a general-purpose model beats the biomedical one here.
    All following experiments use `jina` and the 5,000-document index.

""")
md("""
    ## 3b. Two classical retrieval baselines: BM25 and a BM25 + dense hybrid

    Every experiment so far uses one very strong dense retriever (`jina`, Recall@10 0.97), which leaves little room for any orchestrator. Here we add the two classical baselines a reader would ask for and measure how much weaker retrieval can get.
    **Protocol (fixed before running, no tuning).** Same 5,000 documents (title + abstract), the raw question as the query, 500 development and 500 held-out questions.
    - **BM25**: Okapi BM25, k1 = 1.5, b = 0.75, Lucene idf; lower-cased, tokens `[a-z0-9]+`, English stop words removed, no stemming (a limitation: stemming could help BM25).
    - **Hybrid**: reciprocal rank fusion of the BM25 and `jina` rankings (top 100 of each, k = 60, equal weights).
    - **Metrics**: Recall@1/5/10, MRR, nDCG@10, overall and for original vs paraphrased questions; build time, memory and query latency (laptop CPU for BM25, so only the order of magnitude is comparable with the cluster GPU numbers of `jina`).

    | | Prediction (written before the run) |
    |---|---|
    | B1 | BM25 is clearly weaker than `jina` on the development sample: Recall@1 in [0.55, 0.75], Recall@10 in [0.82, 0.94], at least 0.03 below `jina` (0.970) |
    | B2 | BM25 suffers from paraphrasing, dense retrieval does not: Recall@1 on original minus paraphrased questions is at least 0.08 for BM25, below 0.03 in absolute value for `jina` (observed -0.009) |
    | B3 | The hybrid is at least as good as `jina` (no more than 0.01 below it on Recall@1, @5 and @10), with Recall@10 in [0.965, 0.99] and Recall@1 in [0.82, 0.88] |
    | B4 | The held-out sample shows the same ordering: BM25 Recall@10 at least 0.03 below `jina`; the hybrid no more than 0.01 below `jina` on Recall@1 and Recall@10 |
    | B5 | BM25 is far cheaper: building its index for 5,000 documents takes at most 30 s on one CPU core, uses at most 200 MB, answers a query in at most 20 ms, and is at least 10 times faster to build than the 480 s `jina` needs to encode the corpus |

    **Decision rule (fixed in advance).** Retrieval-only numbers come first. If BM25 Recall@10 on the development sample is **at most 0.93**, the weaker retriever is informative and we also run the answer chains and the agent with BM25 and the hybrid. If it is above 0.93, we stop and report only the retrieval table.
""")
code("""
    if have("results/bm25/retrieval_bm25.csv"):
        _r = pd.read_csv(ROOT / "results" / "bm25" / "retrieval_bm25.csv")
        display(_r[_r.subset == "all"].drop(columns="subset").round(3))
        print("Recall@1 by question type (original vs paraphrased):")
        display(_r[_r.subset != "all"].pivot_table(index=["sample", "method"], columns="subset", values="Recall@1").round(3))
        print("BM25 cost (laptop CPU):", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in json.load(open(ROOT / "results" / "bm25" / "timing_macbook_cpu.json")).items() if k != "latency_ms_per_query"})
""")
md("""
    **Result: B2, B4 and B5 held; B1 held in part; B3 did not hold.** (`jina` here is recomputed from its saved top-100 lists, so it differs from the earlier table by one question: 0.968 vs 0.970 Recall@10.)

    | | Prediction | Result | |
    |---|---|---|---|
    | B1 | BM25 Recall@1 in [0.55, 0.75], Recall@10 in [0.82, 0.94], at least 0.03 below `jina` | Recall@1 **0.774**, Recall@10 0.922 (0.046 below `jina`) | **Recall@1 no** (BM25 is stronger than predicted), rest yes |
    | B2 | Recall@1 original minus paraphrased: at least 0.08 for BM25, below 0.03 for `jina` | BM25 0.183 (held-out 0.134); `jina` -0.009 (+0.012) | yes |
    | B3 | the hybrid is at most 0.01 below `jina` on Recall@1/5/10; Recall@10 in [0.965, 0.99] | Recall@1 -0.002, Recall@5 **-0.022**, Recall@10 **-0.014** (0.954) | **no** |
    | B4 | held-out: same ordering | BM25 0.036 below `jina` at Recall@10; the hybrid +0.014 at Recall@1 and +0.002 at Recall@10 | yes |
    | B5 | build ≤ 30 s, ≤ 200 MB, ≤ 20 ms/query, ≥ 10 times faster than `jina`'s 480 s | 1.2 s, 42 MB peak (5.7 MB index), 0.21 ms; about 400 times faster (different hardware) | yes |

    What we learn:
    - **BM25 is a strong baseline.** It is only 4.6 Recall@10 points and 5.6 Recall@1 points below `jina` on the development sample (0.922 vs 0.968, 0.774 vs 0.830), and about the same on the held-out sample (0.920 vs 0.956, 0.802 vs 0.844). We expected a larger gap.
    - **The dense retriever's advantage lies entirely in the paraphrased questions.** On the original questions BM25 is as good or better (Recall@1 0.873 vs 0.825 on development, 0.865 vs 0.850 held-out); on the paraphrased ones it drops to 0.690 (0.731) while `jina` stays at 0.834 (0.838).
    - **An equal-weight hybrid does not reliably beat the dense retriever.** It is a little worse on development and a little better on held-out data, by a few questions each time (one question is 0.002), i.e. within the noise of 500 questions. We did not tune the fusion weights, by design.
    - **Cost:** BM25 is built in about a second on a laptop CPU with a 5.7 MB index and answers in 0.2 ms, while `jina` needs a GPU and about 480 s to encode the corpus.
    - **Decision rule:** BM25 Recall@10 on the development sample is 0.922, below the 0.93 threshold, so the weaker retriever is used for the answer chains and the agent in the next step (if completed, see below).
""")


# --------------------------------------------------------------------------------------------- 4 LLM and prompt
md("""
    # 🤖 4. The LLM, the Answer Prompt and the Metrics

    We use **`google/medgemma-1.5-4b-it`** for everything that needs a language model (answers, the two query-rewriting tools, and the agent's scheduler).
    It is loaded **locally with Transformers**, in bf16, with greedy decoding.
""")
code("""
    if LIVE:
        from llm import MedGemma
        medgemma = MedGemma(dtype="bfloat16")
        print(medgemma.generate(["Answer with one word: what is the capital of France?"], max_new_tokens=8, prefill="A:"))
    else:
        print("LIVE is off: skipping the model download (the saved results are used instead)")
""")
md("""
    ### ✍️ The answer prompt: why the model's turn is pre-filled

    Asked for a short answer, MedGemma sometimes *starts thinking* (`thought\\nThe user wants me to ...`) and never reaches the answer.
    In our first run this cost about **7 EM points**. We therefore **pre-fill the model turn with `A:`**, so it answers directly.
    Every system in this notebook uses the same prompt and the same answer cleaning, so differences come from the retrieval policy only.
""")
code("""
    from prompts import answer_prompt, PREFILL
    print(answer_prompt("Which imaging technique can identify unilateral hydrocephalus?",
                        "[1] Unilateral hydrocephalus: prenatal sonographic diagnosis. ..."))
    print("\\n--- the model turn then starts with the pre-filled text:", repr(PREFILL))
""")
md("""
    ### 📏 Metrics

    **EM** (exact match): after lower-casing, removing punctuation and articles, the prediction equals the answer or one of its synonyms.
    **Contains**: a gold answer appears inside the prediction (more tolerant). **ROUGE-1/L**: word overlap with the best gold answer.
    EM is strict: for Vanilla RAG on the development sample *Contains* is 5.6 points above EM (0.820 vs 0.764), i.e. about a quarter of the EM errors still contain a gold answer string. Most are wording differences, but not all (`L4` inside `L4 and L5` is not a correct answer), so this is an upper bound; we did not audit the errors one by one. We therefore also report Contains and ROUGE.
""")
code("""
    from qa_utils import clean_answer, score_prediction
    examples = [("**Short-chain fatty acids (SCFAs)**", ["Short-chain fatty acids"]),
                ("The answer is: MHC-I.", ["MHC-I"]),
                ("Based on the provided abstracts, the answer is:\\nDNA or RNA oligonucleotides", ["Oligonucleotides"]),
                ("thought\\nThe user wants me to identify ...", ["Multiple sclerosis"])]
    for raw, golds in examples:
        s = score_prediction(raw, golds)
        print(f"{raw[:55]!r:60} -> {s['pred']!r:42} EM={s['EM']:.0f} Contains={s['Contains']:.0f}")
""")

# --------------------------------------------------------------------------------------------- 5 baselines
md("""
    # 🪜 5. Baselines: No Tools and Fixed Chains

    The experiments are a **performance ladder**. Every rung must beat the previous one to be worth discussing:
    > **Baseline 0**: the LLM answers from its own knowledge.
    > **Fixed chains** (the same steps for every question): *Vanilla RAG* (retrieve → answer), *reformulate → retrieve → answer*,
    > *reasoning query → retrieve → answer*, *FDA-augmented RAG* (retrieve + OpenFDA → answer).
""")
code("""
    if have("results/exp_ladder.csv"):
        ladder = pd.read_csv(ROOT / "results" / "exp_ladder.csv")
        cols = ["system", "EM", "ΔEM", "Contains", "ROUGE-1", "ROUGE-L", "tool_calls", "steps", "latency_s", "R@1", "R@10", "MRR", "nDCG@10"]
        display(ladder[[c for c in cols if c in ladder.columns]].round(3))
""")
md("""
    **Reading the ladder.** One retrieval step lifts EM from about 0.30 to 0.76. Everything we add on top (query rewriting, reasoning queries, OpenFDA)
    changes the result by at most a few points, and reasoning queries are *worse* under our answer prompt v3 (-0.028 EM, p = 0.02), although not significantly so under the paraphrased prompt v4 (-0.012, p = 0.33, section 12c). Our safe claim is that they are *not better* (strong retriever + factoid questions: LLM query expansion adds noise).

""")

# --------------------------------------------------------------------------------------------- 6 tools
md("""
    # 🛠️ 6. Tool Definition

    The agent can use five tools, of three qualitatively different kinds:

    | tool | source | what it does |
    |---|---|---|
    | `retrieve_from_index` | local FAISS | top documents for a query |
    | `fetch_pubmed_live` | PubMed (NCBI E-utilities) | live search outside our corpus, **keywords only** |
    | `query_openfda` | OpenFDA drug labels | structured drug information |
    | `query_reformulator` | LLM prompt | rewrites the question as a keyword query (does **not** search) |
    | `generate_reasoning_query` | LLM prompt | writes a short reasoning trace to use as query (does **not** search) |
""")
md("""
    ### 💊 `query_openfda` and 📰 `fetch_pubmed_live`

    External answers are **cached on disk**, which makes reruns fast and reproducible. A useful detail: PubMed ANDs all terms, so a full question returns *nothing*,
    while a keyword query finds the right papers.
""")
code("""
    try:
        import tempfile, logging
        logging.getLogger("metapub").setLevel(logging.ERROR)      # silence the "NCBI_API_KEY was not set" warning
        from tools import OpenFDA, PubMedLive
        fda = OpenFDA(cache_path=tempfile.mkdtemp() + "/fda.jsonl")
        label = fda("alteplase")
        print("OpenFDA fields for alteplase:", list(label))
        pm = PubMedLive(cache_path=tempfile.mkdtemp() + "/pm.jsonl", retmax=2)
        q_full = "Which imaging technique employed during pregnancy can identify unilateral hydrocephalus in the fetus"
        q_kw = "sonography prenatal diagnosis unilateral hydrocephalus"
        print("full question ->", [d["pmid"] for d in pm(q_full)], "| keywords ->", [d["pmid"] for d in pm(q_kw)])
    except Exception as e:
        print("(tool demo skipped: it needs internet access and the packages in env/requirements.txt;", repr(e)[:80], ")")
""")
md("""
    ### 🧩 The two query-rewriting tools

    They only **transform the query**; the agent still has to call `retrieve_from_index` with the result. This is the specification we follow.
    We will see in section 10 that small models often forget this second step.

    Section 7 shows what these two tools return in real traces.
""")

# --------------------------------------------------------------------------------------------- 7 agent
md("""
    # 🕸️ 7. The ReAct Agent in LangGraph

    **ReAct** = *Reasoning + Acting*: the model writes `Thought`, then `Action` and `Action Input`, the system returns an `Observation`, and so on until `Final Answer`.
    With **LangGraph** the loop is a graph. One design choice matters:

    > The **scheduler** only decides *which tool to call and when to stop*. The **final answer** is always written by the *same answer generator*
    > (MedGemma + the fixed prompt of section 4) from the evidence the agent collected. So all systems share one answer generator.
""")
code("""
    from agent import ReActAgent, TOOL_DOCS, build_header
    stub_tools = {name: (lambda x: ("", {})) for name in TOOL_DOCS}
    agent = ReActAgent(None, stub_tools)
    graph = agent.graph.get_graph()
    print("nodes:", list(graph.nodes))
    print(graph.draw_mermaid())
""")
md("""
    ### 📋 The scheduler's prompt

    Everything goes into **one prompt** (MedGemma was not evaluated for multi-turn chat): instructions, tool list, one tiny example, the question and the scratchpad.
""")
code("""
    print(build_header(list(TOOL_DOCS)))
""")
md("""
    ### 👀 A few real traces

    A trace shows each `Thought → Action → Observation`. Traces are stored per question in `traces/`.
""")
code("""
    def load_traces(name):
        for p in (ROOT / "traces" / f"{name}.jsonl.gz", ROOT / "traces" / f"{name}.jsonl"):
            if p.exists():
                op = gzip.open if p.suffix == ".gz" else open
                with op(p, "rt") as f:
                    return [json.loads(l) for l in f]
        print(f"(no trace file for {name})"); return []

    def show_trace(r, width=110):
        print("Q:", r["question"][:width]); print("GOLD:", r["gold"], "| PRED:", r["pred"], "| EM:", r["EM"], "| tools:", r.get("tool_usage"))
        for st in r["trace"]:
            line = f"  [{st['step']}] {st['kind']}"
            if st.get("tool"): line += f" {st['tool']}({st.get('input', '')[:60]})"
            if st.get("reason"): line += f"  reason={st['reason']}"
            print(line)
            if st.get("observation"): print("       obs:", st["observation"][:90].replace("\\n", " "))

    guard_traces = load_traces("agent_full_guard")      # MedGemma scheduler + guardrail; the same two questions are discussed in section 10
    for qid in (333, 272):
        for r in guard_traces:
            if r["qid"] == qid:
                show_trace(r); print()
""")
md("""
    ### 🧩 What the two rewriting tools return
    A rewriting tool returns a *query*, not documents. The next cell reads the traces of the full agent **without** the guardrail and counts how often the agent retrieved after a rewrite;
    then it prints three questions where it did not.
""")
code("""
    agent_traces = load_traces("agent_full")
    REWRITE = ("query_reformulator", "generate_reasoning_query")
    n_rewrite = n_no_retrieve = 0
    examples = []
    for r in agent_traces:
        steps = [s for s in r["trace"] if s.get("executed")]
        for i, s in enumerate(steps):
            if s["tool"] in REWRITE:
                n_rewrite += 1
                later = [t["tool"] for t in steps[i + 1:]]
                if "retrieve_from_index" not in later:
                    n_no_retrieve += 1
                    if len(examples) < 3:
                        examples.append((r, s, later[0] if later else "(none: it answered)"))
                break
    if n_rewrite:
        print(f"{n_rewrite} of {len(agent_traces)} questions called a rewriting tool; in {n_no_retrieve} ({n_no_retrieve / n_rewrite:.0%}) the agent never called retrieve_from_index afterwards\\n")
    for r, s, nxt in examples:
        print("Q:", r["question"][:100]); print(f"   {s['tool']} returned: {s['observation'][:140]!r}"); print("   next action:", nxt, "| EM:", r["EM"]); print()
""")

# --------------------------------------------------------------------------------------------- 8-9 experiments
md("""
    # 🧪 8. Experiment 1: Which Tool Contributes?

    We remove one tool at a time and compare every configuration with the **full agent** (paired difference, bootstrap 95 % CI; McNemar test on EM).
    `index-only` keeps only `retrieve_from_index`.
""")
code("""
    if have("results/exp1_ablation.csv"):
        ab = pd.read_csv(ROOT / "results" / "exp1_ablation.csv")
        display(ab[ab.metric == "EM"].round(3))
""")
md("""
    **Reading the table.** Without a guardrail, removing tools *helps*: the index-only agent is 6.8 EM points better than the full agent (CI [+3.8, +9.8]), because the rewriting tools lead to answers without evidence (diagnosed in section 10).
    The question that matters is therefore asked on the **guardrail configuration**, where the "no evidence" failure is repaired:
""")
code("""
    if have("results/exp1b_ablation_guardrail.csv"):
        display(pd.read_csv(ROOT / "results" / "exp1b_ablation_guardrail.csv").round(3))
""")
md("""
    **Reading the table (guardrail configuration).**
    - *No single tool matters for accuracy.* Removing OpenFDA, PubMed, the reasoning tool or the reformulator changes MedGemma's EM by -0.006 to +0.014; for Granite by 0.000 to +0.008. All 95 % CIs include 0 (about ±0.025, the resolution of 500 questions).
    - OpenFDA is never used by either scheduler (0.00 calls per question), so removing it is a no-op by construction.
    - *Tools cost time, not accuracy.* Granite without the reasoning tool is as accurate (0.746 vs 0.744) at 1.67 s per question instead of 3.0 s; without the reformulator 0.752 at 1.74 s.
    - So the extra tools give no measurable benefit with a strong retriever (Recall@10 0.97), in line with findings that query expansion does not help strong retrievers (Weller et al.). An ablation can only say "no detectable effect", not "zero effect".
""")

md("""
    # ⚖️ 9. Experiment 2: Fixed Chain vs Dynamic Orchestrator

    The central question: does the dynamic agent beat the best fixed chain, and where?
""")
code("""
    if have("results/exp2_vs_static.csv"):
        e2 = pd.read_csv(ROOT / "results" / "exp2_vs_static.csv")
        display(e2[e2.metric == "EM"].round(3))
    if have("results/exp2_per_category.csv"):
        display(pd.read_csv(ROOT / "results" / "exp2_per_category.csv", index_col=0).round(3))
""")
md("""
    **How much room is there?** If a router always picked the best of the fixed chains *per question* (an **oracle**), EM would rise only from 0.766 to 0.798.
    That is the ceiling for any routing between these chains (`scripts/08_stats.py`). Per-category differences rely on very few questions (20 in the smallest categories) and should be read as trends.
""")
code("""
    if have("results/stats_baselines.json"):
        st = json.load(open(ROOT / "results" / "stats_baselines.json"))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in st["oracle"].items() if k != "over"})
""")

# --------------------------------------------------------------------------------------------- 10 traces
md("""
    # 🔬 10. Experiment 3: What Does the Agent Actually Do?

    ### 🔥 Tool choice by question category
    Each cell is the average number of calls per question.
""")
code("""
    for name in ("agent_full", "agent_full_guard", "agent_full_granite_guardlocal"):    # MedGemma plain, MedGemma + guardrail, Granite + local-index guardrail
        print(name)
        if have(f"results/fig_tool_heatmap_{name}.png"):
            display(Image(str(ROOT / "results" / f"fig_tool_heatmap_{name}.png"), width=620))
""")
md("""
    ### 🧭 Does a longer chain of thoughts help? (reasoning steps vs nDCG@10)
""")
code("""
    for name in ("agent_full", "agent_full_guard", "agent_full_granite_guardlocal"):
        if have(f"results/fig_steps_vs_ndcg_{name}.png"):
            print(name); display(Image(str(ROOT / "results" / f"fig_steps_vs_ndcg_{name}.png"), width=520))
""")
md("""
    The three agents shown below are the MedGemma scheduler without the guardrail, with the guardrail, and the Granite scheduler with the local-index guardrail. The guardrail removes most *retrieval misses*
    (19.2 % → 6.6 % of questions for MedGemma) but not *looping* (about 81 % either way), which is why looping alone does not explain the loss.

    ### ⚠️ Failure modes
    *looping* (repeats the same action), *format* (cannot be parsed), *stopped early* (answers without a tool),
    *bad arguments* (unknown tool, tool error, empty PubMed query), *retrieval miss* (the gold paper never appears), *reading error* (it appears but the answer is wrong).
""")
code("""
    for name in ("agent_full", "agent_full_guard", "agent_full_granite_guardlocal"):
        if have(f"results/failure_rates_{name}.csv"):
            print(name); display(pd.read_csv(ROOT / "results" / f"failure_rates_{name}.csv", index_col=0).round(3).tail(1))   # the "all" row; per-category rows are in the csv
""")
md("""
    ### 🕵️ Why are more tools worse (without the guardrail)? A look at the tool sequences

    We group the questions by the **sequence of tools** the agent used.
""")
code("""
    SHORT = {"retrieve_from_index": "RETR", "query_reformulator": "REFORM", "fetch_pubmed_live": "PUBMED",
             "generate_reasoning_query": "REASON", "query_openfda": "FDA"}

    def sequence_table(name, top=8):
        rows = load_traces(name)
        if not rows: return None
        df = pd.DataFrame([{"sequence": " > ".join(SHORT[s["tool"]] for s in r["trace"] if s.get("executed")) or "(none)",
                            "EM": r["EM"], "gold_seen": r["gold_seen"]} for r in rows])
        g = df.groupby("sequence").agg(n=("EM", "size"), EM=("EM", "mean"), gold_seen=("gold_seen", "mean")).sort_values("n", ascending=False)
        used = df.sequence.str.contains("RETR")
        print(f"{name}: retrieve_from_index called in {used.mean():.0%} of questions | EM {df[used].EM.mean():.3f} with vs {df[~used].EM.mean():.3f} without")
        return g.head(top).round(3)

    display(sequence_table("agent_full"))
    display(sequence_table("agent_full_guard"))
""")
md("""
    **The mechanism.** The reformulator and the reasoning tool return a *query*, not documents. A small model treats the call as "I searched" and stops,
    so the answer generator receives **no evidence** and falls back to Baseline-0 accuracy (EM ≈ 0.30). That explains almost exactly the gap between the full and the index-only agent.

    ### 🛡️ A guardrail that keeps the tools' definition intact
    If the agent stops with no document evidence, the system retrieves once (with the latest reformulated query, else the question) before answering.
    **Prediction written before running:** both schedulers reach about 0.73–0.75, i.e. they catch up with Vanilla RAG.
""")
code("""
    if have("results/exp_ladder.csv"):
        ladder = pd.read_csv(ROOT / "results" / "exp_ladder.csv")
        display(ladder[ladder.system.str.contains("Agent")][["system", "EM", "Contains", "tool_calls", "steps", "latency_s", "R@10"]].round(3))
""")
md("""
    ### 🧾 Side by side: fixed chain vs dynamic agent
    **Annotated pairs** (MedGemma agent + guardrail vs Vanilla RAG; full traces in `results/trace_examples_agent_full_guard.md`). Of the 500 questions the agent is better on 13 and worse on 24.

    | qid | gold | Vanilla RAG | Agent | What differed |
    |---|---|---|---|---|
    | 333 (better) | DRS1 | `DRS1 gene` (EM 0) | `DRS1` (EM 1) | Same paper, same facts. The agent is "better" only through answer wording (no extra word); it also repeated the same query twice and was stopped by the loop breaker. |
    | 272 (better) | Opsonophagocytic activity (OPA) assay | `OPA` (EM 0) | `OPA assay` (EM 1) | The agent called only `query_reformulator` (3 times, 2 repeats), so it had no documents; the **guardrail** ran `retrieve_from_index` with the rewritten query and the gold paper was found. Again an answer-form difference decides EM. |
    | 53 (worse) | covind19.org | `covind19.org` (EM 1) | `datahub.io` (EM 0) | After a first retrieval the agent also called `fetch_pubmed_live` and got generic COVID-vaccine papers; the answer generator then saw unrelated evidence beside the gold paper (gold seen = True) and answered wrongly: **evidence pollution** from an unnecessary tool. |
    | 78 (worse) | Cytosolic leaflet of the plasma membrane | same (EM 1) | `Plasma membrane` (EM 0) | The gold paper was retrieved in both systems, but the agent's answer came out less specific; the agent used its own query wording, which probably changed the retrieved set (not verified). |
    | 423 (worse) | Surgical decompression | same (EM 1) | `Anterior approach` (EM 0) | Reformulator loop, then guardrail retrieval with a query that is almost the question; a reading error on mostly equivalent evidence. |

    **Granite scheduler + local-index guardrail** (`results/trace_examples_agent_full_granite_guardlocal.md`): better on 16 questions, worse on 26.

    | qid | gold | Vanilla RAG | Agent | What differed |
    |---|---|---|---|---|
    | 333 (better) | DRS1 | `DRS1 gene` (EM 0) | `DRS1` (EM 1) | Granite follows the intended chain (reformulate → reasoning → retrieve → final answer in 4 steps). The gain is again answer wording. |
    | 137 (better) | H-2K(b)-tsA58 transgenic mouse (immortomouse) | `H-2K(b)-tsA58 transgenic mice` (EM 0) | `Immortomice` (EM 1) | Granite stopped after the two rewriting tools without retrieving; the guardrail retrieved once. EM counts the synonym; wording again. |
    | 189 (worse) | Angiotensin and acetylcholine | same (EM 1) | `P2X5 receptors` (EM 0) | The reformulated query ("CNS neurotransmitter regulation interaction substances pair") is more generic than the question; the gold paper was never retrieved. **Rewriting lost discriminative content.** |
    | 319 (worse) | Acinetobacter baumannii | same (EM 1) | `Pseudomonas aeruginosa` (EM 0) | Granite put its own wrong guess into the PubMed query ("Pseudomonas aeruginosa multidrug resistance …"), got confirming papers, and the gold paper was never retrieved: **self-confirming search**. |
    | 39 (worse) | Culture-specific intravenous antibiotics and surgical joint irrigation with graft retention | same (EM 1) | `Culture-specific intravenous (IV) antibiotics` (EM 0) | Gold paper retrieved; the answer is correct but incomplete (drops the surgical part), so exact match fails. |

    **Take-aways.** (i) In both "agent better" examples shown, the difference is answer wording, not better orchestration (we did not audit all 13). (ii) Looping on an identical call is the norm for MedGemma (81% of questions) but does not by itself change the answer. (iii) The clearest *real* harms are an unnecessary extra tool adding irrelevant evidence (MedGemma, qid 53), a rewritten query that is more generic than the question (Granite, qid 189) and a scheduler that puts its own wrong guess into the search query (Granite, qid 319).
""")

# --------------------------------------------------------------------------------------------- 11-12
md("""
    # 🆚 11. Granite vs MedGemma as the Scheduler

    A tool-calling-oriented model (`ibm-granite/granite-4.0-micro`, 3B) schedules the tools; MedGemma still runs the LLM tools and writes the answers.
    Same prompt, same tools: only the scheduler changes. (**Caveat:** the MedGemma scheduler prompt was refined three times on smoke tests, the Granite one never.)
""")
code("""
    if have("results/exp_ladder.csv"):
        ladder = pd.read_csv(ROOT / "results" / "exp_ladder.csv")
        display(ladder[ladder.system.str.contains("Granite|Agent: full|index only")][["system", "EM", "tool_calls", "steps", "latency_s", "R@1", "R@10"]].round(3))
    if have("results/fig_tool_heatmap_agent_full_granite.png"):
        display(Image(str(ROOT / "results" / "fig_tool_heatmap_agent_full_granite.png"), width=620))
""")
md("""
    **Reading the table.**
    - Out of the box Granite is far worse than MedGemma (EM 0.550 vs 0.684, paired -0.134, significant): it uses all tools (2.06 per question) but treats the two query-rewriting tools as if they returned evidence, so 133 questions never reach the index.
    - With the guardrail (retrieve once if there is no document evidence) Granite reaches 0.700, still below MedGemma + guardrail (0.742): in 65 questions (13%) it relied on PubMed papers alone (EM 0.431, gold paper seen in about 10%), and the guardrail does not fire when evidence exists.
    - A *post-hoc* variant (designed after this diagnosis) also retrieves when the local index was never queried: Granite reaches **0.744**, inside the range predicted before the run (0.735-0.745), +0.044 over the original guardrail (CI [0.022, 0.068]) and equal to MedGemma + guardrail (0.742). MedGemma never used PubMed without the index (0 questions), so the variant is only relevant for Granite.
    - So most of the Granite deficit was the tool-contract failure, not a lack of model ability; the remaining prompt-fit question is listed under limitations.
""")
md("""
    # 📝 12. An Extra Look: Paraphrased Questions

    About half of the questions are paraphrased. Are they harder, and is the difficulty in *finding* the paper or in *reading* it?
""")
code("""
    if have("results/paraphrase_split.csv"):
        display(pd.read_csv(ROOT / "results" / "paraphrase_split.csv").round(3))
""")
md("""
    **Reading the table.** Paraphrased questions are harder for Vanilla RAG (about −10 EM points, significant), yet Recall@1 is the same for both groups:
    the semantic retriever is not fooled by the changed wording; the loss comes from reading and matching the answer.
""")
md("""
    ## 12b. How much does the answer-prompt wording matter?

    We never searched for an optimal prompt, so we check sensitivity: Vanilla RAG and Baseline 0 are re-run with three reworded answer prompts
    (v4 paraphrased instruction, v5 no format examples, v6 one-line instruction). Same model, same 500 questions, same retrieval; only the text changes.
""")
code("""
    if have("results/prompt_sensitivity.csv"):
        display(pd.read_csv(ROOT / "results" / "prompt_sensitivity.csv").round(3))
    else:
        print("run scripts/14_prompt_sensitivity.py first")
""")
md("""
    **Reading the table.** Wording matters, more than we hoped. Vanilla RAG moves between 0.704 and 0.776 EM (spread 0.072), the same size as the gaps between agent and fixed chains.
    - A paraphrased instruction with the same examples (v4) is statistically the same as v3 (+0.012, CI [-0.006, 0.032]).
    - Removing the format examples (v5, v6) costs 4.6-6.0 EM points (significant), but *Contains* barely moves (0.820 -> 0.798 / 0.818): the loss is answer *form* (length, modifiers), not knowledge.
    - So v3 is on the plateau of "clear instruction + format examples", but we cannot claim it is optimal, and absolute EM should be read with about ±5 points of prompt uncertainty.
    - All systems share the same prompt, so rankings are comparable; section 12c re-checks the fixed-chain ranking under v4.
""")
md("""
    ## 12c. Does the ranking of the fixed chains survive a different prompt?

    The four fixed chains and Baseline 0 are re-run with the paraphrased answer prompt v4 (same model, same 500 questions).
""")
code("""
    import json as _json
    if have("results/stats_baselines_v4.json"):
        _s = _json.load(open(ROOT / "results" / "stats_baselines_v4.json"))
        _t = pd.DataFrame({k: {"EM": v["EM"]["mean"], "diff_vs_vanilla": _s["vs_ref"].get(k, {}).get("diff", 0.0),
                               "p_mcnemar": _s["vs_ref"].get(k, {}).get("p_mcnemar", float("nan"))} for k, v in _s["systems"].items()}).T
        display(_t.round(3)); print("oracle EM", round(_s["oracle"]["oracle_EM"], 3), "headroom", round(_s["oracle"]["headroom"], 3))
""")
md("""
    **Reading the table.** Under v4 Vanilla RAG, reformulate→retrieve and FDA-augmented RAG are indistinguishable (0.776, 0.778, 0.776) and no chain beats Vanilla RAG, as under v3; the oracle headroom is again about +3 points (0.030 vs 0.032).
    One detail does change: the reasoning→retrieve penalty (v3: -0.028, p = 0.02) shrinks to -0.012 and is not significant (p = 0.33), so we only claim that the reasoning chain is *not better* than Vanilla RAG, not that it is worse.
    The conclusions "retrieval is what matters" and "no fixed chain clearly beats plain retrieval" do not depend on the prompt wording.
""")
md("""
    ## 12d. Held-out validation: 500 new questions

    Prompts, the agent loop and the guardrail were developed while looking at the first 500 questions. To check that the conclusions are not an artefact of that, every system was **frozen** and run on
    **500 new questions** (`data/holdout.jsonl`): questions whose gold abstracts were already in the 5,000-paper index as distractors, same category counts as the development sample except
    category 7 (only 10 available; 10 extra questions in category 4). The predictions below were written *before* the runs.

    | | Prediction (written before the run) |
    |---|---|
    | P1 | Baseline 0 EM in [0.22, 0.36]; Vanilla RAG EM in [0.70, 0.82] |
    | P2 | Vanilla, reformulate and FDA chains within ±0.03 of each other; no fixed chain significantly beats Vanilla |
    | P3 | The unmodified MedGemma full agent is at least 0.04 below Vanilla (significant) |
    | P4 | The guardrail lifts the MedGemma full agent by at least +0.03 (significant); MedGemma + guardrail is not significantly above Vanilla |
    | P5 | Granite full < MedGemma full; Granite + local-index guardrail within 0.04 of MedGemma + guardrail; no Granite variant significantly above Vanilla |
    | P6 | The index-only agent is not significantly different from Vanilla |
""")
code("""
    if have("results/holdout/exp_ladder.csv"):
        display(pd.read_csv(ROOT / "results" / "holdout" / "exp_ladder.csv")[["system", "EM", "Contains", "tool_calls", "steps", "latency_s", "R@1", "R@10"]].round(3))
    else:
        print("run the held-out experiments first (README section 5)")
""")
md("""
    **Result: all six predictions held.**

    | | Prediction | Held-out result | |
    |---|---|---|---|
    | P1 | Baseline 0 in [0.22, 0.36]; Vanilla in [0.70, 0.82] | 0.248; 0.746 | yes |
    | P2 | fixed chains within ±0.03 of Vanilla, none significantly better | 0.746 / 0.744 / 0.744; all paired differences ≤ 0.002, p = 1.0 | yes |
    | P3 | MedGemma full ≥ 0.04 below Vanilla | 0.644, −0.102 (CI [−0.138, −0.068]) | yes |
    | P4 | guardrail ≥ +0.03; guarded agent not above Vanilla | +0.078 (CI [+0.050, +0.108]); 0.722 vs 0.746, −0.024 (CI [−0.048, 0.000], p = 0.073) | yes |
    | P5 | Granite full < MedGemma full; local-index Granite within 0.04 of guarded MedGemma; none above Vanilla | 0.518 < 0.644; 0.708 vs 0.722 (−0.014); best Granite variant 0.038 *below* Vanilla (significant) | yes |
    | P6 | index-only not different from Vanilla | 0.726 vs 0.746, −0.020 (CI [−0.046, +0.006], p = 0.164) | yes |

    What the held-out run adds:
    - The development conclusions replicate on new questions: the unmodified agent loses about 10 EM points to every fixed chain, in 10 of 11 categories, and the loss comes with Recall@10 falling from 0.956 to 0.800.
    - The guardrail repairs most of the loss on both samples (+0.058 on development, +0.078 held-out), and the guarded MedGemma agent stays close to, but below, Vanilla RAG on both: -0.022 (p = 0.099) on development and -0.024 (p = 0.073) held-out. This is a replication, not a shrinking effect.
    - The index-only agent (+0.082 over the full agent, as +0.068 on development) is again statistically indistinguishable from Vanilla. No configuration of the dynamic agent beats the fixed chains, on either sample.
    - Granite is the weaker scheduler (0.518 vs 0.644 unguarded). The original guardrail leaves it 4.0 points behind guarded MedGemma (significant). The local-index variant was designed on the development sample, where it was 2.0 points below Vanilla RAG (not significant); on held-out data it is 3.8 points below (significant). This is the one result that moved the way tuning on the development sample would predict.
    - Not repeated on held-out data: the per-tool ablations of section 8. Those are therefore development-sample results only.
    - Caveat: the held-out set has 10 category-7 questions (not 20) and 10 extra in category 4; category-level rows are noise at these sizes.
""")

md("""
    ## 12e. Coverage experiment: what if the local index does not contain the answer?

    On our sample the local index almost always contains the gold paper (Recall@10 0.97), which leaves a dynamic agent little to decide. Here we remove that comfort.
    On the 500 held-out questions the gold paper of a fixed random **0 %, 50 % or 100 %** of the questions is withheld from the index (the sets are nested; implemented as a filter at search time).
    For a withheld question only live PubMed (`fetch_pubmed_live`) can still find the paper. To keep the comparison fair, the fixed side gets the same tools:
    *reformulate → PubMed* and *retrieve + PubMed* (both always call PubMed). The agent is compared with itself **without** PubMed. All systems are frozen; the predictions were written before the runs.

    | | Prediction (written before the run) |
    |---|---|
    | C1 | Vanilla RAG EM falls to 0.44-0.58 with 50 % withheld and to 0.20-0.34 with 100 % withheld (Baseline 0 is 0.248) |
    | C2 | Live PubMed recovers the gold paper only for a minority (10-40 % of withheld questions, retrieve + PubMed chain); that chain beats Vanilla RAG by 0.04-0.20 at 100 % |
    | C3 | The agent does **not** notice that the index is incomplete: its share of questions with a PubMed call differs by less than 0.10 between withheld and retained questions (50 % level) and stays below 0.30 at 100 % |
    | C4 | Hence PubMed adds little inside the agent: full + guardrail minus no-PubMed + guardrail is not significant at 50 % and at most +0.05 at 100 % |
    | C5 | The always-PubMed hybrid chain is at least as good as the agent at 100 % withheld; at 0 % it is within -0.05..0 of Vanilla RAG (extra documents distract) |
""")
code("""
    if have("results/coverage/coverage_em.csv"):
        display(pd.read_csv(ROOT / "results" / "coverage" / "coverage_em.csv"))
        display(pd.read_csv(ROOT / "results" / "coverage" / "coverage_switching.csv"))
        display(pd.read_csv(ROOT / "results" / "coverage" / "coverage_pairs.csv"))
    if have("results/coverage/fig_coverage.png"):
        display(Image(str(ROOT / "results" / "coverage" / "fig_coverage.png"), width=620))
""")
md("""
    **Result: four of the five predictions held; C2 held only in part.**

    | | Prediction | Result | |
    |---|---|---|---|
    | C1 | Vanilla 0.44-0.58 (50 %) and 0.20-0.34 (100 %) | 0.488 and 0.220 | yes |
    | C2 | PubMed finds the gold paper for 10-40 % of withheld questions; retrieve + PubMed beats Vanilla by 0.04-0.20 at 100 % | gold paper found for **8.4-9.0 %** (just below the range); +0.118 | **first half no**, second half yes |
    | C3 | the agent does not notice missing coverage (PubMed-call share differs by < 0.10; < 0.30 at 100 %) | 10.8 % of withheld vs 3.6 % of retained questions (difference 0.072); 7.0 % at 100 % | yes |
    | C4 | PubMed adds little inside the agent | full + guardrail minus no-PubMed: -0.004 (50 %), 0.000 (100 %), both not significant | yes |
    | C5 | the always-PubMed chain is at least as good as the agent at 100 %; within -0.05..0 of Vanilla at 0 % | +0.098 over the agent (p < 0.001); -0.016 vs Vanilla | yes |

    What the experiment shows:
    - **Coverage is what limits accuracy.** Vanilla RAG falls from 0.746 to 0.488 to 0.220 as 0 / 50 / 100 % of the gold papers are withheld. At 100 % it is even below Baseline 0 (0.248): unrelated abstracts do not help (we did not test this small difference).
    - **Live PubMed is a weak but real substitute.** Keyword search finds the exact source paper for only about 9 % of the questions, yet the PubMed-only chain is 0.088 above no tools (CI [0.052, 0.124]) and retrieve + PubMed is 0.118 above Vanilla RAG at 100 %: related abstracts often carry the needed fact even when they are not the source paper.
    - **The agent does not use it.** It calls PubMed on 10.8 % of the withheld questions (3.6 % of the retained ones, 7.0 % overall at 100 %), roughly three times more often when the index cannot answer, but from a very low base: about nine withheld questions in ten get no PubMed call. As a result the agent with PubMed is not better than the same agent without it.
    - **A fixed chain that always calls both tools beats the agent exactly where the tool matters:** -0.038 at 50 % (p = 0.025) and -0.098 at 100 % (p < 0.001); at 0 % there is no difference (-0.008). So the extra tool has value, and the failure is the *decision* to use it, not a lack of value.
    - Descriptive split at 50 % (250 questions each; not tested): the agent against Vanilla is +0.044 on withheld and -0.040 on retained questions (net +0.002); PubMed inside the agent is +0.028 on withheld and -0.036 on retained questions, which fits "helps when needed, adds noise when not".
    - Caveats: one random withheld subset per level (seed 0); MedGemma scheduler only (no Granite run); PubMed queries are written by the LLM and PubMed ANDs all terms, so a better query strategy might find the gold paper more often; live PubMed results are stored in `cache/` for reproducibility.
""")


# --------------------------------------------------------------------------------------------- 13-14
md("""
    # 🧠 13. Discussion, Limitations and Related Work

    ## Conclusion

    *Does dynamic orchestration beat a fixed retrieval pipeline?* **Not in our setting, on either sample.** With a small zero-shot scheduler, no agent configuration is better than plain Vanilla RAG
    (EM 0.764 on the 500 development questions, 0.746 on the 500 frozen held-out questions). The best agents come close but stay below it:

    | | development | held-out | vs Vanilla RAG (dev / held-out) |
    |---|---|---|---|
    | MedGemma agent, all tools, no guardrail | 0.684 | 0.644 | -0.080 / -0.102 (both significant) |
    | MedGemma agent + guardrail | 0.742 | 0.722 | -0.022 (p = 0.099) / -0.024 (p = 0.073) |
    | MedGemma agent, index only | 0.752 | 0.726 | -0.012 / -0.020 (not significant) |
    | Granite agent + local-index guardrail | 0.744 | 0.708 | -0.020 (n.s.) / -0.038 (significant) |

    and they cost roughly 15-20 times the latency (3.0-3.4 s per question against 0.18-0.19 s). The held-out run is the strongest piece of evidence: the six predictions written before it all held (section 12d). The coverage experiment (section 12e) removes the most favourable excuse for this result, a retriever that already finds everything: when the local index is incomplete a fixed chain that always calls PubMed still beats the agent (+0.038 at 50 % and +0.098 at 100 % withheld).

    ## What we found

    - **Retrieval is what matters:** EM 0.30 → 0.76. On top of it, no tool or orchestrator improves reliably (rewriting, reasoning queries and OpenFDA change EM by at most a few points).
    - **Without the guardrail the extra tools hurt** (-8 EM points on development, -10 held-out), because of the "query without evidence" failure: the rewriting tools return a query, the small scheduler stops, and the answer generator gets no documents.
    - **With the guardrail no single tool matters** (development sample only; not repeated held-out): removing OpenFDA, PubMed, the reasoning tool or the reformulator changes EM by -0.006 to +0.014 for MedGemma and 0.000 to +0.008 for Granite, with every 95 % CI including 0 (about ±0.025). OpenFDA is never called. The tools only add time: Granite without the reasoning tool or the reformulator is as accurate at about 1.7 s instead of 3.0 s. An ablation can only say "no detectable effect", not "zero effect".
    - A tool-calling model (Granite) really uses all tools and stops by itself, but misunderstands what the rewriting tools return; it is also the weaker scheduler (-4 points with the original guardrail, significant on both samples).
    - **When the index is incomplete the extra tool becomes useful, but the agent does not exploit it** (section 12e): a fixed chain that always adds live PubMed is +0.118 over Vanilla RAG at 100 % withheld, while the agent calls PubMed on only 7-11 % of the questions and ends up 0.098 behind that chain.
    - Our retrieval baseline is strong (Recall@10 0.97) and the oracle over the four fixed chains is only about +3 to +5 points: little room for any orchestrator.

    **Limitations.** 500 questions per sample (a paired difference needs about ±2 points to be detectable); a closed 5,000-paper corpus (the dataset paper searches 16M abstracts, so absolute numbers are not comparable);
    the answer prompt, the agent prompt and the guardrail were developed by reading failures on the development sample. We then froze every system and re-ran it on 500 new questions: the conclusions hold, the guarded MedGemma gap to Vanilla RAG is unchanged (-0.022 vs -0.024), and only the Granite local-index guardrail variant, which was designed on the development sample, looks worse on new data (-0.020 → -0.038). The held-out set has 10 questions in category 7 (not 20) and 10 extra in category 4, so per-category numbers are noise, and the per-tool ablation was not repeated on it. There is no systematic prompt search (prompt wording moves EM by about ±5 points, section 12b), so the prompts are shared and fixed, not optimal;
    the scheduler prompt was developed with MedGemma and reused unchanged for Granite, so part of the Granite gap may be prompt fit; only zero-shot small schedulers were tested (no few-shot tool demos, no fine-tuning, no larger LLM);
    the coverage experiment uses one random withheld subset per level and only the MedGemma scheduler; the guardrail is a post-hoc fix (its effect was predicted before the run); EM underestimates correct answers with different wording, and we did not audit the errors one by one.

    **Related work.**
    - *Benchmark.* PaperSearchQA (Burgess et al., 2026) introduces the questions; it searches 16M abstracts and trains a search agent with RL (Search-R1, Jin et al., 2025). We do not train anything and search 5,000 abstracts, so absolute numbers are not comparable.
    - *Agent loop.* We use the ReAct pattern (Yao et al., 2022). Splitting a *planner* from an answer writer follows the multi-LLM idea of α-UMi (Shen et al., 2024), whose authors report that small models are weak tool users; FireAct (Chen et al., 2023) fine-tunes agents for this reason.
    - *When to retrieve.* Self-RAG (Asai et al., 2024) and Adaptive-RAG (Jeong et al., 2024) learn to decide whether and how much to retrieve. Our zero-shot schedulers have no such training, which is consistent with the failures we observe (stopping without evidence, looping).
    - *Query rewriting.* Rewriting helps weak retrievers, but Weller et al. (2024) show that expansions can hurt strong ones; with Recall@10 of 0.97 our reformulate and reasoning chains are not better than plain retrieval.
    - *Evaluation.* Exact match underestimates correct answers (Kamalloo et al., 2023), hence the Contains and ROUGE columns; we use paired bootstrap and McNemar tests rather than comparing point estimates.
""")
md("""
    # ♻️ 14. Reproducibility

    Every number in this notebook comes from `results/` and `traces/`, produced by the numbered scripts in `scripts/` (see `README.md` for the exact commands and SLURM jobs).
    Seed 42, greedy decoding, bf16, one question per PMID. All runs are resumable.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent.parent / "notebook" / "PaperSearchQA_ReAct_Agent.ipynb"
out.parent.mkdir(exist_ok=True)
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print(f"wrote {out}  ({sum(c['cell_type'] == 'markdown' for c in cells)} markdown + {sum(c['cell_type'] == 'code' for c in cells)} code cells)")
