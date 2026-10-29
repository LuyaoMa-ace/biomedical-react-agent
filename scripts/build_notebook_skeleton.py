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
    📝 Luyao Ma  ·  project proposal by Lorenzo Molfetta  ·  *Big Data Analytics and Text Mining*, University of Bologna
""")
md("""
    ## 📜 Outline
    0. Goals
    1. Set up: packages, secrets, GPU
    2. Dataset: PaperSearchQA
    3. Local retrieval and the embedding models
    4. The LLM, the answer prompt and the metrics
    5. Baselines: no tools and fixed chains
    6. Tool definition: five tools
    7. The ReAct agent in LangGraph
    8. Experiment 1: which tool contributes?
    9. Experiment 2: fixed chain vs dynamic orchestrator
    10. Experiment 3: what does the agent actually do? (trace analysis)
    11. Granite vs MedGemma as the scheduler
    12. An extra look: paraphrased questions
    13. Discussion, limitations and related work
    14. Reproducibility
""")
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

# --------------------------------------------------------------------------------------------- 1 setup
md("""
    ## ⚙️ 1. Set Up: Packages, Secrets, GPU

    > **Where is the data?** This notebook expects the repository layout (`scripts/`, `data/`, `results/`, `traces/`).
    > On Colab, upload `bigdata_project.zip` and run the unzip cell below. All **tables and figures are read from `results/`**,
    > so only the small *live demos* need a GPU (switch `LIVE`).
""")
code("""
    # Unzip the submission archive if we are on Colab (skip locally)
    import os, zipfile, pathlib
    if pathlib.Path("bigdata_project.zip").exists() and not pathlib.Path("scripts").exists():
        zipfile.ZipFile("bigdata_project.zip").extractall(".")
        os.chdir("bigdata_project")
    print("working directory:", os.getcwd(), "|", sorted(os.listdir("."))[:8])
""")
code("""
    # 📦 Packages.   TODO: pin the exact versions listed in env/requirements.txt before submission
    %pip install -q "transformers>=5" sentence-transformers faiss-cpu langgraph langchain-core metapub datasets rouge-score pandas scipy matplotlib
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
    (three questions per abstract, ten expert-defined categories, 50 % of the questions paraphrased to reduce keyword overlap).
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
    `jina` is the best (Recall@1 0.83 vs 0.76 for `bge-m3` and 0.69 for `pubmedbert`), so a larger general model beats the smaller biomedical one here.
    All following experiments use `jina` and the 5,000-document index.

    <!-- TODO: optional live demo: encode 200 abstracts with jina, build a FAISS IndexFlatIP and search 5 questions (guarded by LIVE) -->
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
    EM is strict: about 21 % of our "errors" are right answers with different wording, so we also report Contains and ROUGE.
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
    changes the result by at most a few points, and reasoning queries are significantly *worse* (strong retriever + factoid questions: LLM query expansion adds noise).

    <!-- TODO: optional live demo: run Baseline 0 and Vanilla RAG on 10 questions with scripts/06_baselines.py logic (guarded by LIVE) -->
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
        import tempfile
        from tools import OpenFDA, PubMedLive
        fda = OpenFDA(cache_path=tempfile.mkdtemp() + "/fda.jsonl")
        label = fda("alteplase")
        print("OpenFDA fields for alteplase:", list(label))
        pm = PubMedLive(cache_path=tempfile.mkdtemp() + "/pm.jsonl", retmax=2)
        q_full = "Which imaging technique employed during pregnancy can identify unilateral hydrocephalus in the fetus"
        q_kw = "sonography prenatal diagnosis unilateral hydrocephalus"
        print("full question ->", [d["pmid"] for d in pm(q_full)], "| keywords ->", [d["pmid"] for d in pm(q_kw)])
    except Exception as e:
        print("(tool demo skipped:", repr(e)[:100], ")")
""")
md("""
    ### 🧩 The two query-rewriting tools

    They only **transform the query**; the agent still has to call `retrieve_from_index` with the result. This is the specification we follow.
    We will see in section 10 that small models often forget this second step.

    <!-- TODO: live demo (LIVE): show query_reformulator and generate_reasoning_query outputs for 3 questions -->
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

    agent_traces = load_traces("agent_full")
    for r in agent_traces[:2]:
        show_trace(r); print()
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
    **Reading the table.** Without a guardrail, removing tools *helps*: the index-only agent is 6.8 EM points better than the full agent (CI [+3.8, +9.8]), because the rewriting tools lead to answers without evidence (see section 8b).
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
    for name in ("agent_full",):
        if have(f"results/fig_tool_heatmap_{name}.png"):
            display(Image(str(ROOT / "results" / f"fig_tool_heatmap_{name}.png"), width=620))
""")
md("""
    ### 🧭 Does a longer chain of thoughts help? (reasoning steps vs nDCG@10)
""")
code("""
    if have("results/fig_steps_vs_ndcg_agent_full.png"):
        display(Image(str(ROOT / "results" / "fig_steps_vs_ndcg_agent_full.png"), width=520))
""")
md("""
    ### ⚠️ Failure modes
    *looping* (repeats the same action), *format* (cannot be parsed), *stopped early* (answers without a tool),
    *bad arguments* (unknown tool, tool error, empty PubMed query), *retrieval miss* (the gold paper never appears), *reading error* (it appears but the answer is wrong).
""")
code("""
    if have("results/failure_rates_agent_full.csv"):
        display(pd.read_csv(ROOT / "results" / "failure_rates_agent_full.csv", index_col=0).round(3))
""")
md("""
    ### 🕵️ Why are more tools worse? A look at the tool sequences

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

# --------------------------------------------------------------------------------------------- 13-14
md("""
    # 🧠 13. Discussion, Limitations and Related Work

    **What we found.**
    - Retrieval is what matters: EM 0.30 → 0.76. On top of it, no tool or orchestrator improves reliably.
    - With a *small, zero-shot* scheduler the dynamic agent is not better than the fixed chain, and extra tools hurt because of the "query without evidence" failure.
    - A tool-calling model (Granite) really uses all tools and stops by itself, but misunderstands what the rewriting tools return.
    - Our retrieval baseline is strong (Recall@10 0.97) and the oracle ceiling is only +3 points: little room for any orchestrator.

    **Limitations.** 500 questions (a paired difference needs about ±2 points to be detectable); a closed 5,000-paper corpus (the dataset paper searches 16M abstracts, so absolute numbers are not comparable);
    the answer prompt and the agent prompt were adjusted by hand after reading failures on the same questions (no separate dev set, no systematic prompt search, so they are shared and fixed, not optimal; see the sensitivity check in section 12b);
    the scheduler prompt was developed with MedGemma and reused unchanged for Granite, so part of the Granite gap may be prompt fit; only zero-shot small schedulers were tested (no few-shot tool demos, no fine-tuning, no larger LLM);
    the guardrail is a post-hoc fix (its effect was predicted before the run); EM underestimates correct answers with different wording.

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
