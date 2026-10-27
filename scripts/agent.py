"""Dynamic ReAct agent built with LangGraph.

Graph (one question):

    [agent] --stop--> [answer] --> END
       |  ^
       v  |
    [tools]

* `agent` decides WHICH tool to call and WHEN to stop (it only schedules).
* `answer` writes the final answer with the SAME answer prompt as Baseline 0 / the static chains,
  from all evidence the agent collected. So every system shares one answer generator and the
  comparison isolates the retrieval policy (fixed vs dynamic).

* `agent` node: ONE single-turn prompt = instructions + tool list + question + scratchpad
  (MedGemma is not evaluated for multi-turn chat). The model turn is pre-filled with
  "Thought:" and generation stops at "\\nObservation:" (the system writes observations).
* `tools` node: runs the chosen tool and appends "Observation: ...".
* Plain-text ReAct format, parsed with regexes (no native function calling in a 4B model).

Everything that can go wrong is logged per question (parse errors, unknown tool, repeated
call, forced final answer, answering without any tool) for the failure analysis.
"""
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from prompts import answer_prompt, PREFILL as ANSWER_PREFILL

TOOL_DOCS = {
    "retrieve_from_index": "search the LOCAL corpus of PubMed abstracts (fast). Input: a search query. "
                           "Returns the top abstracts.",
    "query_reformulator": "rewrite a question into a keyword-style search query (does not search). "
                          "Input: the question. Returns a query for retrieve_from_index.",
    "generate_reasoning_query": "write a short reasoning about what the answer abstract would contain "
                                "(does not search). Input: the question. Returns text to use as the "
                                "query of retrieve_from_index.",
    "fetch_pubmed_live": "search ALL of PubMed online (slower). Use it when the local results do not "
                         "contain the answer. Input: 3-6 specific KEYWORDS, not a full question. "
                         "Returns up to 3 abstracts.",
    "query_openfda": "look up an FDA drug label (indications, mechanism, adverse reactions). "
                     "Input: ONE drug name. Only useful for questions about a specific drug.",
}

EXAMPLE = """Example (format only):
Question: Which planet is known as the Red Planet?
Thought: I should look this up in the local corpus.
Action: retrieve_from_index
Action Input: Red Planet nickname
Observation: [1] Mars. Mars appears red because of iron oxide and is called the Red Planet.
Thought: The abstract says Mars is called the Red Planet.
Final Answer: Mars
"""


def build_header(tools: List[str]) -> str:
    lines = "\n".join(f"- {t}: {TOOL_DOCS[t]}" for t in tools)
    return (
        "You are a biomedical research assistant. Answer the question using the tools, then give a "
        "short final answer.\n\nTools:\n" + lines + "\n\n"
        "Use exactly this format:\n"
        "Thought: one or two sentences about what to do next\n"
        f"Action: one of [{', '.join(tools)}]\n"
        "Action Input: the input for the tool\n"
        "Observation: the result (written by the system, never by you)\n"
        "... (repeat Thought/Action/Action Input/Observation as needed)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: a short noun phrase (at most 6 words); copy the exact term used in the abstracts\n\n"
        "Rules: use only information from observations; never write an Observation yourself; "
        "do not repeat an identical action; finish as soon as you can answer.\n\n" + EXAMPLE)


_FINAL = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.S | re.I)
_ACTION = re.compile(r"Action\s*:\s*([A-Za-z_]+)")
_INPUT = re.compile(r"Action\s*Input\s*:\s*(.*?)(?:\n|$)")


def parse_step(text: str) -> Dict[str, Any]:
    """Parse one model turn (text excludes the 'Thought:' prefill)."""
    t = text.split("\nObservation:")[0].split("Observation:")[0]   # drop hallucinated observations
    a, ai, fa = _ACTION.search(t), _INPUT.search(t), _FINAL.search(t)
    if a and ai and (not fa or a.start() < fa.start()):
        return {"kind": "action", "tool": a.group(1).strip(),
                "input": ai.group(1).strip().strip("\"'`"), "thought": t.split("\nAction")[0].strip()}
    if fa and fa.group(1).strip():
        return {"kind": "final", "answer": fa.group(1).strip().split("\n")[0].strip(),
                "thought": t.split("Final")[0].strip()}
    return {"kind": "error", "text": t.strip()}


class State(TypedDict, total=False):
    question: str
    scratch: str        # everything the model has seen so far (incl. corrective messages)
    step: int
    bad: int            # consecutive unproductive steps (format error / repeat / unknown tool)
    pending: Optional[Dict[str, Any]]
    stop: Optional[str]          # why the agent stopped: final | loop | max_steps
    agent_answer: Optional[str]  # the ReAct-style "Final Answer" the scheduler itself wrote (analysis only)
    evidence: List[str]          # observations that carry documents / drug labels
    final: Optional[str]
    steps: List[Dict[str, Any]]
    flags: Dict[str, int]
    seen: List[str]


NEXT_FIRST = "Now write the first step: choose one action.\n"
NEXT_MORE = ("Now write the next step. If any abstract above mentions a plausible answer to the question, "
             "write 'Thought: I now know the final answer' and then 'Final Answer: ...'. Search again ONLY "
             "if the abstracts are about a different topic than the question, and then use a different "
             "query or tool. Keep the Thought to one short sentence.\n")
MAX_THOUGHT_CHARS = 300
EVIDENCE_TOOLS = ("retrieve_from_index", "fetch_pubmed_live", "query_openfda")
MAX_EVIDENCE_CHARS = 9000


def initial_state(question: str) -> State:
    return {"question": question, "scratch": "", "step": 0, "bad": 0, "pending": None, "stop": None,
            "agent_answer": None, "evidence": [], "final": None, "steps": [], "flags": {}, "seen": []}


class ReActAgent:
    def __init__(self, llm, tools: Dict[str, Callable[[str], Any]], max_steps: int = 6,
                 max_new_tokens: int = 160, loop_break: int = 2, answer_llm=None, guardrail: bool = False,
                 guardrail_local: bool = False):
        """llm.generate(prompt, max_new_tokens=, prefill=, stop=) -> str   (one prompt)
        tools[name](input) -> (observation_text, meta_dict)
        loop_break: after this many consecutive unproductive steps, stop and answer.
        llm = the scheduler (decides tools / when to stop); answer_llm = the answer generator
        (defaults to llm; set it to MedGemma when the scheduler is another model).
        guardrail: if the agent stops with NO document evidence, run one retrieve_from_index call
        (latest reformulated query, else the question) before answering. Tool semantics are unchanged.
        guardrail_local (variant, implies guardrail): also fire when evidence exists but none of it came from the
        local index (e.g. only PubMed papers), i.e. whenever retrieve_from_index was never executed."""
        self.llm, self.tools, self.max_steps, self.max_new = llm, tools, max_steps, max_new_tokens
        self.loop_break = loop_break
        self.answer_llm = answer_llm or llm
        self.guardrail_local = guardrail_local
        self.guardrail = guardrail or guardrail_local
        self.header = build_header(list(tools))
        g = StateGraph(State)
        g.add_node("agent", self.agent_node)
        g.add_node("tools", self.tools_node)
        g.add_node("answer", self.answer_node)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", lambda s: "answer" if s.get("stop") else "tools",
                                {"answer": "answer", "tools": "tools"})
        g.add_edge("tools", "agent")
        g.add_edge("answer", END)
        self.graph = g.compile()

    def _prompt(self, s):
        nxt = NEXT_MORE if s["evidence"] or s["seen"] else NEXT_FIRST   # instruction closest to generation
        return f"{self.header}\nBegin!\nQuestion: {s['question']}\n{s['scratch']}\n{nxt}"

    def agent_node(self, s: State) -> State:
        flags, steps, step = dict(s["flags"]), list(s["steps"]), s["step"] + 1
        for reason, cond in (("loop", s["bad"] >= self.loop_break), ("max_steps", step > self.max_steps)):
            if cond:                                    # the agent did not stop by itself
                flags["forced_final"] = 1
                if reason == "loop":
                    flags["loop_break"] = 1
                steps.append({"step": step, "kind": "forced_stop", "reason": reason})
                return {"stop": reason, "flags": flags, "steps": steps, "step": step - 1}
        raw = self.llm.generate(self._prompt(s), max_new_tokens=self.max_new, prefill="Thought:",
                                stop=["\nObservation:"])
        p = parse_step(raw)
        rec = {"step": step, "raw": raw[:600], "kind": p["kind"]}
        if p["kind"] == "final":
            rec.update(thought=p["thought"][:300], answer=p["answer"])
            steps.append(rec)
            if not any(x.get("kind") == "action" for x in steps):
                flags["no_tool_answer"] = 1             # decided to answer without looking anything up
            return {"stop": "final", "agent_answer": p["answer"], "flags": flags, "steps": steps, "step": step}
        if p["kind"] == "action":
            rec.update(thought=p["thought"][:300], tool=p["tool"], input=p["input"][:300])
            steps.append(rec)
            p = {**p, "thought": p["thought"][:MAX_THOUGHT_CHARS]}
            return {"pending": p, "flags": flags, "steps": steps, "step": step,
                    "scratch": s["scratch"] + f"Thought: {p['thought']}\nAction: {p['tool']}\n"
                                              f"Action Input: {p['input']}\n"}
        flags["parse_error"] = flags.get("parse_error", 0) + 1
        steps.append(rec)
        return {"pending": {"kind": "error"}, "flags": flags, "steps": steps, "step": step,
                "scratch": s["scratch"] + f"Thought: {p['text'][:200]}\n"}

    def tools_node(self, s: State) -> State:
        p, flags, steps, seen = s["pending"], dict(s["flags"]), list(s["steps"]), list(s["seen"])
        evidence = list(s["evidence"])
        meta, ran = {}, False
        if p["kind"] == "error":
            obs = "Invalid format. Reply with 'Action:' and 'Action Input:', or with 'Final Answer:'."
        elif p["tool"] not in self.tools:
            flags["unknown_tool"] = flags.get("unknown_tool", 0) + 1
            obs = f"Unknown tool '{p['tool']}'. Available tools: {', '.join(self.tools)}."
        else:
            key = f"{p['tool']}|{p['input']}"
            if key in seen:
                flags["repeat_call"] = flags.get("repeat_call", 0) + 1
                obs = "You already ran this exact action. Use a different input or tool, or give the Final Answer."
            else:
                seen.append(key)
                try:
                    obs, meta = self.tools[p["tool"]](p["input"])
                    ran = True
                    if p["tool"] in EVIDENCE_TOOLS and not obs.startswith(("No ", "Tool error")):
                        evidence.append(obs)
                except Exception as e:                  # tool failures are observations, not crashes
                    flags["tool_error"] = flags.get("tool_error", 0) + 1
                    obs, meta = f"Tool error: {repr(e)[:120]}", {}
        if steps:
            steps[-1] = {**steps[-1], "observation": obs[:700], "meta": meta, "executed": ran}
        return {"scratch": s["scratch"] + f"Observation: {obs}\n", "pending": None, "steps": steps,
                "flags": flags, "seen": seen, "evidence": evidence,
                "bad": 0 if ran else s["bad"] + 1}

    def answer_node(self, s: State) -> State:
        """Final answer from the collected evidence, with the shared answer prompt."""
        evidence, steps, flags = list(s["evidence"]), list(s["steps"]), dict(s["flags"])
        no_local = not any(st.get("tool") == "retrieve_from_index" and st.get("executed") for st in steps)
        if self.guardrail and (not evidence or (self.guardrail_local and no_local)) and "retrieve_from_index" in self.tools:
            query = s["question"]
            for st in reversed(steps):                  # most recent reformulated query, if any
                obs = str(st.get("observation", ""))
                if st.get("tool") == "query_reformulator" and st.get("executed") and obs.startswith("Search query:"):
                    query = obs[len("Search query:"):].strip() or query
                    break
            try:
                obs, meta = self.tools["retrieve_from_index"](query)
                evidence.append(obs)
                flags["guardrail_retrieval"] = 1
                steps.append({"step": s["step"] + 1, "kind": "guardrail", "tool": "retrieve_from_index",
                              "input": query[:300], "observation": obs[:700], "meta": meta, "executed": True})
            except Exception as e:
                flags["tool_error"] = flags.get("tool_error", 0) + 1
        ctx = "\n\n".join(evidence)
        if len(ctx) > MAX_EVIDENCE_CHARS:               # keep the most recent evidence
            ctx = ctx[-MAX_EVIDENCE_CHARS:]
        out = self.answer_llm.generate(answer_prompt(s["question"], ctx), max_new_tokens=48, prefill=ANSWER_PREFILL)
        return {"final": out, "steps": steps, "flags": flags, "evidence": evidence}

    def run(self, question: str) -> Dict[str, Any]:
        return self.graph.invoke(initial_state(question),
                                 config={"recursion_limit": 4 * self.max_steps + 12})

    def run_many(self, questions: List[str], max_concurrency: int = 16):
        """Concurrent runs; combine with a micro-batching LLM wrapper to share the GPU."""
        return self.graph.batch([initial_state(q) for q in questions],
                                config={"max_concurrency": max_concurrency,
                                        "recursion_limit": 4 * self.max_steps + 12})
