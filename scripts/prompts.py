"""Answer prompt shared by every system's final generation step.

MUST stay identical to prompt "v3" in 06_baselines.py (v2 wording + pre-filled "A:"), so that
Baseline 0, the static chains and the agent all use the same answer generator.
"""
HEAD = ("You are a biomedical expert. Reply with ONLY the answer: a short noun phrase "
        "(at most 6 words), no explanation, no full sentence. When PubMed abstracts are "
        "provided, copy the exact term they use, including its modifiers.\n\n")
EXAMPLE = ("Examples of the answer format:\nQ: What is the capital of France?\nA: Paris\n\n"
           "Q: Which gas do plants absorb during photosynthesis?\nA: Carbon dioxide\n\n")
PREFILL = "A:"


def answer_prompt(question, context=""):
    ctx = (f"Use the PubMed abstracts below if they help.\n\nContext:\n{context}\n\n" if context else "")
    return HEAD + ctx + EXAMPLE + f"Q: {question}\nA:"
