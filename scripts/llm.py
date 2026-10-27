"""MedGemma wrapper: batched, greedy, single-turn text generation.

MedGemma is not evaluated for multi-turn chat, so every call is ONE user message
(the ReAct agent will pack its whole scratchpad into that single prompt).
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-1.5-4b-it"


class MedGemma:
    def __init__(self, dtype="bfloat16", model_id=MODEL_ID):
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto").eval()

    def _generate(self, prompts, max_new_tokens, prefill=None, stop=None):
        msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
        texts = [self.proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                 for m in msgs]
        if prefill:  # start the model turn ourselves, e.g. "A:", so it cannot open with "thought"
            texts = [t + prefill for t in texts]
        # the template already contains <bos>, so do not add special tokens again
        inp = self.proc.tokenizer(texts, return_tensors="pt", padding=True,
                                  add_special_tokens=False).to(self.model.device)
        n_in = inp["input_ids"].shape[1]
        with torch.inference_mode():
            kw = dict(stop_strings=stop, tokenizer=self.proc.tokenizer) if stop else {}
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, **kw)
        return [self.proc.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]

    def generate(self, prompts, max_new_tokens=64, prefill=None, stop=None):
        """prompts: list[str] -> list[str]. Falls back to one-by-one if batching fails.
        The returned text does NOT include the prefill.
        stop: list of strings; each sequence stops once it has generated one of them."""
        try:
            return self._generate(prompts, max_new_tokens, prefill, stop)
        except Exception as e:
            print("batched generation failed, falling back to single:", repr(e)[:200])
            return [self._generate([p], max_new_tokens, prefill, stop)[0] for p in prompts]


class CausalLM:
    """Any chat-tuned causal LM (e.g. ibm-granite/granite-4.0-micro) with the MedGemma interface:
    batched greedy generation, pre-filled model turn, stop strings. Returned text excludes the prefill."""

    def __init__(self, model_id, dtype="bfloat16"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto").eval()

    def _generate(self, prompts, max_new_tokens, prefill=None, stop=None):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
        if prefill:
            texts = [t + prefill for t in texts]
        inp = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        n_in = inp["input_ids"].shape[1]
        with torch.inference_mode():
            kw = dict(stop_strings=stop, tokenizer=self.tok) if stop else {}
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id, **kw)
        return [self.tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]

    def generate(self, prompts, max_new_tokens=64, prefill=None, stop=None):
        try:
            return self._generate(prompts, max_new_tokens, prefill, stop)
        except Exception as e:
            print("batched generation failed, falling back to single:", repr(e)[:200])
            return [self._generate([p], max_new_tokens, prefill, stop)[0] for p in prompts]


import queue, threading, time
from collections import defaultdict
from concurrent.futures import Future


class BatchingLLM:
    """Thread-safe front for MedGemma: many graph runs (threads) call generate() with ONE prompt
    each; a worker thread merges concurrent requests into a single batched GPU call.

    generate(str)  -> str          generate(list[str]) -> list[str]
    """

    def __init__(self, llm, max_batch=16, wait_s=0.05):
        self.llm, self.max_batch, self.wait_s = llm, max_batch, wait_s
        self.q = queue.Queue()
        self.n_calls = self.n_requests = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def generate(self, prompts, max_new_tokens=64, prefill=None, stop=None):
        single = isinstance(prompts, str)
        futs = []
        for p in ([prompts] if single else prompts):
            f = Future()
            self.q.put((p, max_new_tokens, prefill, tuple(stop) if stop else None, f))
            futs.append(f)
        outs = [f.result() for f in futs]
        return outs[0] if single else outs

    def _loop(self):
        while True:
            batch = [self.q.get()]
            t0 = time.time()
            while len(batch) < self.max_batch and time.time() - t0 < self.wait_s:
                try:
                    batch.append(self.q.get(timeout=self.wait_s))
                except queue.Empty:
                    break
            groups = defaultdict(list)
            for item in batch:
                groups[item[1:4]].append(item)       # same (max_new_tokens, prefill, stop)
            for (mnt, prefill, stop), items in groups.items():
                try:
                    outs = self.llm.generate([i[0] for i in items], max_new_tokens=mnt,
                                             prefill=prefill, stop=list(stop) if stop else None)
                    self.n_calls += 1; self.n_requests += len(items)
                    for i, o in zip(items, outs):
                        i[4].set_result(o)
                except Exception as e:
                    for i in items:
                        i[4].set_exception(e)
