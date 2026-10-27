"""Smoke test: load MedGemma, answer a few sample questions with no tools.

Checks (1) it loads in the chosen dtype, (2) output is sane (no NaN / garbage in fp16),
(3) generation speed. Run on rtx2080 with --dtype float16 and on l40 with --dtype bfloat16.

Usage: python 05_test_medgemma.py --dtype float16 --n 8
"""
import argparse, time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from common import load_data

p = argparse.ArgumentParser()
p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
p.add_argument("--n", type=int, default=8)
p.add_argument("--max_new_tokens", type=int, default=64)
args = p.parse_args()

dtype = getattr(torch, args.dtype)
mid = "google/medgemma-1.5-4b-it"
print("gpu:", torch.cuda.get_device_name(0), "| dtype:", args.dtype)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=dtype, device_map="auto")
proc = AutoProcessor.from_pretrained(mid)
print(f"loaded in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")

sample, _ = load_data()
SYSTEM = "Answer the biomedical question with a short phrase only (no explanation)."
total_tok, total_t = 0, 0.0
for _, r in sample.head(args.n).iterrows():
    msgs = [{"role": "user", "content": [{"type": "text",
             "text": f"{SYSTEM}\n\nQuestion: {r.question}\nAnswer:"}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(model.device)
    n_in = inp["input_ids"].shape[-1]
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
    dt = time.time() - t
    gen = out[0][n_in:]
    total_tok += len(gen); total_t += dt
    print(f"\nQ: {r.question}\nGOLD: {r.answer}\nPRED: {proc.decode(gen, skip_special_tokens=True).strip()}\n({len(gen)} tok, {dt:.1f}s)")
print(f"\nspeed: {total_tok / total_t:.1f} tok/s, {total_t / args.n:.1f} s/question")
print("peak GPU mem GB:", round(torch.cuda.max_memory_allocated() / 1e9, 1))
