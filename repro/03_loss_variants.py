# -*- coding: utf-8 -*-
"""Loss-formulation bisect: which loss shape triggers the backward segfault.

Run **one mode per process**. A segmentation fault terminates the whole
interpreter, so the only way to sweep several formulations is one invocation per
mode.

Modes::

    full     : mean(logits ** 2)                      entire logits tensor
    slice64  : mean(logits[0, -1, :64] ** 2)          first 64 vocab entries
    slice512 : mean(logits[0, -1, :512] ** 2)         first 512 vocab entries
    lasttok  : sum(logits[0, -1])                     last token, full vocab
    hidden   : mean(hidden_states[h] ** 2)            CONTROL: bypasses lm_head

Observed on the affected setup:

    full / slice64 / slice512 / lasttok  ->  segfault (no traceback)
    hidden                               ->  completes normally

Usage::

    python 03_loss_variants.py --mode full
    python 03_loss_variants.py --mode hidden
    python 03_loss_variants.py --mode full --threads 1
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "Hello world"
MODES = ("full", "slice64", "slice512", "lasttok", "hidden")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Which loss formulation triggers the lm_head backward segfault?")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--model",
                    default=os.environ.get("MODEL_ID", DEFAULT_MODEL),
                    help="local path or hub id (env fallback: MODEL_ID)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--threads", type=int, default=1,
                    help="torch.set_num_threads value (default 1)")
    ap.add_argument("--hidden-index", type=int, default=-1,
                    help="hidden_states layer index for --mode hidden")
    args = ap.parse_args()

    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    print("mode=%s num_threads=%d" % (args.mode, torch.get_num_threads()), flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    print("LOADED", flush=True)

    ids = tok(args.prompt, return_tensors="pt").input_ids
    out = model(input_ids=ids, output_hidden_states=True)
    print("FWD ok: logits=%s vocab=%d"
          % (tuple(out.logits.shape), out.logits.shape[-1]), flush=True)

    if args.mode == "full":
        loss = out.logits.float().pow(2).mean()
    elif args.mode == "slice64":
        loss = out.logits[0, -1, :64].float().pow(2).mean()
    elif args.mode == "slice512":
        loss = out.logits[0, -1, :512].float().pow(2).mean()
    elif args.mode == "lasttok":
        loss = out.logits[0, -1].float().sum()
    elif args.mode == "hidden":
        loss = out.hidden_states[args.hidden_index].float().pow(2).mean()
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit("unknown mode: %s" % args.mode)

    print("loss=%.6f" % float(loss.detach()), flush=True)

    if args.mode == "hidden":
        print("calling backward() on hidden-state loss (bypasses lm_head) ...",
              flush=True)
        loss.backward(retain_graph=False)
    else:
        print("calling backward() on logits loss (through lm_head) ...", flush=True)
        loss.backward()

    print("BACKWARD_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
