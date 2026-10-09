# -*- coding: utf-8 -*-
"""Reproduce the lm_head backward segfault on a *real* pretrained model.

The model path or hub id is taken from ``--model`` (or the ``MODEL_ID``
environment variable). The default is a small public model so the script works
out of the box when network access is available; point ``--model`` at a local
directory to stay fully offline.

Usage::

    python 02_real_model.py
    python 02_real_model.py --model Qwen/Qwen2.5-0.5B-Instruct
    MODEL_ID=/path/to/local/model python 02_real_model.py
    python 02_real_model.py --threads 1     # thread-count check (1, 4, 8 all crash)

Path under test: input_ids -> model -> logits -> loss -> backward().
On affected setups the backward() call terminates the process with a
segmentation fault and no Python traceback.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "Hello world"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="lm_head backward segfault repro on a real pretrained model.")
    ap.add_argument("--model",
                    default=os.environ.get("MODEL_ID", DEFAULT_MODEL),
                    help="local path or hub id (env fallback: MODEL_ID)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads value; 0 keeps the default")
    args = ap.parse_args()

    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    print("num_threads=%d" % torch.get_num_threads(), flush=True)
    print("torch=%s cuda_available=%s"
          % (torch.__version__, torch.cuda.is_available()), flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    print("transformers=%s" % transformers.__version__, flush=True)

    dtype = getattr(torch, args.dtype)

    print("loading model: %s ..." % args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    print("loaded", flush=True)

    ids = tok(args.prompt, return_tensors="pt").input_ids
    print("input_ids=%s" % (tuple(ids.shape),), flush=True)

    # Forward without grad, just to confirm the forward path is healthy.
    with torch.no_grad():
        _ = model(input_ids=ids).logits[0, -1].clone()
    print("forward (no_grad) ok", flush=True)

    out = model(input_ids=ids)
    print("forward (with grad) ok: logits=%s"
          % (tuple(out.logits.shape),), flush=True)

    # Loss derived from logits -> passes through lm_head.
    loss = out.logits.float().pow(2).mean()
    print("loss=%.6f" % float(loss.detach()), flush=True)

    print("calling backward() ...", flush=True)
    loss.backward()
    grad_present = any(p.grad is not None for p in model.parameters())
    print("BACKWARD OK: grad_present=%s" % grad_present, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
