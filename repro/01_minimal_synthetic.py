# -*- coding: utf-8 -*-
"""Minimal synthetic reproduction: CPU backward through lm_head.

Builds a *randomly initialized* causal language model (no downloads, no network)
whose vocabulary size matches the environment where a segmentation fault was
observed (``vocab_size=151936``), then runs the standard LM path::

    input_ids -> model -> logits -> loss -> backward()

The model is deliberately tiny (few layers, small hidden size); only ``lm_head``
is kept at the full vocabulary size, since the vocabulary projection is the
component under test.

Usage::

    python 01_minimal_synthetic.py --dry-run   # build model + loss only; NO backward
    python 01_minimal_synthetic.py             # full run (may segfault on affected setups)

On affected setups the full run terminates the process with a segmentation fault
and no Python traceback. See ../README.md and ./bisect_steps.md for context.
"""

from __future__ import annotations

import argparse
import sys


def _build_config(vocab_size: int, hidden_size: int, num_layers: int,
                  num_heads: int, max_positions: int):
    """Create a small causal-LM config with a full-size vocabulary.

    Uses ``transformers.AutoConfig`` only; no weights are downloaded.
    """
    from transformers import AutoConfig

    kwargs = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=max_positions,
    )
    try:
        return AutoConfig.for_model("qwen2", **kwargs)
    except Exception:
        # Fallback for transformers builds where for_model cannot resolve the
        # architecture; still config-only and offline.
        from transformers.models.qwen2 import Qwen2Config
        return Qwen2Config(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Minimal synthetic repro of an lm_head backward segfault.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the model and the loss, but do NOT call backward()")
    ap.add_argument("--vocab-size", type=int, default=151936)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("python=%s" % sys.version.split()[0], flush=True)

    import torch
    print("torch=%s cuda_available=%s"
          % (torch.__version__, torch.cuda.is_available()), flush=True)

    try:
        from transformers import AutoModelForCausalLM
        import transformers
    except ImportError:
        print("ERROR: transformers is required. Run: pip install -r requirements.txt",
              flush=True)
        return 2

    print("transformers=%s" % transformers.__version__, flush=True)
    torch.manual_seed(args.seed)

    config = _build_config(args.vocab_size, args.hidden_size, args.num_layers,
                           args.num_heads, max_positions=128)
    # from_config() builds the model from the config with random weights; no
    # download, no network access.
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print("model built: params=%d vocab=%d layers=%d hidden=%d"
          % (n_params, args.vocab_size, args.num_layers, args.hidden_size),
          flush=True)

    input_ids = torch.randint(0, args.vocab_size, (1, args.seq_len))
    out = model(input_ids=input_ids)
    print("forward ok: logits=%s" % (tuple(out.logits.shape),), flush=True)

    # A loss derived from logits: it therefore passes through lm_head.
    loss = out.logits.float().pow(2).mean()
    print("loss=%.6f" % float(loss.detach()), flush=True)

    if args.dry_run:
        print("DRY-RUN: skipping backward(); this path cannot segfault", flush=True)
        return 0

    print("calling backward() ...", flush=True)
    loss.backward()
    grad_present = any(p.grad is not None for p in model.parameters())
    print("BACKWARD OK: grad_present=%s" % grad_present, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
