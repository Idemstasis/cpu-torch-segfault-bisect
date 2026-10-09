# CPU segfault when calling `backward()` on a loss computed through `lm_head`

English | [简体中文](README.zh-CN.md)

A minimal, self-contained report and reproduction kit for one narrow observation:

> On a **CPU-only** PyTorch setup, computing a loss from the `logits` produced by a
> causal language model whose `lm_head` projects to a **large vocabulary
> (151936)**, and then calling `.backward()`, **kills the process with a
> segmentation fault**. There is **no Python traceback** — the interpreter dies
> immediately.

The forward pass is fine. Only the backward pass through the vocabulary
projection crashes.

This repository reports **what was observed**, **what was ruled out**, and **one
workaround that happens to work**. It does **not** claim a root cause. See
[What this is / is not](#what-this-is--is-not) before drawing conclusions.

---

## Symptom

| Item | Observed |
|---|---|
| Device | CPU only (`torch.cuda.is_available() == False`) |
| Trigger | `loss.backward()` where `loss` is derived from `logits` (i.e. passes through `lm_head`) |
| Vocabulary size | 151936 |
| Failure mode | **Segmentation fault. No Python traceback. The process terminates.** |
| Forward pass | Succeeds (`logits` and `hidden_states` are produced normally) |
| Backward pass | Dies before any Python-level error can be raised |

The crash is not accompanied by an exception, a stack trace, or a Python error
message of any kind. The only visible signal is that the process ends.

---

## Minimal repro

The first script needs **no network access** and **no model download**.

```
repro/
  01_minimal_synthetic.py   # zero-download repro: tiny random model, vocab=151936
  02_real_model.py          # same path on a real pretrained model (--model / env)
  03_loss_variants.py       # four loss formulations + a hidden-state control
  bisect_steps.md           # how the behaviour was narrowed down, step by step
```

### 01 — synthetic, zero download

Builds a small **randomly initialized** model with `vocab_size=151936` using
`transformers.AutoConfig` + `AutoModelForCausalLM.from_config(config)`, then runs
the standard path `input_ids -> model -> logits -> loss -> backward()`.

```bash
# Safe: builds the model and the loss, but does NOT call backward().
python repro/01_minimal_synthetic.py --dry-run

# Full run. On an affected setup this is expected to segfault.
python repro/01_minimal_synthetic.py
```

The synthetic model is deliberately tiny (2 layers, hidden size 64). Only
`lm_head` is kept at the full vocabulary size. If the crash is tied to the wide
vocabulary projection, this small model still exercises it; if the crash requires
other factors, this model may **not** reproduce it. That case is itself useful
information (see [What this is / is not](#what-this-is--is-not)).

### 02 — real model

Same `lm_head -> loss -> backward()` path on a real pretrained model. The path or
hub id is taken from `--model` (or the `MODEL_ID` environment variable); the
default is a small public model.

```bash
python repro/02_real_model.py
python repro/02_real_model.py --model Qwen/Qwen2.5-0.5B-Instruct
MODEL_ID=/path/to/local/model python repro/02_real_model.py
```

Point `--model` at a local directory to work fully offline.

### 03 — loss variants

Runs one loss formulation per invocation. **Each mode must be a separate
process**, because a segfault kills the interpreter and no further mode could run
after it.

```bash
python repro/03_loss_variants.py --mode full
python repro/03_loss_variants.py --mode hidden   # control: bypasses lm_head
```

See [`repro/bisect_steps.md`](repro/bisect_steps.md) for the full procedure.

---

## What we ruled out

### Ruled out: thread count

Setting `torch.set_num_threads` to **1, 4, and 8** — the process segfaults in
**all three** cases. So the crash is **not** a thread-count problem.

```bash
python repro/02_real_model.py --threads 1
python repro/02_real_model.py --threads 4
python repro/02_real_model.py --threads 8
```

### Ruled out: loss shape / scale

Four different loss formulations were tried, all computed from `logits` (i.e. all
passing through `lm_head`):

| Mode | Loss |
|---|---|
| `full` | mean of squared logits, entire tensor |
| `slice64` | mean of squared logits, first 64 vocab entries of the last token |
| `slice512` | mean of squared logits, first 512 vocab entries of the last token |
| `lasttok` | sum of the full logit vector of the last token |

**All four segfault.** So the crash is **not** a function of how much of the
vocabulary the loss touches, nor of the loss magnitude/shape.

### Effective workaround

Compute the loss in **hidden-state space**, i.e. do **not** pass through `lm_head`:

```python
out = model(input_ids=ids, output_hidden_states=True)
loss = out.hidden_states[-1].float().pow(2).mean()
loss.backward()          # completes in the observed environment
```

Reproduce the comparison with:

```bash
python repro/03_loss_variants.py --mode hidden
```

---

## Operational note

The real-model run also passed `low_cpu_mem_usage=True`. It is listed here only
because it was part of the exact configuration in which the observation was made.
It is **not** claimed to be necessary.

> **Correction (2026-10-09).** An earlier revision of this note stated that loading
> failed without this flag. That statement does not hold in the pinned environment:
> in `transformers==5.13.1` the `low_cpu_mem_usage` kwarg is unconditionally dropped
> from the loading kwargs (see `transformers/modeling_utils.py`, comment
> "Not used anymore -- remove them from the kwargs"), and loading was verified to
> behave identically with the flag set to `True` and to `False`.
> If you observe a load failure without this flag, it is on a different
> `transformers` version than the one pinned here. Corrections are left in place
> rather than removed.

---

## Version matrix

Exact configuration in which the observation was made. See
[`versions.md`](versions.md) for the full matrix.

| Component | Version |
|---|---|
| Python | 3.14.4 |
| torch | 2.13.0+cpu (CPU-only build) |
| transformers | 5.13.1 |
| CUDA | unavailable (CPU only) |

Install the pinned versions with:

```bash
pip install -r requirements.txt
```

`requirements.txt` assumes the **CPU-only** build of torch.

---

## What this is / is not

- This is an **environment- and version-specific observation**. It may not hold
  on other operating systems, Python versions, torch builds, hardware, or model
  configurations. Reproduce it against the **exact** version matrix before
  generalising anything.
- **No root-cause analysis was performed.** We do **not** claim this is a bug in
  torch, in transformers, or anywhere else. We report only: the observed symptom,
  the alternatives that were ruled out, and one workaround that works here.
- The segmentation fault is reported exactly as it appears: **no traceback, the
  process simply dies.** No error text is invented anywhere in this repository.
- Negative results (a configuration that does **not** crash) are just as useful as
  positive ones. Independent reproductions and feedback are welcome.

---

## Disclaimer

This software is provided "as is", without warranty of any kind, express or
implied. Running the full (non-`--dry-run`) scripts may terminate your process.
See [`LICENSE`](LICENSE).

---

## License

MIT — see [`LICENSE`](LICENSE).
