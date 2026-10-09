# Bisect record: how the backward segfault was narrowed down

A short, honest record of the three steps that turned "the process dies on
`backward()`" into a specific, repeatable statement:

> **Any loss that passes through `lm_head` (vocabulary 151936) segfaults on
> `backward()`; a loss computed in hidden-state space does not.**

Notes that apply to every step:

- The failure mode is always the same: **segmentation fault, no Python
  traceback, the process ends.** No error text is reproduced here because none is
  produced.
- Each variant must be run in its **own process**. A segfault is unrecoverable in
  Python, so a single process cannot try more than one crashing variant.
- All runs are CPU-only (`torch.cuda.is_available() == False`).

---

## Step 1 — Isolate the class and the attention implementation

**Question:** Does the model load and run a forward pass at all, and is `logits`
actually produced? Which model class and which attention implementation are in
use?

**Method:** Load the model via `AutoModelForCausalLM`, with
`attn_implementation="eager"`, run a forward pass with `output_hidden_states=True`
under `torch.no_grad()`, and report the shape of the produced `logits` and
`hidden_states`.

**Observation:**

- Model loads successfully.
- Forward pass succeeds; `logits` has the expected shape with last dimension
  `151936`, and `hidden_states` is populated.
- `torch.cuda.is_available()` is `False` (CPU only).

**Conclusion:** The crash is **not** in loading and **not** in the forward pass.
It is specific to the **backward** pass. This step establishes that everything up
to `logits` is healthy, which is what makes the next two steps meaningful.

---

## Step 2 — Rule out thread count

**Question:** Is the crash a threading / thread-count problem?

**Method:** Repeat the same `logits -> loss -> backward()` run with
`torch.set_num_threads` set to **1**, **4**, and **8** (in separate processes).

**Observation:**

| `torch.set_num_threads` | Result |
|---|---|
| 1 | segfault |
| 4 | segfault |
| 8 | segfault |

**Conclusion:** **All three crash.** The crash is **not** caused by the thread
count. Forcing single-threaded execution does not avoid it.

Reproduce with:

```bash
python 02_real_model.py --threads 1
python 02_real_model.py --threads 4
python 02_real_model.py --threads 8
```

---

## Step 3 — Rule out loss shape / scale, and find the workaround

**Question:** Is the crash a function of *which part* or *how much* of the logits
tensor the loss touches? And does a loss that bypasses `lm_head` behave
differently?

**Method:** Four different losses derived from `logits` (all passing through
`lm_head`), plus one control loss computed directly on a hidden state (which does
**not** pass through `lm_head`). One process per mode:

| Mode | Loss | Passes through `lm_head`? |
|---|---|---|
| `full` | `mean(logits ** 2)`, entire tensor | yes |
| `slice64` | `mean(logits[0, -1, :64] ** 2)` | yes |
| `slice512` | `mean(logits[0, -1, :512] ** 2)` | yes |
| `lasttok` | `sum(logits[0, -1])`, last token only | yes |
| `hidden` | `mean(hidden_states[h] ** 2)` (control) | **no** |

**Observation:**

| Mode | Result |
|---|---|
| `full` | segfault |
| `slice64` | segfault |
| `slice512` | segfault |
| `lasttok` | segfault |
| `hidden` | **completes normally** |

**Conclusion:**

- The four `lm_head` losses all crash regardless of shape or scale (full tensor
  down to a single token's logits). So the crash is **not** a function of loss
  shape or magnitude.
- The single distinguishing factor between the crashing and non-crashing cases is
  whether the loss **passes through `lm_head`**. The hidden-state control, which
  never touches `lm_head`, completes.

Reproduce with:

```bash
python 03_loss_variants.py --mode full
python 03_loss_variants.py --mode slice64
python 03_loss_variants.py --mode slice512
python 03_loss_variants.py --mode lasttok
python 03_loss_variants.py --mode hidden
```

---

## Effective workaround

Compute the loss in hidden-state space:

```python
out = model(input_ids=ids, output_hidden_states=True)
loss = out.hidden_states[-1].float().pow(2).mean()
loss.backward()
```

Caveat: this is a **different objective**. A hidden-state loss is **not**
equivalent to a cross-entropy loss over the vocabulary. It is a way to make *a*
backward pass run, **not** a drop-in replacement for LM training.

---

## What was *not* done

- **No root-cause analysis.** The bisect narrows *which operation* is involved
  (`lm_head` in the backward graph) but does not explain *why*. No claim is made
  that this is a bug in torch, in transformers, or anywhere else.
- No probing of internal memory layout, no debugger attachment, no core-dump
  inspection.
- The result is reported as an environment- and version-specific observation. See
  `../versions.md` for the exact matrix. It may not hold elsewhere.
