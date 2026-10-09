# Version matrix

Precise versions of the environment in which the backward segmentation fault was
observed. Reproduce against these exact versions before drawing conclusions.

## Core

| Component | Version | Notes |
|---|---|---|
| Python | 3.14.4 | CPython |
| torch | 2.13.0+cpu | **CPU-only build** (`torch.cuda.is_available() == False`) |
| transformers | 5.13.1 | |
| CUDA | unavailable | no GPU / CPU-only runtime |

## Host

| Item | Value |
|---|---|
| OS | Windows 11 (x86-64) |
| Device | CPU only |
| Accelerator | none |

## Model configuration in which it was observed

| Item | Value |
|---|---|
| `lm_head` output size (vocabulary) | 151936 |
| Model class | causal LM (`AutoModelForCausalLM`) |
| Attention implementation | `eager` |
| Load flag | `low_cpu_mem_usage=True` **(no-op in `transformers==5.13.1`)** |

> **Note on the load flag.** In `transformers==5.13.1` the `low_cpu_mem_usage` kwarg
> is unconditionally discarded during loading, so it has no effect on this version.
> It is recorded here only because it was present in the observed configuration.
> Loading was verified to behave identically with the flag `True` and `False`.

## What was varied (all still crashed)

| Varied | Values tried | Result |
|---|---|---|
| `torch.set_num_threads` | 1, 4, 8 | segfault in all three |
| Loss formulation | `full`, `slice64`, `slice512`, `lasttok` | segfault in all four |

## What did NOT crash

| Varied | Value | Result |
|---|---|---|
| Loss location | hidden-state space (bypasses `lm_head`) | completes |

## Reproducibility caveats

- Multi-threaded reductions are not bit-deterministic. Numerical outputs may vary
  between runs even on the same machine.
- This observation is not claimed to be portable. Different OSes, Python
  versions, torch builds, or hardware may behave differently (including not
  reproducing at all).
- No root cause was established. This document records versions only; it does not
  attribute the behaviour to any component.
