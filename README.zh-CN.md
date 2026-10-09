# 通过 `lm_head` 计算的 loss 在调用 `backward()` 时导致 CPU 段错误

[English](README.md) | 简体中文

一份最小化、自包含的观察报告与复现套件，针对一个很窄的现象：

> 在**纯 CPU** 的 PyTorch 环境下，当一个因果语言模型的 `lm_head` 投影到
> **大词表（151936）**时，由它产生的 `logits` 算出 loss，再调用
> `.backward()`，**进程会以段错误（segmentation fault）直接死亡**。
> **没有任何 Python traceback**——解释器立刻终止。

前向传播完全正常。只有经过词表投影的反向传播会崩溃。

本仓库报告**观察到了什么**、**排除了什么**，以及**一个恰好有效的绕行方法**。
它**不**声称找到了根本原因。在得出任何结论之前，请先看
[这是什么 / 这不是什么](#这是什么--这不是什么)。

---

## 症状

| 项目 | 观察结果 |
|---|---|
| 设备 | 纯 CPU（`torch.cuda.is_available() == False`） |
| 触发条件 | `loss.backward()`，其中 `loss` 由 `logits` 计算（即经过了 `lm_head`） |
| 词表大小 | 151936 |
| 失败形态 | **段错误。没有 Python traceback。进程直接终止。** |
| 前向传播 | 正常（`logits` 与 `hidden_states` 都能正常产生） |
| 反向传播 | 在任何 Python 层错误能被抛出之前就已死亡 |

崩溃不伴随任何异常、堆栈跟踪或 Python 错误信息。唯一可见的信号就是进程结束了。

---

## 最小复现

第一个脚本**不需要联网**、**不需要下载模型**。

```
repro/
  01_minimal_synthetic.py   # 零下载复现：微型随机模型，vocab=151936
  02_real_model.py          # 在真实预训练模型上走同样的路径（--model / 环境变量）
  03_loss_variants.py       # 四种 loss 写法 + 一个 hidden-state 对照
  bisect_steps.md           # 当初如何一步一步把范围缩小到这里的
```

### 01 — 合成模型，零下载

用 `transformers.AutoConfig` +
`AutoModelForCausalLM.from_config(config)` 构建一个**随机初始化**的小模型
（`vocab_size=151936`），然后走标准路径
`input_ids -> model -> logits -> loss -> backward()`。

```bash
# 安全：构建模型和 loss，但不调用 backward()。
python repro/01_minimal_synthetic.py --dry-run

# 完整运行。在受影响的环境上，预期会段错误。
python repro/01_minimal_synthetic.py
```

合成模型刻意做得很小（2 层，hidden size 64）。只有 `lm_head`
保持完整词表大小。如果崩溃与"宽词表投影"有关，这个小模型同样会触发它；
如果崩溃还需要其他因素，这个小模型**可能**复现不出来。那种情况本身就是有用的
信息（见[这是什么 / 这不是什么](#这是什么--这不是什么)）。

### 02 — 真实模型

在真实预训练模型上走同样的 `lm_head -> loss -> backward()` 路径。路径或
hub id 来自 `--model`（或 `MODEL_ID` 环境变量）；默认是一个小的公开模型。

```bash
python repro/02_real_model.py
python repro/02_real_model.py --model Qwen/Qwen2.5-0.5B-Instruct
MODEL_ID=/path/to/local/model python repro/02_real_model.py
```

把 `--model` 指向本地目录即可完全离线运行。

### 03 — loss 变体

每次运行只跑一种 loss 写法。**每种模式必须是独立进程**——因为段错误会杀死
解释器，后面的模式没法在同一进程里继续跑。

```bash
python repro/03_loss_variants.py --mode full
python repro/03_loss_variants.py --mode hidden   # 对照：绕开 lm_head
```

完整过程见 [`repro/bisect_steps.md`](repro/bisect_steps.md)。

---

## 我们排除了什么

### 已排除：线程数

把 `torch.set_num_threads` 设为 **1、4、8**——三种情况**全部**段错误。
所以崩溃**不是**线程数问题。

```bash
python repro/02_real_model.py --threads 1
python repro/02_real_model.py --threads 4
python repro/02_real_model.py --threads 8
```

### 已排除：loss 的形状 / 大小

试了四种 loss 写法，全部由 `logits` 算出（即全部经过 `lm_head`）：

| 模式 | Loss |
|---|---|
| `full` | 全部 logit 平方的均值（整个张量） |
| `slice64` | 最后一个 token 的前 64 个词表条目的 logit 平方均值 |
| `slice512` | 最后一个 token 的前 512 个词表条目的 logit 平方均值 |
| `lasttok` | 最后一个 token 的完整 logit 向量之和 |

**四种全部段错误。** 所以崩溃**不是** loss 触及多少词表条目的函数，
也与 loss 的大小/形状无关。

### 有效的绕行方法

在 **hidden-state 空间**里算 loss，即**不**经过 `lm_head`：

```python
out = model(input_ids=ids, output_hidden_states=True)
loss = out.hidden_states[-1].float().pow(2).mean()
loss.backward()          # 在观察到的环境中可以正常完成
```

复现该对照：

```bash
python repro/03_loss_variants.py --mode hidden
```

---

## 运维备注

真实模型运行时也传了 `low_cpu_mem_usage=True`。把它列在这里只是因为它属于
观察发生时的确切配置之一。它**不被**声称是必需的。

> **更正（2026-10-09）。** 本备注的早期版本曾声称没有这个参数会导致加载失败。
> 该说法在锁定的环境上不成立：在 `transformers==5.13.1` 中，`low_cpu_mem_usage`
> 关键字参数会被无条件地从加载参数中丢弃（见 `transformers/modeling_utils.py`，
> 注释 "Not used anymore -- remove them from the kwargs"），并且经验证该参数设为
> `True` 与 `False` 时加载行为完全相同。
> 如果你在没有这个参数时观察到加载失败，那是与这里锁定版本不同的
> `transformers` 版本。更正保留在原处，不做删除。

---

## 版本矩阵

观察发生时的确切配置。完整矩阵见 [`versions.md`](versions.md)。

| 组件 | 版本 |
|---|---|
| Python | 3.14.4 |
| torch | 2.13.0+cpu（纯 CPU 构建） |
| transformers | 5.13.1 |
| CUDA | 不可用（纯 CPU） |

安装锁定版本：

```bash
pip install -r requirements.txt
```

`requirements.txt` 假定使用 **CPU-only** 构建的 torch。

---

## 这是什么 / 这不是什么

- 这是一个**环境与版本相关**的观察。在其他操作系统、Python 版本、torch 构建、
  硬件或模型配置上未必成立。在推广任何结论之前，请先对照**完全相同**的版本
  矩阵复现。
- **没有做任何根因分析。** 我们**不**声称这是 torch、transformers
  或任何其他组件的 bug。我们只报告：观察到的症状、被排除的备选解释，
  以及在这里有效的一个绕行方法。
- 段错误按它出现的原样报告：**没有 traceback，进程就是死了。**
  本仓库的任何地方都没有编造错误信息。
- 阴性结果（一种**不**崩溃的配置）与阳性结果同样有用。欢迎独立复现与反馈。

---

## 免责声明

本软件按"原样"提供，不附带任何形式的明示或暗示保证。运行完整（非
`--dry-run`）脚本可能终止你的进程。见 [`LICENSE`](LICENSE)。

---

## 许可证

MIT — 见 [`LICENSE`](LICENSE)。
