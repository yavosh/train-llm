# train-llm

A 1-month self-study project: learning Markov chains and LLMs from first
principles by implementing them from scratch.

Builds three language models of increasing complexity on the same corpus
(Tiny Shakespeare), then runs a head-to-head comparison:

1. **Markov chain** — NumPy, dict and matrix variants, N-gram sweep
2. **MLP language model** — manual backprop in NumPy, then PyTorch
3. **Transformer** — single-head attention → multi-head → nanoGPT-scale GPT,
   trained on Apple MPS

After the shootout, the plan extends with three follow-on milestones:
scaling the Transformer, QLoRA fine-tuning a small open model, and an
applied LLM system (RAG / agent).

## Status

Planning complete. Implementation hasn't started yet.

## Progress

Task checkboxes live in the [implementation plan](docs/superpowers/plans/2026-04-18-markov-to-llm.md) — tick them there as you go. This table is a rollup.

| Phase | Tasks | Status | Artifact |
|---|---|---|---|
| **Task 0 — Bootstrap** | uv env, deps, corpus, dirs | ⬜ not started | working Python env |
| **Week 1 — Markov** | 1.1 notes · 1.2 dict model · 1.3 matrix model · 1.4 perplexity sweep · 1.5 checkpoint | ⬜ not started | `week1-markov/perplexity-vs-order.png` |
| **Week 2 — Neural bridge** | 2.1 notes · 2.2 NumPy MLP · 2.3 PyTorch MLP · 2.4 embedding viz · 2.5 checkpoint | ⬜ not started | `week2-neural-bridge/mlp-loss.png` |
| **Week 3 — Transformer** | 3.1 notes · 3.2 attention · 3.3 multi-head · 3.4 training · 3.5 checkpoint | ⬜ not started | trained GPT + `train-loss.png` |
| **Week 4 — Shootout** | 4.1 harness · 4.2 blind rating · 4.3 retro + milestones · 4.4 final writeup | ⬜ not started | `week4-shootout/RESULTS.md` |
| **Milestone B — Scale** | BPE tokenizer, larger corpus, bigger model | ⬜ queued | — |
| **Milestone C — QLoRA** | fine-tune Qwen-0.5B on a task | ⬜ queued | — |
| **Milestone D — Applied LLM** | RAG or simple agent | ⬜ queued | — |

Legend: ⬜ not started · 🟡 in progress · ✅ done

## Documents

- **Design spec** — [`docs/superpowers/specs/2026-04-18-markov-to-llm-design.md`](docs/superpowers/specs/2026-04-18-markov-to-llm-design.md)
- **Implementation plan** — [`docs/superpowers/plans/2026-04-18-markov-to-llm.md`](docs/superpowers/plans/2026-04-18-markov-to-llm.md)

## Planned layout

```
week1-markov/         N-gram models + perplexity sweep
week2-neural-bridge/  MLP LM (NumPy + PyTorch), embedding viz
week3-transformer/    attention → Transformer → trained GPT
week4-shootout/       head-to-head comparison + retrospective
milestones/           B: scale, C: QLoRA fine-tune, D: RAG/agent
```

## Stack

Python 3.11, NumPy, PyTorch (MPS), matplotlib, uv for env. Hardware target:
MacBook Pro M1 Pro, 16 GB.

## Corpus

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt),
~1 MB. Same train/val split across all three models for a fair comparison.
