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
