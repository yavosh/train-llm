# Markov Chains to LLMs — Deep Dive Learning Plan

**Date:** 2026-04-18
**Owner:** yavosh
**Duration:** ~1 month (~60+ hours)
**Hardware:** MacBook Pro M1 Pro, 16GB unified memory (MPS-capable)

## Goal

Learn Markov chains and LLMs from first principles through a mix of theory and
from-scratch code. Capstone with a head-to-head comparison of Markov, neural,
and Transformer language models on the same corpus. Then extend with scale,
fine-tuning, and applied-LLM milestones.

## Learning principles

- **From scratch first.** Implement the core math (Markov transitions,
  attention, backprop for the MLP) in NumPy / pure Python before reaching for
  libraries.
- **Ship an artifact every few days.** Each week produces runnable code, a
  measurable result (perplexity, loss curve, sample text), and brief notes.
- **Same corpus across stages.** Tiny Shakespeare (~1MB) through all three
  models so the quality jump at each step is visible. Corpus is swappable
  later.
- **Python + NumPy + PyTorch.** NumPy for the math-heavy early work; PyTorch
  (MPS backend) once we're past tensors-from-scratch.

## Repo layout

```
train-llm/
├── README.md
├── docs/
│   └── superpowers/specs/2026-04-18-markov-to-llm-design.md  (this file)
├── data/
│   └── tiny-shakespeare.txt
├── week1-markov/
│   ├── NOTES.md
│   ├── char_markov.py
│   ├── matrix_markov.py
│   └── perplexity.py
├── week2-neural-bridge/
│   ├── NOTES.md
│   ├── mlp_lm.py
│   └── embeddings_viz.py
├── week3-transformer/
│   ├── NOTES.md
│   ├── attention.py
│   ├── transformer.py
│   └── train.py
├── week4-shootout/
│   ├── RESULTS.md
│   └── compare.py
└── milestones/
    ├── B-scale-transformer.md
    ├── C-finetune-qlora.md
    └── D-rag-or-agent.md
```

## Week 1 — Markov chains

**Theory:**
- Probability refresher (conditional probability, chain rule)
- Discrete-time Markov property, state transitions
- N-gram models; smoothing (Laplace / add-k)
- Perplexity as an evaluation metric
- Why Markov models hit a wall: context length vs. state-space explosion

**Code:**
- `char_markov.py` — character-level 1st / 2nd / Nth-order Markov model using
  Python dicts. Trains, samples, and evaluates log-likelihood.
- `matrix_markov.py` — same model as a transition matrix, vectorized with
  NumPy. Compare performance with the dict version.
- `perplexity.py` — train/val split, perplexity computation, sweep over N to
  produce a plot.

**Artifact:** Shakespeare sampler + perplexity-vs-context-length chart that
motivates the move to neural models.

## Week 2 — Bridge from Markov to neural

**Theory:**
- Word/character embeddings; why dense representations beat one-hot
- Bengio 2003 neural language model
- Softmax and cross-entropy loss
- Backpropagation intuition (chain rule, one hidden layer by hand)
- The attention *idea* before the mechanism: data-dependent weighting

**Code:**
- `mlp_lm.py` — fixed-context-window neural language model
  (Karpathy's `makemore` style). Two versions:
  1. Pure NumPy with hand-written forward + backward pass
  2. PyTorch reimplementation to validate the NumPy version
- `embeddings_viz.py` — PCA / t-SNE plot of learned character embeddings
  to build intuition.

**Artifact:** On the same Tiny Shakespeare split, neural LM achieves lower
perplexity than Markov. Write up *why* in `NOTES.md`.

## Week 3 — Transformers from scratch

**Theory:**
- Scaled dot-product attention (derive, not copy)
- Multi-head attention, positional encoding
- Residual connections, LayerNorm, causal masking
- Decoder-only architecture (GPT family)
- Training mechanics: AdamW, LR warmup + cosine decay, gradient clipping

**Code (PyTorch, MPS):**
- `attention.py` — single-head attention in ~30 lines, unit-tested against a
  hand-computed 3-token example
- `transformer.py` — full decoder-only Transformer, ~10M parameters
  (nanoGPT-scale)
- `train.py` — training loop on Tiny Shakespeare via MPS; logs loss, samples
  periodically, checkpoints

**Artifact:** A trained GPT that generates recognizable Shakespeare-style
text. Expected training time on M1 Pro: ~30–60 minutes.

## Week 4 — Capstone shootout + milestone plan

**Shootout (`week4-shootout/`):**
- Train all three models on the same train/val split of Tiny Shakespeare
- Metrics: validation perplexity, sample quality (blind rating rubric), train
  time, inference tokens/sec, parameter count
- `compare.py` produces the metrics table
- `RESULTS.md` writes up numbers, samples, and personal takeaways

**Follow-on milestones (one per 1-2 weekends, in order):**

- **Milestone B — Scale your Transformer**
  - Swap char-level for a BPE tokenizer (reuse `tiktoken` or train
    `sentencepiece`)
  - Larger corpus (full Shakespeare, or a code corpus)
  - More parameters, longer context, better LR schedule
  - Goal: understand the practical knobs of pretraining

- **Milestone C — Fine-tune a small open model**
  - Base model: TinyLlama-1.1B or Qwen-0.5B (16GB RAM is tight but workable
    with QLoRA / 4-bit quantization)
  - Pick a task: chat persona, code-explainer, or summarizer
  - HuggingFace `transformers` + `peft` + `bitsandbytes`
  - Goal: understand LoRA, quantization, and the HF fine-tuning stack

- **Milestone D — Applied LLM system**
  - Build a RAG system or a simple agent on top of Milestone C's model, or on
    a hosted API if local is too slow for iteration
  - Embeddings + vector store + retrieval + prompt engineering + (optional)
    tool use
  - Goal: LLM engineering skills (evals, prompting, retrieval) on top of the
    internals you now understand

## Deliverables per week

- `NOTES.md` with theory notes and the "why" behind decisions
- Runnable Python code with a top-level `run.sh` or clear command
- One measurable artifact (sample output, loss curve, or metrics table)
- Git commit at end of each week

## Success criteria

- Can explain, without notes, how attention works and why it beats fixed-window
  neural LMs
- Can read a modern LLM paper and follow the architecture section
- Have a Transformer you trained from your own code, and a RESULTS.md
  comparing it quantitatively to baselines
- Milestone plan (B/C/D) ready to execute when Week 4 is done

## Non-goals

- Building a production system
- Distributed / multi-GPU training
- Reinforcement learning from human feedback (RLHF) — out of scope for the
  month; revisit if Milestone D calls for it
- MoE, state-space models, or alternative architectures — keep focus on the
  Transformer lineage
