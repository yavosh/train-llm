# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A self-study project for learning Markov chains and LLMs from scratch. The human
owner (yavosh) is the one executing the learning plan — your role is to help
unblock, explain concepts, review their code, and fix tooling issues, **not** to
implement the exercises for them. When they ask for help on a week's task,
prefer hints and Socratic prompts over writing the solution outright.

Exception: infrastructure / scaffolding (project setup, data download, CI,
environment issues) is fine to do fully.

## Source of truth

- **Spec:** `docs/superpowers/specs/2026-04-18-markov-to-llm-design.md` — the
  "what and why"
- **Plan:** `docs/superpowers/plans/2026-04-18-markov-to-llm.md` — the
  task-by-task "how", with checkbox steps

When the user references "the plan" or "Week N" or "Task N.M", look here first.
Update the checkboxes in the plan as tasks complete.

## Stack

- Python 3.11+, managed with `uv`
- NumPy for the from-scratch math (Weeks 1–2)
- PyTorch 2.x with the **MPS** backend for Week 3 onward (M1 Pro, 16 GB)
- `matplotlib` for charts, `tiktoken` for BPE (milestone B onward)

Expected commands once Task 0 runs:

```bash
uv sync                    # install deps
uv run python path/to/script.py
uv run pytest              # run tests
uv run ruff check .        # lint
```

None of this exists yet — the repo currently contains only docs.

## Planned layout

```
week1-markov/         N-gram models + perplexity sweep
week2-neural-bridge/  MLP LM (NumPy + PyTorch), embedding viz
week3-transformer/    attention → Transformer → trained GPT
week4-shootout/       head-to-head comparison + retrospective
milestones/           follow-on stubs: B (scale), C (QLoRA), D (RAG/agent)
data/                 Tiny Shakespeare corpus (gitignored)
```

Each week directory will contain a `NOTES.md` (theory) and a `CHECKPOINT.md`
(self-assessment) alongside code. Shared utilities (e.g. `load_shakespeare()`
from Week 1) should be imported, not copy-pasted.

## Hardware constraints worth remembering

- MPS, not CUDA — watch for PyTorch ops that silently fall back to CPU
- 16 GB unified memory — anything ≥ 1B params needs 4-bit quantization (QLoRA)
  and will still be tight
- Tiny Shakespeare is small enough that full-batch eval is fine; don't optimize
  for scale that isn't needed

## Workflow

- Feature branches + PRs for any change that isn't the initial bootstrap; no
  direct pushes to `main`
- Conventional Commits (`feat:`, `docs:`, `chore:`, etc.)
- Don't commit model checkpoints (`.pt`), data files, or notebook outputs —
  `.gitignore` covers these

## Non-goals

Beating SOTA, distributed training, RLHF, MoE/Mamba. Keep focus on the learning
objectives in the spec.
