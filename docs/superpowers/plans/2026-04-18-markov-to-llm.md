# Markov Chains to LLMs — Implementation Plan

> **For agentic workers:** This plan is primarily executed by **the human learner** (yavosh). Each task is a concrete study+code chunk with a verification checkpoint. Steps use `- [ ]` for progress tracking.

**Goal:** Learn Markov chains and LLMs end-to-end by implementing them from scratch, culminating in a Markov-vs-MLP-vs-Transformer shootout on Tiny Shakespeare.

**Architecture:** Three language models built in increasing complexity, sharing a single corpus and train/val split so results are directly comparable. Week 1 = NumPy Markov. Week 2 = NumPy-then-PyTorch MLP LM. Week 3 = PyTorch Transformer on MPS. Week 4 = shootout + milestone roadmap.

**Tech Stack:** Python 3.11+, NumPy, PyTorch 2.x (MPS backend), matplotlib, uv (env), Tiny Shakespeare dataset.

**Spec:** `docs/superpowers/specs/2026-04-18-markov-to-llm-design.md`

---

## Task 0: Project bootstrap

**Files:**
- Create: `README.md`, `.gitignore`, `pyproject.toml`, `data/README.md`
- Create: `week1-markov/`, `week2-neural-bridge/`, `week3-transformer/`, `week4-shootout/`, `milestones/` (dirs with `.gitkeep`)

- [ ] **Step 1: Install uv if not present**

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

- [ ] **Step 2: Initialize Python project**

```bash
cd /Users/yavosh/Projects/yavosh/train-llm
uv init --python 3.11
uv add numpy matplotlib torch tiktoken jupyter
uv add --dev pytest ruff
```

- [ ] **Step 3: Add `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
*.pt
*.ckpt
data/*.txt
!data/README.md
runs/
wandb/
.DS_Store
```

- [ ] **Step 4: Download Tiny Shakespeare**

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tiny-shakespeare.txt
wc -c data/tiny-shakespeare.txt  # expect ~1,115,394
```

- [ ] **Step 5: Create week dirs**

```bash
for d in week1-markov week2-neural-bridge week3-transformer week4-shootout milestones; do
  mkdir -p "$d" && touch "$d/.gitkeep"
done
```

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "chore: scaffold project and fetch tiny shakespeare"
```

---

## Week 1 — Markov chains

### Task 1.1: Theory notes — probability, Markov property, perplexity

**Files:** Create `week1-markov/NOTES.md`

- [ ] **Step 1: Read (pick one path, ~2 hrs)**
  - **Primary:** Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), **Ch. 3 (N-gram Language Models)** — free: https://web.stanford.edu/~jurafsky/slp3/
  - **Alt (video):** Stanford CS124 Week 2 (same author, n-gram LMs)

- [ ] **Step 2: In `NOTES.md`, answer in your own words:**
  1. State the Markov property in one sentence
  2. Why does perplexity = `exp(cross_entropy)` in nats?
  3. What does "bigram" mean in terms of the order of the Markov model? (trick: bigram = order-1)
  4. Why does Laplace smoothing exist? What problem does it solve?
  5. Intuition: what happens to an N-gram model's generalization as N grows?

- [ ] **Step 3: Commit**

```bash
git add week1-markov/NOTES.md && git commit -m "docs(week1): theory notes on n-gram LMs"
```

### Task 1.2: Char-level Markov model (dict-based)

**Files:** Create `week1-markov/char_markov.py`

- [ ] **Step 1: Target interface**

```python
class CharMarkov:
    def __init__(self, order: int): ...
    def fit(self, text: str) -> None: ...
    def sample(self, n: int, seed: str = "", rng=None) -> str: ...
    def log_prob(self, text: str) -> float: ...  # sum of log p(c_t | context)
```

Store transitions as `dict[str, dict[str, int]]` — context (last `order` chars) → next char → count. Use add-k smoothing with `k=1` by default.

- [ ] **Step 2: Implement `fit`**

Walk the text with a sliding window of size `order`. For each window, increment `counts[context][next_char]`.

- [ ] **Step 3: Implement `sample`**

Start from `seed` (pad with a sentinel `"\0" * order` if empty). At each step, look up the context's next-char distribution and sample from it. Use `numpy.random.Generator` for reproducibility.

- [ ] **Step 4: Implement `log_prob`**

For each position `t >= order`, compute `log((counts[context][c_t] + k) / (sum(counts[context].values()) + k * V))` where `V` is vocab size. Sum across all positions.

- [ ] **Step 5: Smoke test**

```python
if __name__ == "__main__":
    text = open("data/tiny-shakespeare.txt").read()
    m = CharMarkov(order=4)
    m.fit(text)
    print(m.sample(200, seed="ROMEO: "))
```

Expected: recognizable fragments of words, some pseudo-Shakespeare syntax.

- [ ] **Step 6: Commit**

```bash
git add week1-markov/char_markov.py && git commit -m "feat(week1): dict-based char markov model"
```

### Task 1.3: Matrix-based Markov (NumPy vectorized)

**Files:** Create `week1-markov/matrix_markov.py`

- [ ] **Step 1: Rationale in a comment block**

Dict approach is O(contexts) memory. Matrix approach: fix a vocab `V` of unique chars. Transition matrix shape `(V**order, V)`. Row = context index, column = next-char probability. Fast sampling via `np.random.choice` with a full row.

- [ ] **Step 2: Implement `class MatrixMarkov`**

- `_encode(context: str) -> int` — base-V integer encoding
- `fit` — build dense `(V**order, V)` counts matrix; add-k smooth; normalize rows to probabilities
- `sample` — at each step, take the current context's row and `np.random.choice` with `p=row`

- [ ] **Step 3: Practical limit**

With `V≈65` (Tiny Shakespeare chars), `order=3` gives ~275K rows × 65 cols ≈ 140MB of float64 — fine. `order=4` ≈ 9GB — don't. Document this in the file.

- [ ] **Step 4: Smoke test — same samples as dict version**

Seed both `CharMarkov` and `MatrixMarkov` with the same RNG and verify sample() produces the same string for small orders. Commit a small `pytest` assertion in `week1-markov/test_markov.py`.

- [ ] **Step 5: Commit**

```bash
git add week1-markov/matrix_markov.py week1-markov/test_markov.py
git commit -m "feat(week1): numpy matrix markov model"
```

### Task 1.4: Perplexity + N-sweep chart

**Files:** Create `week1-markov/perplexity.py`

- [ ] **Step 1: Shared train/val split utility**

At the top of the file:

```python
def load_shakespeare(path="data/tiny-shakespeare.txt", val_frac=0.1):
    text = open(path).read()
    split = int(len(text) * (1 - val_frac))
    return text[:split], text[split:]
```

You'll reuse this in every week. Resist copy-paste — import it.

- [ ] **Step 2: Perplexity function**

```python
def perplexity(model, text: str) -> float:
    # perplexity = exp(-mean log-prob per token)
    import math
    return math.exp(-model.log_prob(text) / len(text))
```

- [ ] **Step 3: Sweep and plot**

Train `CharMarkov(order=N)` on train for `N in [1, 2, 3, 4, 5, 6]`. Compute val perplexity for each. Plot `matplotlib` line chart, save `week1-markov/perplexity-vs-order.png`.

- [ ] **Step 4: Interpret in NOTES.md**

Add a section: where does perplexity start rising again (overfitting)? Which N is best? What does that tell you about state-space explosion?

- [ ] **Step 5: Commit**

```bash
git add week1-markov/perplexity.py week1-markov/perplexity-vs-order.png week1-markov/NOTES.md
git commit -m "feat(week1): perplexity sweep and chart"
```

### Task 1.5: Week 1 checkpoint

- [ ] **Self-check — can you explain without notes:**
  - Why higher-order Markov models overfit
  - The definition of perplexity and why lower is better
  - Why dicts are more memory-efficient than matrices at high order

If not, revisit NOTES.md before moving on. Commit a `week1-markov/CHECKPOINT.md` with your answers.

---

## Week 2 — Bridge from Markov to neural

### Task 2.1: Theory notes — embeddings, softmax, backprop

**Files:** Create `week2-neural-bridge/NOTES.md`

- [ ] **Step 1: Read / watch (~3 hrs)**
  - Karpathy, *"The spelled-out intro to neural networks and backpropagation: building micrograd"* (YouTube, ~2.5 hr) — essential
  - Bengio et al. 2003, *A Neural Probabilistic Language Model* — skim abstract + §2 + §3
  - Jurafsky & Martin Ch. 7 (Neural Networks) — reference

- [ ] **Step 2: Answer in NOTES.md**
  1. Why does a one-hot vector carry no similarity information?
  2. What does an embedding lookup table learn? (hint: nearby in embedding space ↔ nearby in distribution)
  3. Derive cross-entropy loss from max-likelihood on softmax output (one line, not rigorous)
  4. In one paragraph: what is backpropagation actually doing?

- [ ] **Step 3: Commit**

### Task 2.2: NumPy MLP LM with manual backprop

**Files:** Create `week2-neural-bridge/mlp_lm_numpy.py`

- [ ] **Step 1: Architecture**

Fixed-window LM, context length `block_size=3` (predict char `t` from chars `t-3, t-2, t-1`).

```
input (3 char indices) → embedding (V, d_emb=16) → flatten → Linear(3*16, 64) → tanh → Linear(64, V) → softmax → cross-entropy
```

- [ ] **Step 2: Implement in NumPy**

No autograd. Hand-derive gradients:
- `dL/dlogits = softmax(logits) - one_hot(y)`
- `dL/dW2 = h.T @ dL/dlogits`, `dL/db2 = sum`
- `dL/dh = dL/dlogits @ W2.T`, `dL/dpre = dL/dh * (1 - tanh²(pre))`
- `dL/dW1 = emb_flat.T @ dL/dpre`, etc.
- `dL/demb` scatters back into the embedding table

Keep under 150 lines. Use mini-batches of 32.

- [ ] **Step 3: Train**

SGD, lr=0.01, 5000 steps. Print train loss every 500 steps. Expected final loss: ~2.4 nats/token.

- [ ] **Step 4: Sanity check — gradient check**

Pick one weight, compute numerical gradient `(L(w+ε) - L(w-ε)) / 2ε` and compare to your analytical gradient. Should match to ~1e-5.

- [ ] **Step 5: Commit**

### Task 2.3: PyTorch MLP LM (same architecture, autograd)

**Files:** Create `week2-neural-bridge/mlp_lm_torch.py`

- [ ] **Step 1: Reimplement as `torch.nn.Module`**

Same architecture exactly. Use `F.cross_entropy`, `torch.optim.AdamW`.

- [ ] **Step 2: Train and verify**

Same hyperparameters. Final val perplexity should be similar to the NumPy version (both should be clearly better than Markov N=4).

- [ ] **Step 3: Plot train + val loss curves**

Save `week2-neural-bridge/mlp-loss.png`.

- [ ] **Step 4: Commit**

### Task 2.4: Embedding visualization

**Files:** Create `week2-neural-bridge/embeddings_viz.py`

- [ ] **Step 1: Load trained embeddings**

Load the `(V, d_emb)` embedding matrix from your PyTorch checkpoint.

- [ ] **Step 2: Project to 2D**

Use `sklearn.decomposition.PCA` or `sklearn.manifold.TSNE`. Plot each char at its 2D position, annotated with the char itself.

- [ ] **Step 3: Interpret in NOTES.md**

Which chars cluster? Vowels together? Punctuation together? Space vs. newline? This is your first visceral "the model learned something" moment.

- [ ] **Step 4: Commit**

### Task 2.5: Week 2 checkpoint

- [ ] **Compare Markov vs MLP in NOTES.md:**
  - Val perplexity of Markov N=4 vs MLP
  - Sample text from both — which is better and why?
  - What did the MLP learn that Markov cannot?
- [ ] **Commit `week2-neural-bridge/CHECKPOINT.md`**

---

## Week 3 — Transformer from scratch

### Task 3.1: Theory — attention, Transformer architecture

**Files:** Create `week3-transformer/NOTES.md`

- [ ] **Step 1: Read / watch (~4 hrs)**
  - Karpathy, *"Let's build GPT: from scratch, in code, spelled out"* (YouTube, ~2 hr) — essential
  - Vaswani et al. 2017, *Attention Is All You Need* — read §3 carefully
  - Lilian Weng blog, *"The Transformer Family"* — reference

- [ ] **Step 2: Derive scaled dot-product attention by hand**

Three tokens, `d_k=2`. Make up Q, K, V matrices. Compute `softmax(QK^T / sqrt(d_k)) V` on paper. Photograph or re-type in NOTES.md.

- [ ] **Step 3: Answer in NOTES.md**
  1. Why divide by `sqrt(d_k)`? (hint: variance of the dot product)
  2. What does the causal mask do, and why is it triangular?
  3. Why is multi-head attention better than a single head of equivalent width?
  4. What does LayerNorm do, and where does it go relative to the residual?

- [ ] **Step 4: Commit**

### Task 3.2: Single-head attention from scratch

**Files:** Create `week3-transformer/attention.py`, `week3-transformer/test_attention.py`

- [ ] **Step 1: Hand-computed test case**

Using the 3-token example from Task 3.1, write a pytest that feeds those exact Q, K, V tensors into your attention function and asserts the output matches your hand calculation to 1e-5.

- [ ] **Step 2: Implement**

```python
def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    # q, k, v: (B, T, d_k); mask: (T, T) lower-triangular of 0s and -inf
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores + mask
    weights = scores.softmax(dim=-1)
    return weights @ v
```

- [ ] **Step 3: Run test**

```bash
uv run pytest week3-transformer/test_attention.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

### Task 3.3: Multi-head attention + Transformer block

**Files:** Create `week3-transformer/transformer.py`

- [ ] **Step 1: `class MultiHeadAttention(nn.Module)`**

Params: `n_embd`, `n_head`. Projects to `qkv` with a single `Linear(n_embd, 3*n_embd)`, splits into heads, applies your `attention` function per head (or use the batched form directly), concats, projects back.

- [ ] **Step 2: `class FeedForward(nn.Module)`**

Two-layer MLP: `Linear(n_embd, 4*n_embd) → GELU → Linear(4*n_embd, n_embd) → Dropout`.

- [ ] **Step 3: `class Block(nn.Module)` — pre-norm style**

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ff(self.ln2(x))
    return x
```

- [ ] **Step 4: `class GPT(nn.Module)`**

Token embedding + positional embedding + `n_layer` blocks + final LN + LM head (tied to token embedding is fine but optional). Target config: `n_embd=192, n_head=6, n_layer=6, block_size=128` → ~2-3M params (small enough for M1 Pro to train fast, big enough to work).

- [ ] **Step 5: Parameter count sanity check**

Print total params. Should match a back-of-envelope calculation (±10%).

- [ ] **Step 6: Forward pass smoke test**

Random input of shape `(2, 128)` → output shape `(2, 128, V)`. No errors.

- [ ] **Step 7: Commit**

### Task 3.4: Training loop on MPS

**Files:** Create `week3-transformer/train.py`

- [ ] **Step 1: Data loader**

Encode Tiny Shakespeare as a long `torch.long` tensor. `get_batch(split)` samples random start indices and returns `(x, y)` of shape `(B, block_size)` each, where `y = x` shifted by 1.

- [ ] **Step 2: Training loop**

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT(...).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

Loop: 5000 iters, batch size 32. Every 500 iters, run eval (avg loss over 200 val batches) and generate a 200-char sample from the current model. LR schedule: linear warmup for 100 steps then cosine decay.

- [ ] **Step 3: Expected numbers**

Final train loss ~1.4–1.6. Val loss ~1.5–1.7. Samples should form real words and occasional recognizable Shakespearean cadence.

- [ ] **Step 4: Checkpoint the model**

Save `week3-transformer/gpt.pt`. Don't commit the `.pt` file (`.gitignore` covers it).

- [ ] **Step 5: Save loss curve**

`week3-transformer/train-loss.png`.

- [ ] **Step 6: Commit**

```bash
git add week3-transformer/train.py week3-transformer/train-loss.png
git commit -m "feat(week3): nanogpt training on MPS"
```

### Task 3.5: Week 3 checkpoint

- [ ] **Self-check:**
  - Can you draw the Transformer block architecture from memory?
  - Can you explain what each of Q, K, V "means" intuitively?
  - Is your Transformer's val perplexity better than MLP and Markov?
- [ ] **Commit `week3-transformer/CHECKPOINT.md`**

---

## Week 4 — Shootout + milestone plan

### Task 4.1: Unified evaluation harness

**Files:** Create `week4-shootout/compare.py`

- [ ] **Step 1: Shared load**

```python
from week1_markov.perplexity import load_shakespeare
train, val = load_shakespeare()
```

(You may need to rename dirs to valid Python modules or use path hackery — easiest: add `__init__.py` files and run from repo root with `uv run python -m week4_shootout.compare`.)

- [ ] **Step 2: For each model, compute and tabulate:**

| Model | Params | Val perplexity | Train time | Inference tok/s | Sample (first 200 chars) |

Models: `Markov(order=4)`, `MLP-torch`, `GPT`.

- [ ] **Step 3: Save table**

Write results as Markdown table into `week4-shootout/RESULTS.md`.

- [ ] **Step 4: Commit**

### Task 4.2: Sample-quality blind rating

**Files:** Update `week4-shootout/RESULTS.md`

- [ ] **Step 1: Generate 5 samples per model**

200 chars each, same random seed across models (for reproducibility, not identical output).

- [ ] **Step 2: Shuffle and rate 1-5 on:**
  - Word validity
  - Syntactic structure
  - "Feels like Shakespeare"

Do the rating blind — have the script emit samples without labels, rate them, then reveal. (Yes, it's a sample size of 1 rater. It's still useful.)

- [ ] **Step 3: Write up findings**

Where does the Transformer win decisively? Where does Markov surprisingly hold up? What's the biggest subjective quality gap?

- [ ] **Step 4: Commit**

### Task 4.3: Retrospective + milestone roadmap

**Files:** Create `week4-shootout/RETROSPECTIVE.md`, `milestones/B-scale-transformer.md`, `milestones/C-finetune-qlora.md`, `milestones/D-rag-or-agent.md`

- [ ] **Step 1: Retrospective**

What surprised you? What took longest? What would you redo? What do you still not understand? (Be honest — this is for you.)

- [ ] **Step 2: Milestone B stub — scale your Transformer**

Specific next steps, not vague intentions:
- Swap char tokenizer for `tiktoken.get_encoding("gpt2")` (BPE)
- Train on full Shakespeare corpus (Project Gutenberg, ~5MB)
- Config bump: `n_embd=384, n_head=6, n_layer=6, block_size=256`
- Expected training time on M1 Pro: ~3-4 hrs
- Success: perplexity on held-out Shakespeare < your Week 3 model

- [ ] **Step 3: Milestone C stub — QLoRA fine-tune**

- Base model: `Qwen/Qwen2.5-0.5B` (smaller, faster on 16GB)
- Stack: `transformers`, `peft`, `bitsandbytes` (check MPS compatibility; may need CPU fallback for quantization)
- Task: pick one concrete task (e.g., "code explainer — takes Python, outputs plain-English description")
- Dataset: 500-2000 examples either curated by you or from an existing dataset
- Success: qualitative eval on 20 held-out examples beats the base model

- [ ] **Step 4: Milestone D stub — applied LLM**

- Option 1: RAG over your own documents (embeddings via `sentence-transformers`, FAISS or SQLite-vss)
- Option 2: Simple ReAct agent with 2-3 tools
- Base LLM: your Milestone-C model, or a hosted API for speed of iteration
- Success: a demo that actually solves a problem you have

- [ ] **Step 5: Commit**

```bash
git add week4-shootout/ milestones/
git commit -m "docs(week4): shootout results, retrospective, milestone plan"
```

### Task 4.4: Final writeup — README

**Files:** Update `README.md`

- [ ] **Step 1: Write project overview**

Short: what this is, what you built, results table (copy from RESULTS.md), links to each week's NOTES.md.

- [ ] **Step 2: Add "What I learned" section**

Bullet list of your biggest takeaways — this is the memorable artifact you'll refer back to.

- [ ] **Step 3: Commit**

```bash
git add README.md && git commit -m "docs: top-level writeup"
```

---

## Checkpoints & pacing

- **Week 1 target:** 10-15 hrs over ~7 days
- **Week 2 target:** 15-20 hrs (MLP with manual backprop is the time sink)
- **Week 3 target:** 15-20 hrs (Transformer + training)
- **Week 4 target:** 8-10 hrs

If a week overruns by >50%, pause and re-plan. Don't grind through.

## Success criteria (end of Week 4)

- [ ] Three working language models in the repo
- [ ] `RESULTS.md` with quantitative comparison
- [ ] You can explain attention without notes
- [ ] You can read a modern LLM paper's architecture section and follow it
- [ ] Milestones B/C/D have concrete next-step plans

## Non-goals

- Beating SOTA on anything
- Distributed training
- RLHF (out of scope for this month)
- MoE / Mamba / other alternative architectures
