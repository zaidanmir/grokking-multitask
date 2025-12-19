# Research log

Daily-resolution notes on what was tried, what worked, what didn't, and
what was read. Append-only.

## 2026-05-08 — project kickoff (Weeks 1–4 condensed)

Spinning up the repo and the single-task replication baseline in one
sitting. The main goal of this session is to get from zero to a
reproducible Nanda 2023 replication and a working multi-task training
pipeline.

### Infrastructure

- `python3 -m venv .venv` + `pip install -r requirements.txt`. Torch
  2.11, NumPy 2.4. MPS backend available on this Mac.
- Project lives at `~/Desktop/projects/grokking-multitask`. Mirrors the
  layout of the previous from-scratch repos (`naive-bayes-spam`,
  `mnist-nn-from-scratch`, `transformer-from-scratch`,
  `diffusion-from-scratch`).

### Reading

Re-read Power et al. 2022 (arXiv 2201.02177) and Nanda et al. 2023
(arXiv 2301.05217). The load-bearing piece is Section 4 of Nanda — the
Fourier-multiplication algorithm. Briefly:

- The trained embedding matrix `W_E ∈ R^{p × d}` decomposes cleanly into
  a small number of Fourier basis vectors `cos(2π k n / p)`,
  `sin(2π k n / p)` for `k ∈ {k₁, ..., k_K}` (~5 frequencies).
- The network computes `cos(2π k (a+b) / p)` from features of `a` and
  `b` via the trig identity
  `cos(α)cos(β) − sin(α)sin(β) = cos(α+β)`.
- Multiplication in MLP and attention layers is what implements the
  product-to-sum, hence why a single attention layer + an MLP suffice.

### Implementation choices

- 1-layer decoder-only transformer. `d_model=128`, `n_heads=4`,
  `d_head=32`, `d_mlp=512`. No layer norm (matches Nanda 2023 exactly).
- Hand-rolled multi-head attention so per-head outputs are accessible
  for the ablation experiments. Forgoes `nn.MultiheadAttention`.
- AdamW, `betas=(0.9, 0.98)`, `weight_decay=1.0`, `lr=1e-3`, full-batch.
  These are the published Nanda hyperparameters; weight decay is the
  one parameter that absolutely must be 1.0 for grokking to occur.
- Vocabulary: `p + 4 = 117` for `p = 113`. Tokens `0..112`, then `+`,
  `-`, `*`, `=`. Input is always 4 tokens `[a, op, b, =]`; predict at
  the `=` position. Same input shape across single- and multi-task
  experiments simplifies code reuse.
- Multiplication task: operands restricted to `{1, ..., p-1}`. Including
  zero would cap accuracy at `1 - 1/p`.

### Plan for today

1. Scaffold the repo, get a smoke test passing.
2. Replicate single-task addition grokking. Sanity-check against
   Nanda's reported numbers.
3. Replicate single-task subtract and multiply.
4. Implement Fourier and ablation analysis utilities.
5. Run multi-task experiments (k=2, k=3).
6. Robustness sweep across seeds.
7. Curriculum and (p, train fraction) sweeps.
8. Write the manuscript end-to-end.
9. Build the PDF, push everything to `github.com/zaidanmir/grokking-multitask`.

This compresses Weeks 1–24 of the original plan. The scientific content
is the same — what's been cut is reading time, paper-writing iterations,
and feedback loops.

### Build out

By 01:30 BST the repo has:
- `data/tasks.py`, `src/model.py`, `src/train.py`, `src/eval.py`,
  `src/analysis/{fourier,ablation,attention}.py`.
- `tests/test_pipeline.py`, 13 tests, all passing on Python 3.13 + Torch
  2.11. Tests cover: token ids; per-task targets; multiplication-domain
  exclusion; train/test disjointness; per-task split balance; model
  forward shape; per-head output decomposition (heads sum to total);
  eval per-task slicing; Fourier basis recovers a pure sinusoid;
  multiplicative-group Fourier basis recovers a character; head
  ablation runs without crashing.
- `experiments/01_baseline_addition.py` … `08_robustness_sweeps.py`,
  plus `plot_results.py`, `run_all.py`, and a `Makefile`.
- LaTeX skeleton at `paper/main.tex` with sections drafted top to
  bottom (intro, background, methods, replication, multitask,
  analysis, discussion, future work). Built with tectonic for
  zero-install reproducibility.

### Smoke test

A 1500-step run on `+ mod 23` (small `p` for speed) produced the
expected pattern: train_acc → 1.0 by step 500, test_acc 0.05 → 0.15
by step 1500, on track to grok with more steps. ~6.3 ms/step on the
PyTorch MPS backend.

### Per-step cost on the real configuration

Benchmark (CPU vs MPS) at `p=113`, `train_frac=0.30`:
- CPU: 133 ms/step → 30k steps = 67 min
- MPS: ~30 ms/step → 30k steps ≈ 15 min

Confirms MPS is the right backend even for this tiny model. CPU is
slower because the embedding lookup and small matmuls don't amortise
well over PyTorch's CPU backend.

### Experiment 01 — single-task addition (Nanda baseline replication)

30,000 steps, `p=113`, `train_frac=0.30`, `seed=42`.

| step      | train_loss | test_loss  | train_acc | test_acc |
| --------- | ---------- | ---------- | --------- | -------- |
| 0         | 4.78       | 4.78       | 0.000     | 0.000    |
| ~1,000    | < 1e-3     | ~4.7       | 1.000     | 0.000    |
| 7,200     | < 1e-5     | ~0.05      | 1.000     | 0.951    |
| 30,000    | 5.95e-08   | 3.13e-07   | 1.000     | 1.000    |

**Grok step = 7,200**. Substantially earlier than the typically reported
~25k step in Nanda 2023 (which uses a 0.30 train fraction and the same
hyperparameters), but the replication is unambiguous: train loss
collapses fast, test loss sits at the random-guess plateau for ~3,000
steps, then drops by ~7 orders of magnitude by step 10,000. Wall time
~18 minutes on MPS. Figure looks textbook: blue (train) drops
monotonically with high-frequency oscillations after grokking; orange
(test) stays high then plummets through the grok step marker.

This replication validates the entire pipeline. Every later
experiment is a delta on this baseline.

### Step-count retuning

With Nanda's grokking pattern reproducing at 7-8k steps rather than
25k, I halved the step budgets on every other experiment. New
budgets:

- 02 sub: 15k (was 30k)
- 03 mul: 25k (was 50k) — multiplication is harder so still longer
- 04 two-task add+sub: 20k (was 40k)
- 05 three-task add+sub+mul: 25k (was 60k)
- 06 curriculum: 10k single + 20k three-task (was 20k+40k)
- 07 seed sweep: 25k * 2 (was 60k * 2)
- 08 robustness sweeps: small budgets per cell, unchanged

### Experiment 02 — single-task subtraction

15,000 steps, `seed=42`. Grokked at step ~8,000 — same dynamics as
addition, just slightly later. Test_loss collapses from 16.9 at step
4k to 1.1e-7 at step 8k. Final test_acc 1.000. Wall time 10:15.

Subtraction grokking at the same time as addition makes sense — it's
the same cyclic group `Z/pZ`, so the same Fourier basis works. The
small difference (8k vs 7k for grok) is well within seed-to-seed
variation expected for grokking.

