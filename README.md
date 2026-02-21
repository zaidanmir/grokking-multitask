# grokking-multitask

A study of grokking dynamics in a 1-layer transformer trained jointly on
multiple modular arithmetic tasks (addition, subtraction, multiplication
mod *p*).

This project replicates the single-task addition grokking phenomenon of
[Nanda et al. 2023](https://arxiv.org/abs/2301.05217) and extends the
analysis to a multi-task setting, asking whether the discovered Fourier
features are shared across tasks or task-specific.

## What grokking is, in two paragraphs

Train a small neural network on a synthetic task such as
`(a + b) mod 113`, with a hard 30/70 train/test split and aggressive
weight decay. The training loss collapses within a few hundred steps,
so the network has memorised the training set. The test loss does not
move. After tens of thousands of *additional* steps with no apparent
progress, the test loss abruptly collapses and the model generalises.
This is **grokking** ([Power et al. 2022](https://arxiv.org/abs/2201.02177)).

[Nanda et al. 2023](https://arxiv.org/abs/2301.05217) showed that during
the apparent plateau the network is silently constructing a discrete-
Fourier-transform algorithm for modular addition, and that
weight-decay-driven implicit regularisation is what tips the optimiser
from memorisation circuits to generalising circuits.

## Research question

When a single 1-layer transformer is trained jointly on
**add, subtract, and multiply mod *p***, does grokking still occur on
each task? Are the discovered Fourier features shared across tasks or
task-specific? Does multi-task training accelerate, delay, or
qualitatively change the grokking dynamics relative to single-task
training?

The two additive operations live on the cyclic group
$\mathbb{Z}/p\mathbb{Z}$ of order *p*; multiplication lives on the
multiplicative group $(\mathbb{Z}/p\mathbb{Z})^*$ of order *p*−1. These
are *different* cyclic groups, so a single residual stream that solves
all three tasks must either reuse one basis for both or carve out
task-specific sub-spaces.

## Headline results

1. **Single-task** addition, subtraction, and multiplication all
   grok cleanly at the published hyperparameters of Nanda 2023 (*p*
   = 113, train fraction 0.30, AdamW with weight decay 1.0). Grok
   steps: addition **7,200**, subtraction **7,000**, multiplication
   **4,200**.
2. **Two-task** training on $+$ and $-$ groks *faster* than either
   single-task baseline (5,800 / 6,000), suggesting that the shared
   additive Fourier basis is discovered more quickly under joint
   gradient pressure.
3. **Three-task** training on $+$, $-$, and $\times$ does *not* grok
   in 75,000 steps. Instead, test accuracy rises smoothly through
   training, reaching 0.60 / 0.48 / 0.32 on the three tasks. We
   argue this slow-ramp dynamic reflects a residual-stream capacity
   competition between the additive and multiplicative Fourier
   circuits. This is the paper's main empirical contribution.
4. **Mechanistic analysis.** Single-task models cleanly use the
   Fourier basis appropriate to their group operation: add and sub
   concentrate power on a small set of additive-group frequencies;
   multiplication uses the multiplicative-group basis. Per-head
   ablations on the multi-task model show partial specialisation:
   one head is critical for the additive tasks while heads 0+3 carry
   the multiplicative work.

See [`paper/main.pdf`](paper/main.pdf) for the full manuscript.

## Repo layout

```
data/         modular-arithmetic task generation
src/          model, training loop, eval
src/analysis/ Fourier decomposition, ablations, attention vis
experiments/  one Python entrypoint per experiment + plot_results
paper/        NeurIPS-format LaTeX source + figures
tests/        pipeline smoke tests (13 tests)
runs/         per-step metric histories + checkpoints (gitignored)
```

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test           # 13 unit tests
make replicate      # single-task baselines (~30 min total on MPS)
make multitask      # add+sub, add+sub+mul, seed sweep
make figures        # generate paper figures from metric histories
make paper          # build paper PDF (needs tectonic)
```

Or run everything end-to-end:

```bash
make all
```

Each experiment writes a YAML config + per-step metric NPZ + JSONL log
to `runs/<name>/`. Random seeds 42, 137, 271 are used for the three
primary replicates. Final figures are written to `paper/figures/`.

## Hardware

Designed to run on a 2024 Apple-silicon laptop using PyTorch's MPS
backend. The single-task baselines and the two-task multitask
experiment fit comfortably in ~30 minutes each. The three-task
multitask experiments (`05`, `06_stage_b`, `07`) are much slower in
wall-clock time on MPS (~5 hours each at 75k steps) — they benefit
substantially from a CUDA GPU. CPU works but is ~4× slower for this
small model size due to per-kernel launch overhead.

### Running unfinished experiments on a CUDA box

Final checkpoints and per-step metric histories are committed in
`runs/`, so the project is resumable across machines:

```bash
git clone https://github.com/zaidanmir/grokking-multitask
cd grokking-multitask
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python experiments/run_all.py --skip-existing
PYTHONPATH=. python experiments/finalize.py
```

Runs whose `final.pt` is already in `runs/` are skipped. The
`06_curriculum.py` script is also resumable mid-script: stage 1 of
the curriculum (10k addition warm-start) is skipped if
`runs/06_curriculum_stage_a_add/final.pt` exists.

The remaining experiments after the laptop run is:
- `05_multitask_three` (75k steps, multi-task add+sub+mul) — already
  completed at 75k on MPS; partial generalisation (test acc ~0.5)
  but did not fully grok.
- `06_curriculum_stage_b_multi` (65k steps from a grokked-addition
  warm-start)
- `07_multitask_three_seed_137` (75k steps, additional seed)
- `08_robustness_p59_tf30` (40k steps, small-p sweep cell)

## Citation

If you use this code or ideas, please cite the manuscript:

```bibtex
@misc{mir2026grokking,
  title  = {Multi-task grokking on modular arithmetic: do shared Fourier features emerge?},
  author = {Mir, Zaidan},
  year   = {2026},
  note   = {Manuscript: paper/main.pdf}
}
```

## License

[MIT](LICENSE).
