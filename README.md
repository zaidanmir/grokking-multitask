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

## Headline result

Single-task addition replicates the grokking phenomenon at the published
hyperparameters of Nanda 2023. With *p* = 113, train fraction 0.30,
AdamW (weight decay = 1.0), the network memorises the training set
within ~1,000 steps, sits at the random-guess test-loss plateau, then
groks: test accuracy first crosses 0.95 at **step 7,200**, reaches
1.000 by step 30,000.

See [`paper/main.pdf`](paper/main.pdf) for the full write-up and
multi-task analysis.

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
backend. All experiments fit in memory; full sweep takes ~3 hours on
an M-series chip. CUDA is also supported (auto-detected). CPU works
but is ~4× slower for this small model size due to the per-kernel
launch overhead being a smaller fraction of the GPU compute.

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
