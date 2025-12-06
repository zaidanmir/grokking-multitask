# grokking-multitask

A study of grokking dynamics in a 1-layer transformer trained jointly on
multiple modular arithmetic tasks (addition, subtraction, multiplication
mod *p*).

This project replicates the single-task addition grokking phenomenon of
[Nanda et al. 2023](https://arxiv.org/abs/2301.05217), then extends the
analysis to a multi-task setting and asks whether the discovered
Fourier features are shared across tasks or task-specific.

## Status

Active. See [`LOG.md`](LOG.md) for a daily research log and the in-progress
manuscript at [`paper/main.pdf`](paper/main.pdf).

## What grokking is, in two paragraphs

Train a small neural network on a synthetic task such as
`(a + b) mod 113`, with a hard 30/70 train/test split and aggressive
weight decay. The network first memorises the training set: training
loss drops to zero within a few hundred steps, while test loss stays at
random-guessing level. Then, after tens of thousands of *additional*
training steps with no apparent progress, test loss abruptly collapses
and the model generalises. This is **grokking** (Power et al. 2022).

[Nanda et al. 2023](https://arxiv.org/abs/2301.05217) showed that during
the apparent plateau the network is silently constructing a discrete-
Fourier-transform-based algorithm for modular addition, and that
weight-decay-driven implicit regularisation is what tips it from
memorisation circuits to generalising circuits.

## Research question

When a single 1-layer transformer is trained jointly on
**add, subtract, and multiply mod *p***, does grokking still occur on
each task? Are the discovered Fourier features shared across tasks or
task-specific? Does multi-task training accelerate, delay, or
qualitatively change the grokking dynamics relative to single-task
training?

## Repo layout

```
data/         modular-arithmetic task generation
src/          model, training loop, eval
src/analysis/ Fourier decomposition, ablations, attention vis
experiments/  one Python entrypoint per experiment
paper/        NeurIPS-format LaTeX source + figures
tests/        pipeline tests
```

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python experiments/01_baseline_addition.py
```

Each experiment writes a YAML config + per-step metrics to `runs/<name>/`
and final figures to `paper/figures/`. Random seeds 42, 137, 271 are
used for the three primary replicates.

## Citation

If you use this code or ideas, please cite the manuscript (see
`paper/main.pdf`).

## License

MIT, see [LICENSE](LICENSE).
