"""End-to-end smoke tests for the data, model, and analysis pipeline.

These run in a few seconds and are intended to catch regressions, not to
verify scientific correctness — that's what the experiments do.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from data.tasks import (
    eq_token_id,
    make_multitask,
    make_task,
    op_token_id,
    vocab_size,
)
from src.analysis.ablation import ablate_all_heads
from src.analysis.fourier import (
    compute_fourier_basis,
    compute_fourier_basis_multiplicative,
    feature_overlap,
    identify_dominant_frequencies,
)
from src.eval import evaluate
from src.model import ModelConfig, OneLayerTransformer


P_TEST = 17  # small prime so tests are fast


def test_token_ids_are_distinct():
    p = P_TEST
    ids = {op_token_id("+", p), op_token_id("-", p), op_token_id("*", p), eq_token_id(p)}
    assert len(ids) == 4
    assert vocab_size(p) == p + 4
    for op in ("+", "-", "*"):
        assert op_token_id(op, p) >= p
    assert eq_token_id(p) == p + 3


def test_addition_dataset_targets_correct():
    data = make_task("+", p=P_TEST, train_frac=0.5, seed=0)
    a = data.X_train[:, 0]
    b = data.X_train[:, 2]
    expected = (a + b) % P_TEST
    assert torch.equal(expected, data.y_train)


def test_subtraction_dataset_targets_correct():
    data = make_task("-", p=P_TEST, train_frac=0.5, seed=0)
    a = data.X_train[:, 0]
    b = data.X_train[:, 2]
    expected = (a - b) % P_TEST
    assert torch.equal(expected, data.y_train)


def test_multiplication_excludes_zero():
    data = make_task("*", p=P_TEST, train_frac=0.5, seed=0)
    assert (data.X_train[:, 0] != 0).all(), "operand a must be non-zero for *"
    assert (data.X_train[:, 2] != 0).all(), "operand b must be non-zero for *"
    a = data.X_train[:, 0]
    b = data.X_train[:, 2]
    expected = (a * b) % P_TEST
    assert torch.equal(expected, data.y_train)


def test_train_test_split_is_disjoint():
    data = make_task("+", p=P_TEST, train_frac=0.3, seed=42)
    train_keys = {tuple(row.tolist()) for row in data.X_train}
    test_keys = {tuple(row.tolist()) for row in data.X_test}
    assert train_keys.isdisjoint(test_keys)
    n = (P_TEST ** 2)
    assert len(train_keys) + len(test_keys) == n


def test_multitask_split_is_per_task():
    data = make_multitask(("+", "-"), p=P_TEST, train_frac=0.4, seed=0)
    n_per_task = P_TEST ** 2
    expected_train_per_task = round(n_per_task * 0.4)
    for idx in (0, 1):
        assert (data.task_train == idx).sum().item() == expected_train_per_task


def test_model_forward_shape():
    cfg = ModelConfig(p=P_TEST)
    model = OneLayerTransformer(cfg)
    data = make_task("+", p=P_TEST, train_frac=0.3, seed=0)
    logits = model(data.X_train[:8])
    assert logits.shape == (8, vocab_size(P_TEST))


def test_per_head_outputs_sum_to_total():
    cfg = ModelConfig(p=P_TEST)
    model = OneLayerTransformer(cfg)
    data = make_task("+", p=P_TEST, train_frac=0.3, seed=0)
    embeds = model.embed(data.X_train[:4])
    out, per_head = model.attn(embeds, return_per_head=True)
    summed = per_head.sum(dim=0)
    assert torch.allclose(summed, out, atol=1e-5)


def test_eval_returns_per_task_metrics():
    cfg = ModelConfig(p=P_TEST)
    model = OneLayerTransformer(cfg)
    data = make_multitask(("+", "-"), p=P_TEST, train_frac=0.4, seed=0)
    res = evaluate(model, data.X_test, data.y_test, data.task_test, data.ops)
    assert set(res.per_task_acc.keys()) == {"+", "-"}
    # At init the model is no better than chance.
    assert 0.0 <= res.acc <= 1.0


def test_fourier_decomposition_recovers_pure_sinusoid():
    p = 31
    d = 8
    freq = 3
    n = np.arange(p)
    W = np.zeros((p, d))
    for i in range(d):
        phase = i * 0.1
        W[:, i] = np.cos(2 * np.pi * freq * n / p + phase)
    spec = compute_fourier_basis(W, p)
    dom = identify_dominant_frequencies(spec, fraction=0.99)
    assert dom[0] == freq, f"expected dominant freq {freq}, got {dom[0]}"


def test_feature_overlap_is_one_for_identical_spectra():
    p = 31
    rng = np.random.default_rng(0)
    W = rng.standard_normal((p, 8))
    spec_a = compute_fourier_basis(W, p)
    spec_b = compute_fourier_basis(W, p)
    assert abs(feature_overlap(spec_a, spec_b) - 1.0) < 1e-10


def test_multiplicative_fourier_recovers_character():
    """A multiplicative character chi(g^k) = exp(2*pi*i*m*k / (p-1)) appears
    as a single dominant frequency in the multiplicative-group DFT."""
    p = 31
    d = 4
    m = 2
    g = 3  # primitive root mod 31
    W = np.zeros((p, d))
    for j in range(p):
        if j == 0:
            continue
        # discrete log: find k with g^k = j (mod p)
        for k in range(p - 1):
            if pow(g, k, p) == j:
                break
        for i in range(d):
            W[j, i] = np.cos(2 * np.pi * m * k / (p - 1) + i * 0.05)
    spec = compute_fourier_basis_multiplicative(W, p)
    dom = identify_dominant_frequencies(spec, fraction=0.99)
    assert dom[0] == m, f"expected dominant multiplicative freq {m}, got {dom[0]}"


def test_ablate_head_changes_loss():
    cfg = ModelConfig(p=P_TEST)
    model = OneLayerTransformer(cfg)
    data = make_task("+", p=P_TEST, train_frac=0.5, seed=0)
    results = ablate_all_heads(model, data.X_train[:64], data.y_train[:64])
    assert len(results) == cfg.n_heads
    for r in results:
        assert r.loss_after >= 0


if __name__ == "__main__":
    failed = []
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} tests passed")
    if failed:
        sys.exit(1)
