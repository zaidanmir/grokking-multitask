"""Modular arithmetic task generation.

A token vocabulary of size ``p + 4`` covers digit tokens ``0..p-1`` plus the
operator tokens ``+``, ``-``, ``*`` and the equals token ``=``. Every example
has the same shape — four input tokens ``[a, op, b, =]`` and a target digit
token ``(a op b) mod p``. Predictions are read from the position of the ``=``
token. The fixed input length lets us mix tasks freely in a multi-task batch
without padding.

The multiplication task is defined on the multiplicative group
``(Z/pZ)^*`` for prime ``p``: operands are restricted to ``{1, ..., p-1}``.
Including zero would cap test accuracy at ``1 - 1/p`` for any model that
otherwise solves the task perfectly, because the model would have to memorise
the ``p`` rows of the Cayley table for which the result is ``0``. This is a
well-known foot-gun (see e.g. Nanda 2023 footnote 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

OPS = ("+", "-", "*")


def op_token_id(op: str, p: int) -> int:
    """Token id for an operation, given the modulus."""
    if op == "+":
        return p
    if op == "-":
        return p + 1
    if op == "*":
        return p + 2
    raise ValueError(f"unknown op {op!r}")


def eq_token_id(p: int) -> int:
    return p + 3


def vocab_size(p: int) -> int:
    return p + 4


def _apply_op(a: np.ndarray, b: np.ndarray, op: str, p: int) -> np.ndarray:
    if op == "+":
        return (a + b) % p
    if op == "-":
        return (a - b) % p
    if op == "*":
        return (a * b) % p
    raise ValueError(f"unknown op {op!r}")


def _enumerate_pairs(op: str, p: int) -> tuple[np.ndarray, np.ndarray]:
    """All ordered (a, b) pairs in the operand domain for the given op."""
    if op == "*":
        # Multiplicative group: exclude zero.
        domain = np.arange(1, p)
    else:
        domain = np.arange(0, p)
    a, b = np.meshgrid(domain, domain, indexing="ij")
    return a.reshape(-1), b.reshape(-1)


@dataclass
class TaskData:
    """Tokenised dataset for one or more modular-arithmetic tasks.

    Tensors live on the CPU; move them to the desired device with
    :py:meth:`to`. ``task_ids`` indexes into ``ops`` and lets per-task metrics
    be computed efficiently in the training loop.
    """

    X_train: torch.Tensor          # (N_train, 4) long
    y_train: torch.Tensor          # (N_train,)   long
    task_train: torch.Tensor       # (N_train,)   long, op index
    X_test: torch.Tensor
    y_test: torch.Tensor
    task_test: torch.Tensor
    ops: tuple[str, ...]
    p: int
    train_frac: float
    seed: int

    def to(self, device: torch.device | str) -> "TaskData":
        return TaskData(
            X_train=self.X_train.to(device),
            y_train=self.y_train.to(device),
            task_train=self.task_train.to(device),
            X_test=self.X_test.to(device),
            y_test=self.y_test.to(device),
            task_test=self.task_test.to(device),
            ops=self.ops,
            p=self.p,
            train_frac=self.train_frac,
            seed=self.seed,
        )


def _stratified_split(
    n: int, train_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    return perm[:n_train], perm[n_train:]


def make_task(
    op: str,
    p: int = 113,
    train_frac: float = 0.30,
    seed: int = 42,
) -> TaskData:
    """Build a single-task dataset for one modular-arithmetic operation."""
    return make_multitask((op,), p=p, train_frac=train_frac, seed=seed)


def make_multitask(
    ops: Sequence[str],
    p: int = 113,
    train_frac: float = 0.30,
    seed: int = 42,
) -> TaskData:
    """Build a dataset covering multiple operations with a per-task split.

    Each task is split independently (so each contributes ``train_frac`` of
    its own pairs to training), then the per-task halves are concatenated.
    The shuffle is per-task to keep the test set balanced across operations.
    """
    if len(ops) == 0:
        raise ValueError("ops must contain at least one operation")
    for op in ops:
        if op not in OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
    rng = np.random.default_rng(seed)
    eq = eq_token_id(p)

    X_tr, y_tr, t_tr = [], [], []
    X_te, y_te, t_te = [], [], []
    for task_idx, op in enumerate(ops):
        a, b = _enumerate_pairs(op, p)
        y = _apply_op(a, b, op, p)
        op_tok = np.full_like(a, op_token_id(op, p))
        eq_tok = np.full_like(a, eq)
        X = np.stack([a, op_tok, b, eq_tok], axis=1)  # (N, 4)
        train_idx, test_idx = _stratified_split(len(X), train_frac, rng)
        X_tr.append(X[train_idx])
        y_tr.append(y[train_idx])
        t_tr.append(np.full(len(train_idx), task_idx, dtype=np.int64))
        X_te.append(X[test_idx])
        y_te.append(y[test_idx])
        t_te.append(np.full(len(test_idx), task_idx, dtype=np.int64))

    X_train = torch.from_numpy(np.concatenate(X_tr, axis=0)).long()
    y_train = torch.from_numpy(np.concatenate(y_tr, axis=0)).long()
    task_train = torch.from_numpy(np.concatenate(t_tr, axis=0)).long()
    X_test = torch.from_numpy(np.concatenate(X_te, axis=0)).long()
    y_test = torch.from_numpy(np.concatenate(y_te, axis=0)).long()
    task_test = torch.from_numpy(np.concatenate(t_te, axis=0)).long()

    return TaskData(
        X_train=X_train,
        y_train=y_train,
        task_train=task_train,
        X_test=X_test,
        y_test=y_test,
        task_test=task_test,
        ops=tuple(ops),
        p=p,
        train_frac=train_frac,
        seed=seed,
    )
