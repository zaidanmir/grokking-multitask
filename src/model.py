"""1-layer decoder-only transformer matching Nanda 2023.

The architecture is intentionally minimal — no layer norm, no dropout,
learned positional embeddings, a single attention block followed by a
single MLP. The attention is hand-rolled so per-head outputs are
independently addressable for ablation studies.

Shape conventions
-----------------
``B`` is batch, ``T`` is sequence length (always 4 here: ``[a, op, b, =]``),
``H`` is the number of heads, ``D`` is ``d_model``, ``Dh`` is ``d_head``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    p: int = 113
    seq_len: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    extra_vocab: int = 4   # +, -, *, =
    init_scale: float = 1.0

    @property
    def vocab_size(self) -> int:
        return self.p + self.extra_vocab


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention without LN.

    ``W_QKV`` is stored as a single matrix to keep the implementation tight,
    but the heads are visible at the output level — :py:meth:`forward` returns
    per-head contributions to the residual stream when ``return_per_head`` is
    ``True``, which is what the ablation utilities consume.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_QKV = nn.Linear(cfg.d_model, 3 * cfg.n_heads * cfg.d_head, bias=False)
        self.W_O = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, return_per_head: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head
        qkv = self.W_QKV(x)                          # (B, T, 3*H*Dh)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)      # (B, H, T, Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)   # (B, H, T, T)
        # Causal mask. The result is read at the last position so the mask
        # affects only the intermediate positions; included for parity with
        # standard decoder transformers.
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        z = attn @ v                                  # (B, H, T, Dh)
        # Per-head residual contributions: split W_O column-wise so each
        # head has its own d_model output, summed by W_O implicitly.
        z_perm = z.transpose(1, 2).reshape(B, T, H * Dh)
        out = self.W_O(z_perm)
        if return_per_head:
            # Build the per-head output by zeroing all other heads in turn.
            per_head = torch.zeros(H, B, T, D, device=x.device, dtype=x.dtype)
            for h in range(H):
                z_h = torch.zeros_like(z_perm)
                z_h[:, :, h * Dh : (h + 1) * Dh] = z_perm[:, :, h * Dh : (h + 1) * Dh]
                per_head[h] = self.W_O(z_h)
            return out, per_head
        return out


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp, bias=True)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class OneLayerTransformer(nn.Module):
    """A single attention block + MLP, no layer norm, no dropout.

    Notes
    -----
    The output projection ``W_U`` is not tied to the embedding ``W_E``.
    Tying makes the Fourier story slightly cleaner (Nanda 2023) but the
    interpretability story is unchanged.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.W_pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg)
        self.W_U = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        s = self.cfg.init_scale
        nn.init.normal_(self.W_E.weight, std=s / math.sqrt(self.cfg.d_model))
        nn.init.normal_(self.W_pos.weight, std=s / math.sqrt(self.cfg.d_model))
        nn.init.normal_(self.attn.W_QKV.weight, std=s / math.sqrt(self.cfg.d_model))
        nn.init.normal_(self.attn.W_O.weight, std=s / math.sqrt(self.cfg.n_heads * self.cfg.d_head))
        nn.init.normal_(self.mlp.fc1.weight, std=s / math.sqrt(self.cfg.d_model))
        nn.init.zeros_(self.mlp.fc1.bias)
        nn.init.normal_(self.mlp.fc2.weight, std=s / math.sqrt(self.cfg.d_mlp))
        nn.init.zeros_(self.mlp.fc2.bias)
        nn.init.normal_(self.W_U.weight, std=s / math.sqrt(self.cfg.d_model))

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        return self.W_E(tokens) + self.W_pos(pos)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        # Read out at the last position only — that's where the [=] token sits.
        last = x[:, -1, :]
        return self.W_U(last)

    def forward_with_internals(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the network and return per-stage activations for analysis."""
        embeds = self.embed(tokens)
        attn_out, per_head = self.attn(embeds, return_per_head=True)
        x_post_attn = embeds + attn_out
        mlp_out = self.mlp(x_post_attn)
        x_final = x_post_attn + mlp_out
        last = x_final[:, -1, :]
        logits = self.W_U(last)
        return {
            "embeds": embeds,
            "attn_out": attn_out,
            "attn_per_head": per_head,         # (H, B, T, D)
            "x_post_attn": x_post_attn,
            "mlp_out": mlp_out,
            "x_final": x_final,
            "logits": logits,
        }

    def parameter_groups_for_adamw(
        self, weight_decay: float
    ) -> list[dict]:
        """Apply weight decay everywhere — that's the published recipe."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay}]
