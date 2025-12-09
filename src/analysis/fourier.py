"""Fourier decomposition of the trained embedding matrix.

For a 1-layer transformer trained to grok ``(a op b) mod p`` the digit
embeddings ``W_E[0:p]`` lie close to a low-dimensional subspace spanned by
the Fourier basis vectors of ``Z/pZ``. ``compute_fourier_basis`` returns the
power spectrum (one number per frequency) and ``identify_dominant_frequencies``
picks the small set of frequencies that explain most of the variance.

For the multiplicative-group task we also expose
``compute_fourier_basis_multiplicative`` which performs the analogous
decomposition with respect to the cyclic group ``(Z/pZ)^*``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FourierSpectrum:
    p: int
    freq_power: np.ndarray   # (p,) total power per frequency, summed across d_model
    per_dim_power: np.ndarray  # (p, d_model)
    fft: np.ndarray          # (p, d_model) complex
    group: str               # 'additive' or 'multiplicative'


def to_numpy(W: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(W, torch.Tensor):
        return W.detach().cpu().numpy()
    return np.asarray(W)


def compute_fourier_basis(W_E_digits: torch.Tensor | np.ndarray, p: int) -> FourierSpectrum:
    """DFT of the digit embeddings along the ``Z/pZ`` axis.

    Parameters
    ----------
    W_E_digits : (p, d_model) array-like
        The first ``p`` rows of the embedding matrix — the digit tokens.
    p : int
        Modulus.
    """
    W = to_numpy(W_E_digits)
    if W.shape[0] != p:
        raise ValueError(f"W_E_digits must have {p} rows, got {W.shape[0]}")
    fft = np.fft.fft(W, axis=0)               # (p, d_model)
    per_dim_power = np.abs(fft) ** 2
    freq_power = per_dim_power.sum(axis=1)
    return FourierSpectrum(
        p=p,
        freq_power=freq_power,
        per_dim_power=per_dim_power,
        fft=fft,
        group="additive",
    )


def compute_fourier_basis_multiplicative(
    W_E_digits: torch.Tensor | np.ndarray, p: int
) -> FourierSpectrum:
    """DFT of the embeddings reordered along the multiplicative group.

    The multiplicative group ``(Z/pZ)^*`` is cyclic of order ``p-1``. We pick
    a generator ``g``, list the group elements as ``g^0, g^1, ..., g^{p-2}``,
    and FFT the embeddings reordered into that sequence. Frequencies are
    therefore indexed by ``k = 0, 1, ..., p-2``.
    """
    W = to_numpy(W_E_digits)
    if W.shape[0] != p:
        raise ValueError(f"W_E_digits must have {p} rows, got {W.shape[0]}")
    g = _primitive_root(p)
    order = np.array([pow(g, i, p) for i in range(p - 1)])  # the (p-1) non-zero residues
    W_mult = W[order]                                       # (p-1, d_model)
    fft = np.fft.fft(W_mult, axis=0)
    per_dim_power = np.abs(fft) ** 2
    freq_power = per_dim_power.sum(axis=1)
    return FourierSpectrum(
        p=p,
        freq_power=freq_power,
        per_dim_power=per_dim_power,
        fft=fft,
        group="multiplicative",
    )


def identify_dominant_frequencies(
    spectrum: FourierSpectrum,
    fraction: float = 0.90,
    max_k: int = 20,
    skip_dc: bool = True,
) -> list[int]:
    """Frequencies ranked by power, taken until they explain ``fraction`` of total.

    The DC component (k=0) is dropped by default — it's the mean of each
    column, which carries no positional information on the cyclic group.

    Real DFT power is symmetric around ``p/2``; we report each pair
    (``k``, ``-k``) only once by their lower index.
    """
    p = spectrum.p
    power = spectrum.freq_power.copy()
    if skip_dc:
        power[0] = 0.0
    order = np.argsort(power)[::-1]
    # Deduplicate symmetric pairs: keep min(k, p - k).
    seen: set[int] = set()
    canonical_order: list[int] = []
    for k in order:
        kc = int(min(k, p - k)) if k != 0 else 0
        if kc in seen:
            continue
        seen.add(kc)
        canonical_order.append(kc)
    canonical_power = np.array([power[k] for k in canonical_order])
    total = canonical_power.sum()
    if total <= 0:
        return []
    cumulative = np.cumsum(canonical_power) / total
    cutoff = int(np.searchsorted(cumulative, fraction)) + 1
    cutoff = max(1, min(cutoff, max_k))
    return canonical_order[:cutoff]


def feature_overlap(
    spectrum_a: FourierSpectrum, spectrum_b: FourierSpectrum
) -> float:
    """Cosine similarity of two power spectra over their shared frequency axis.

    Used to compare which Fourier features two embedding matrices have
    learned. The DC component is excluded.
    """
    if spectrum_a.p != spectrum_b.p:
        raise ValueError("spectra must have the same modulus")
    if len(spectrum_a.freq_power) != len(spectrum_b.freq_power):
        raise ValueError("spectra must have the same length")
    a = spectrum_a.freq_power.copy()
    b = spectrum_b.freq_power.copy()
    a[0] = 0.0
    b[0] = 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(a @ b / denom)


def _primitive_root(p: int) -> int:
    """Smallest primitive root mod a prime ``p``."""
    if p == 2:
        return 1
    phi = p - 1
    factors = _factorise(phi)
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"no primitive root found for p={p}")


def _factorise(n: int) -> list[int]:
    factors: list[int] = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
