"""Small numeric helpers shared by the metrics."""

from __future__ import annotations

import numpy as np

__all__ = ["rowwise_cosine", "cosine_to_one", "wrap_angle", "nanmean", "spearman"]


def rowwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine between matching rows of *A* and *B*.  Zero rows give NaN."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    num = np.einsum("ij,ij->i", A, B)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    out = np.full(A.shape[0], np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def cosine_to_one(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine between every row of *M* and the single vector *v*."""
    return rowwise_cosine(M, np.broadcast_to(v, M.shape))


def wrap_angle(d: np.ndarray) -> np.ndarray:
    """Wrap angular differences into ``(-pi, pi]``."""
    return (np.asarray(d) + np.pi) % (2 * np.pi) - np.pi


def nanmean(x) -> float:
    """Mean ignoring NaN; NaN when nothing is left."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def spearman(a, b) -> float:
    """Spearman rank correlation, ignoring pairs where either side is NaN."""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    good = ~(np.isnan(a) | np.isnan(b))
    if good.sum() < 3:
        return float("nan")
    rho = spearmanr(a[good], b[good]).statistic
    return float(rho)
