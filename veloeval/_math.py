"""Small numeric helpers shared by the metrics."""

from __future__ import annotations

import numpy as np

__all__ = [
    "rowwise_cosine",
    "cosine_to_one",
    "wrap_angle",
    "nanmean",
    "spearman",
    "uniform_scores",
    "fisher_lee",
]


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


def uniform_scores(x) -> np.ndarray:
    """Ranks spread evenly round the circle, ``2 pi (r - 1) / n``; ties share a rank."""
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=np.float64)
    return 2 * np.pi * (rankdata(x) - 1) / x.size


def fisher_lee(a, b) -> float:
    r"""Fisher & Lee's (1983) circular-circular correlation of angles *a*, *b*.

    .. math::

        \rho = \frac{\sum_{i<j} \sin(a_i-a_j)\sin(b_i-b_j)}
                    {\sqrt{\sum_{i<j}\sin^2(a_i-a_j)\sum_{i<j}\sin^2(b_i-b_j)}}

    in O(n) via the sums of :math:`\cos`, :math:`\sin` and their double angles.
    NaN when either side is degenerate (all equal, or two antipodal values).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    A, B, C, D = ca @ cb, sa @ sb, ca @ sb, sa @ cb
    va = n**2 - np.cos(2 * a).sum() ** 2 - np.sin(2 * a).sum() ** 2
    vb = n**2 - np.cos(2 * b).sum() ** 2 - np.sin(2 * b).sum() ** 2
    if va <= 1e-9 * n**2 or vb <= 1e-9 * n**2:
        return float("nan")
    return float(4 * (A * B - C * D) / np.sqrt(va * vb))
