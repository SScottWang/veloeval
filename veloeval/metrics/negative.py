"""Negative controls: does the method stay quiet where nothing is happening?

On a population with no ongoing differentiation, a trustworthy method should
produce a diffuse, low-confidence field.  A confident one there is a
hallucination -- and the conventional accuracy metrics cannot see it, because
they only ever ask "is the arrow pointing the right way" on data where there is
a right way.

Both metrics here read a transition matrix from ``adata.obsp``.  It is not
computed for you; see :func:`veloeval.prepare`.
"""

from __future__ import annotations

import numpy as np

from ..access import get_transition_matrix
from ..result import NotApplicable, metric

__all__ = ["sts", "ees"]


def _rows(T):
    """Iterate over the non-zero values of each row of a dense or sparse matrix."""
    if hasattr(T, "tocsr"):
        T = T.tocsr()
        for i in range(T.shape[0]):
            yield T.data[T.indptr[i] : T.indptr[i + 1]]
    else:
        T = np.asarray(T)
        for i in range(T.shape[0]):
            row = T[i]
            yield row[row > 0]


@metric
def sts(adata, *, tkey: str = "T_fwd"):
    """Self-transition score.

    Mean probability that a cell transitions to itself.

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry a row-stochastic transition matrix in ``obsp[tkey]``.
    tkey : str, default: "T_fwd"
        Key of the transition matrix in ``adata.obsp``.

    Returns
    -------
    MetricResult
        ``per_cell`` holds each cell's self-transition probability.

    Notes
    -----
    Interpret on a **negative control** -- a population with no ongoing
    differentiation.  A cell that is not moving should mostly stay put, so a
    high score there is the correct answer, and a low one means the method
    invented movement.
    """
    T = get_transition_matrix(adata, tkey)
    diag = T.diagonal() if hasattr(T, "diagonal") else np.diag(np.asarray(T))
    diag = np.asarray(diag, dtype=np.float64)
    return float(np.nanmean(diag)), diag


@metric
def ees(adata, *, tkey: str = "T_fwd"):
    """Effective entropy score.

    Shannon entropy of each cell's transition distribution, normalised by the
    entropy of a uniform distribution over that cell's candidate transitions.

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry a row-stochastic transition matrix in ``obsp[tkey]``.
    tkey : str, default: "T_fwd"
        Key of the transition matrix in ``adata.obsp``.

    Returns
    -------
    MetricResult
        ``per_cell`` holds each cell's normalised entropy; rows with fewer than
        two candidate targets are ``nan``.

    Notes
    -----
    1.0 means "no opinion about where this cell goes next".  On a negative
    control that is the honest answer; a low score there means the method
    produced a confident but fabricated trajectory.
    """
    T = get_transition_matrix(adata, tkey)

    per_cell = []
    for p in _rows(T):
        if p.size < 2:
            per_cell.append(np.nan)
            continue
        p = p / p.sum()
        h = -np.sum(p * np.log(p))
        per_cell.append(h / np.log(p.size))

    per_cell = np.asarray(per_cell, dtype=np.float64)
    if np.all(np.isnan(per_cell)):
        raise NotApplicable("transition matrix has no multi-target rows")
    return float(np.nanmean(per_cell)), per_cell
