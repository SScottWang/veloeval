"""Negative controls: does the method stay quiet where nothing is happening?

On a population with no ongoing differentiation, a trustworthy method should
produce a diffuse, low-magnitude field.  A confident, coherent field there is a
hallucination -- and the conventional accuracy metrics cannot see it, because
they only ever ask "is the arrow pointing the right way" on data where there is
a right way.

:func:`sts` and :func:`ees` read a transition matrix and are single-run.
:func:`mag_ratio` compares two runs of the *same method* on a negative- and a
positive-control dataset, so it takes two AnnData objects.
"""

from __future__ import annotations

import numpy as np

from ..access import get_transition_matrix, get_velocity
from ..result import NotApplicable, metric

__all__ = ["sts", "ees", "mag_ratio"]


def _rows(T):
    """Iterate over ``(row_values,)`` of a dense or sparse transition matrix."""
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
    """Self-transition score.  Higher is better; range [0, 1].

    Mean probability that a cell transitions to itself.  A cell that is not
    moving should mostly stay put, so on a negative control a high STS is the
    correct answer.
    """
    T = get_transition_matrix(adata, tkey)
    diag = T.diagonal() if hasattr(T, "diagonal") else np.diag(np.asarray(T))
    diag = np.asarray(diag, dtype=np.float64)
    return float(np.nanmean(diag)), diag


@metric
def ees(adata, *, tkey: str = "T_fwd"):
    """Effective entropy score.  Higher is better; range [0, 1].

    Shannon entropy of each cell's transition distribution, normalised by the
    entropy of a uniform distribution over that cell's candidate transitions.
    1.0 means "no opinion about where this cell goes next", which on a negative
    control is the honest answer; a low score there means the method invented a
    trajectory.
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


@metric
def mag_ratio(adata_negative, adata_positive, *, vkey: str = "velocity"):
    """Relative velocity magnitude ratio.  Closer to 0 is better.

    Mean ``||v||`` on a negative control divided by mean ``||v||`` on a
    positive control, for the *same method*.  A method that genuinely detects
    steady state shrinks its arrows when there is nothing to detect; one that
    always emits unit-ish vectors gives a ratio near 1.

    Both runs must come from the same method with the same preprocessing --
    otherwise this compares the two datasets, not the method.
    """
    v_neg = get_velocity(adata_negative, vkey)
    v_pos = get_velocity(adata_positive, vkey)

    denom = float(np.nanmean(np.linalg.norm(v_pos, axis=1)))
    if not np.isfinite(denom) or denom == 0:
        raise NotApplicable("positive control has zero mean velocity magnitude")

    numer = float(np.nanmean(np.linalg.norm(v_neg, axis=1)))
    return numer / denom
