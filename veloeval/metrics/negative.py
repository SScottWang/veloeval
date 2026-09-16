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


@metric
def mag_ratio(adata_negative, adata_positive, *, vkey: str = "velocity"):
    """Relative velocity magnitude ratio.

    Mean ``||v||`` on a negative control divided by mean ``||v||`` on a
    positive control, for the *same method*.

    Closer to 0 is better; unbounded above.

    Parameters
    ----------
    adata_negative : anndata.AnnData
        Run on the negative control (e.g. the pancreas terminal-state subset).
    adata_positive : anndata.AnnData
        Run on the paired positive control (e.g. the full pancreas dataset).
    vkey : str, default: "velocity"
        Velocity layer key, used in both runs.

    Returns
    -------
    MetricResult
        The ratio, or ``not_applicable`` when the positive control has zero
        mean magnitude.

    Warnings
    --------
    Both runs must come from the **same method** with the **same
    preprocessing**, differing only in the dataset.  Otherwise this compares
    two datasets rather than measuring the method.

    Notes
    -----
    A method that genuinely detects steady state shrinks its arrows when there
    is nothing to detect; one that always emits unit-ish vectors gives a ratio
    near 1 whatever it is shown.  This is a two-run metric, so
    :func:`~veloeval.compute_all` does not include it -- call it from the
    pipeline, which knows how the control pairs are matched.
    """
    v_neg = get_velocity(adata_negative, vkey)
    v_pos = get_velocity(adata_positive, vkey)

    denom = float(np.nanmean(np.linalg.norm(v_pos, axis=1)))
    if not np.isfinite(denom) or denom == 0:
        raise NotApplicable("positive control has zero mean velocity magnitude")

    numer = float(np.nanmean(np.linalg.norm(v_neg, axis=1)))
    return numer / denom
