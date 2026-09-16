"""Local smoothness of the velocity field.

Coherence says nothing about whether the field points the *right* way -- a
confidently wrong field scores high.  These are companions to the direction
metrics, never substitutes: read them together.
"""

from __future__ import annotations

import numpy as np

from .._math import nanmean, rowwise_cosine
from ..access import get_labels, get_neighbor_indices, get_velocity
from ..result import metric

__all__ = ["icvcoh", "velocity_consistency"]


@metric
def icvcoh(adata, *, label_key: str, vkey: str = "velocity"):
    """In-cluster coherence.  Higher is better; range [-1, 1].

    Mean cosine between a cell's velocity and that of its *same-cluster* kNN
    neighbours; averaged within each cluster, then equally across clusters so
    that large clusters do not dominate.
    """
    labels = get_labels(adata, label_key)
    indices = get_neighbor_indices(adata)
    V = get_velocity(adata, vkey)

    per_cell = np.full(adata.n_obs, np.nan)
    for i in range(adata.n_obs):
        same = np.array(
            [n for n in indices[i] if n != i and labels[n] == labels[i]], dtype=int
        )
        if same.size == 0:
            continue
        per_cell[i] = nanmean(
            rowwise_cosine(V[same], np.broadcast_to(V[i], V[same].shape))
        )

    cluster_means = [
        nanmean(per_cell[labels == c])
        for c in np.unique(labels)
        if not np.all(np.isnan(per_cell[labels == c]))
    ]
    if not cluster_means:
        return float("nan"), per_cell
    return nanmean(cluster_means), per_cell


@metric
def velocity_consistency(adata, *, vkey: str = "velocity"):
    """Velocity consistency.  Higher is better; range [-1, 1].

    scVelo's ``velocity_confidence``: the cosine between a cell's velocity and
    the mean velocity of its kNN neighbourhood.  Needs no cell-type labels, so
    it is available on datasets such as RPE1/U2OS where :func:`icvcoh` is not.
    """
    indices = get_neighbor_indices(adata)
    V = get_velocity(adata, vkey)

    per_cell = np.full(adata.n_obs, np.nan)
    for i in range(adata.n_obs):
        nb = np.array([n for n in indices[i] if n != i], dtype=int)
        if nb.size == 0:
            continue
        per_cell[i] = rowwise_cosine(V[i][None, :], V[nb].mean(axis=0)[None, :])[0]

    return nanmean(per_cell), per_cell
