"""Direction correctness against known differentiation edges.

These metrics need curated ``cluster_edges`` -- a list of ``[source, target]``
cell-type pairs that are known to be a differentiation step.  A dataset without
them yields ``not_applicable``, which is a property of the dataset, not a
failure of the method.

Following the original VeloAE formulation (Qiao & Huang, PNAS 2021), CBDir and
CBVCoh are computed in a shared low-dimensional embedding rather than gene
space, so methods whose velocity lives in different native spaces stay
comparable.  The basis must be identical across every method in a comparison.
"""

from __future__ import annotations

import numpy as np

from .._math import nanmean, rowwise_cosine
from ..access import (
    get_embedding,
    get_labels,
    get_neighbor_indices,
    get_velocity_embedding,
)
from ..result import NotApplicable, metric

__all__ = ["cbdir", "cbvcoh", "cto"]


def _boundary_pairs(labels, cluster_edges, indices):
    """Yield ``(cell_index, array_of_boundary_neighbours)`` for every edge."""
    if not cluster_edges:
        raise NotApplicable("dataset has no curated cluster_edges")

    seen_any = False
    for src, tgt in cluster_edges:
        src_cells = np.where(labels == src)[0]
        tgt_cells = np.where(labels == tgt)[0]
        if src_cells.size == 0 or tgt_cells.size == 0:
            continue
        tgt_set = set(tgt_cells.tolist())
        for i in src_cells:
            boundary = np.array(
                [n for n in indices[i] if n in tgt_set and n != i], dtype=int
            )
            if boundary.size:
                seen_any = True
                yield i, boundary

    if not seen_any:
        raise NotApplicable(
            "no cell had a kNN neighbour across any cluster edge; "
            "check that label values match cluster_edges"
        )


@metric
def cbdir(
    adata,
    *,
    label_key: str,
    cluster_edges,
    basis: str = "umap",
    vkey: str = "velocity",
):
    """Cross-boundary direction correctness.  Higher is better; range [-1, 1].

    For each edge ``A -> B`` and each cell ``i`` in ``A``, take the cosine
    between ``i``'s velocity and the displacement towards each kNN neighbour
    that lies in ``B``.  Averaged over neighbours, then over all such cells.
    """
    labels = get_labels(adata, label_key)
    indices = get_neighbor_indices(adata)
    X = get_embedding(adata, basis)
    V = get_velocity_embedding(adata, basis, vkey)

    per_cell = np.full(adata.n_obs, np.nan)
    for i, boundary in _boundary_pairs(labels, cluster_edges, indices):
        disp = X[boundary] - X[i]
        cos = rowwise_cosine(disp, np.broadcast_to(V[i], disp.shape))
        per_cell[i] = nanmean(cos)

    return nanmean(per_cell), per_cell


@metric
def cbvcoh(
    adata,
    *,
    label_key: str,
    cluster_edges,
    basis: str = "umap",
    vkey: str = "velocity",
):
    """Cross-boundary velocity coherence.  Higher is better; range [-1, 1].

    Same boundary cells as :func:`cbdir`, but compares ``i``'s velocity with
    the *velocity* of each boundary neighbour rather than with the displacement
    towards it.  Measures continuity of the field across the boundary.
    """
    labels = get_labels(adata, label_key)
    indices = get_neighbor_indices(adata)
    V = get_velocity_embedding(adata, basis, vkey)

    per_cell = np.full(adata.n_obs, np.nan)
    for i, boundary in _boundary_pairs(labels, cluster_edges, indices):
        cos = rowwise_cosine(V[boundary], np.broadcast_to(V[i], V[boundary].shape))
        per_cell[i] = nanmean(cos)

    return nanmean(per_cell), per_cell


@metric
def cto(adata, *, label_key: str, cluster_edges, time_key: str = "latent_time"):
    """Cluster temporal ordering accuracy.  Higher is better; range [0, 1].

    Fraction of known edges ``A -> B`` for which the mean inferred time in
    ``A`` is smaller than in ``B``.  Unlike :func:`cbdir` this reads the
    method's own time estimate, so methods that infer no time are
    ``not_applicable``.
    """
    if not cluster_edges:
        raise NotApplicable("dataset has no curated cluster_edges")
    labels = get_labels(adata, label_key)
    if time_key not in adata.obs:
        raise NotApplicable(f"method infers no obs['{time_key}']")

    t = np.asarray(adata.obs[time_key].values, dtype=np.float64)
    correct = []
    for src, tgt in cluster_edges:
        a = t[labels == src]
        b = t[labels == tgt]
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0:
            continue
        correct.append(float(a.mean() < b.mean()))

    if not correct:
        raise NotApplicable("no edge had cells on both sides")
    return float(np.mean(correct))
