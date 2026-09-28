"""Local smoothness of the velocity field.

Coherence says nothing about whether the field points the *right* way -- a
confidently wrong field scores high.  These are companions to the direction
metrics, never substitutes: read them together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._math import nanmean, rowwise_cosine
from ..access import (
    get_labels,
    get_neighbor_indices,
    get_velocity,
    get_velocity_embedding,
)
from ..result import MissingInput, metric

__all__ = ["icvcoh", "velocity_consistency"]


@metric
def icvcoh(
    adata, *, label_key: str, basis: str | None = "umap", vkey: str = "velocity"
):
    """In-cluster coherence.

    Mean cosine between a cell's velocity and that of its *same-cluster* kNN
    neighbours; averaged within each cluster, then equally across clusters so
    that large clusters do not dominate.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry ``obsm['{vkey}_{basis}']`` (or gene-space
        ``layers[vkey]`` when *basis* is ``None``) and ``obsm['veloeval_knn']``.
    label_key : str
        Column in ``adata.obs`` holding cell-type labels.  ``None`` yields
        ``not_applicable`` -- single cell-line datasets such as RPE1 and U2OS
        have no labels, and :func:`velocity_consistency` is the metric to use
        there.
    basis : str or None, default: "umap"
        Embedding the velocity is read in, the same one as
        :func:`~veloeval.metrics.cbdir`.  ``None`` reads the gene-space
        ``layers[vkey]`` instead, as VeloAE did; methods whose velocity lives
        in a latent space are then ``not_applicable``.
    vkey : str, default: "velocity"
        Velocity key prefix.

    Returns
    -------
    MetricResult
        ``per_cell`` holds each cell's mean cosine against its same-cluster
        neighbours, before the two-stage averaging.

    Notes
    -----
    Says nothing about whether the field points the right way: a confidently
    wrong field scores high.  Read next to :func:`~veloeval.metrics.cbdir`.

    The embedding default follows the Genome Biology benchmark (2026) and lets
    latent-space methods be scored alongside the rest.  Method rankings can
    differ between spaces, so compare methods only within one *basis*.
    """
    labels = get_labels(adata, label_key)
    indices = get_neighbor_indices(adata)
    if basis is None:
        V = get_velocity(adata, vkey)
    else:
        V = get_velocity_embedding(adata, basis, vkey)

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
    """Velocity consistency.

    scVelo's ``velocity_confidence``, computed by calling
    :func:`scvelo.tl.velocity_confidence` itself: each cell's velocity and its
    neighbours' are centred across genes, the cosine is taken against each
    neighbour separately and averaged, and negative values are clipped to 0.

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry gene-space ``layers[vkey]`` and ``obsm['veloeval_knn']``.
        ``var['{vkey}_genes']`` and ``var['spearmans_score']`` are honoured
        as scVelo honours them.
    vkey : str, default: "velocity"
        Velocity layer key.

    Returns
    -------
    MetricResult
        ``per_cell`` holds scVelo's per-cell ``velocity_confidence``.

    Notes
    -----
    Needs no cell-type labels (input requirement R0), so it is available on
    every dataset -- including the single cell-line FUCCI data where
    :func:`icvcoh` is ``not_applicable``.

    The neighbours are the shared ``obsm['veloeval_knn']`` rather than
    whatever graph the method left in ``obsp['distances']``, so this metric
    sees the same neighbourhoods as the others.  With the default
    :func:`veloeval.build_neighbor_indices` the two are the same graph.

    Requires scvelo (``pip install veloeval[prepare]``).
    """
    import anndata as ad
    import scvelo as scv
    from scipy.sparse import csr_matrix

    indices = get_neighbor_indices(adata)
    get_velocity(adata, vkey)
    if vkey not in adata.layers:
        raise MissingInput(f"layers['{vkey}']")

    n = adata.n_obs
    rows = [r[r != i] for i, r in enumerate(np.asarray(indices))]
    indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    distances = csr_matrix(
        (np.ones(indptr[-1]), np.concatenate(rows), indptr), shape=(n, n)
    )

    keep_var = [c for c in (f"{vkey}_genes", "spearmans_score") if c in adata.var]
    sub = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names), var=adata.var[keep_var])
    layer = adata.layers[vkey]
    sub.layers[vkey] = layer.toarray() if hasattr(layer, "toarray") else np.asarray(layer)
    sub.obsp["distances"] = distances
    # scVelo then goes on to velocity_confidence_transition, which needs Ms and
    # a transition matrix; a pre-filled column is how it is told to skip that.
    sub.obs[f"{vkey}_confidence_transition"] = 0.0

    verbosity = scv.settings.verbosity
    scv.settings.verbosity = 0
    try:
        scv.tl.velocity_confidence(sub, vkey=vkey)
    finally:
        scv.settings.verbosity = verbosity

    per_cell = sub.obs[f"{vkey}_confidence"].to_numpy(dtype=np.float64)
    return nanmean(per_cell), per_cell
