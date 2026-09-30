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
    SPATIAL_KNN_KEY,
    get_labels,
    get_neighbor_indices,
    get_velocity,
    get_velocity_embedding,
)
from ..result import MetricResult, MissingInput, NotApplicable, metric

__all__ = [
    "icvcoh",
    "velocity_consistency",
    "spatial_consistency",
    "time_morans_i",
    "field_constancy",
]


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

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        res = M.icvcoh(adata, label_key="clusters")
        res.value

        # the method's own velocity instead of the UMAP projection
        M.icvcoh(adata, label_key="clusters", basis=None)
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

    Requires scvelo (``pip install veloeval[prepare]``); without it the
    status is ``missing_input``.

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        res = M.velocity_consistency(adata)
        res.value
        res.per_cell  # equals scVelo's obs["velocity_confidence"]
    """
    return _scvelo_confidence(adata, get_neighbor_indices(adata), vkey)


def _scvelo_confidence(adata, indices, vkey):
    """scVelo's ``velocity_confidence`` of ``layers[vkey]`` over *indices*."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    try:
        import scvelo as scv
    except ImportError:
        raise MissingInput("scvelo (pip install veloeval[prepare])") from None

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


@metric
def spatial_consistency(adata, *, vkey: str = "velocity"):
    """Spatial velocity consistency.

    :func:`velocity_consistency` with the neighbours taken in physical space,
    ``obsm['veloeval_spatial_knn']``, instead of expression space: scVelo's
    ``velocity_confidence`` -- per-neighbour cosine of gene-centred velocities,
    averaged, negatives clipped to 0 -- as in TopoVelo (Gu et al., *Nat
    Biotechnol* 2025).

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry gene-space ``layers[vkey]`` and
        ``obsm['veloeval_spatial_knn']`` (see
        :func:`veloeval.build_spatial_neighbors`).
    vkey : str, default: "velocity"
        Velocity layer key.

    Returns
    -------
    MetricResult
        ``per_cell`` holds each spot's score.

    Notes
    -----
    Cannot detect a reversed field: :math:`V` and :math:`-V` score the same.
    It is a coherence measure, not a correctness one.

    Methods that smooth over space in their model (TopoVelo, spVelo) are
    favoured by construction, since the metric comes from the same line of
    work; score correctness on spatial data with a known lineage with
    :func:`~veloeval.metrics.cbdir` and :func:`~veloeval.metrics.cto`.

    Values are not guaranteed to match Huang et al. (bioRxiv 2026), which does
    not state its k or whether negatives are clipped.

    Requires scvelo (``pip install veloeval[prepare]``); without it the
    status is ``missing_input``.

    Examples
    --------
    .. code-block:: python

        import veloeval as ve
        from veloeval import metrics as M

        ve.build_spatial_neighbors(adata, spatial_key="spatial")  # Visium: 6
        res = M.spatial_consistency(adata)
        res.value
        res.per_cell
    """
    return _scvelo_confidence(
        adata, get_neighbor_indices(adata, key=SPATIAL_KNN_KEY), vkey
    )


def _knn_adjacency(indices, n):
    """Binary, symmetric adjacency from a kNN table, no self-loops."""
    from scipy.sparse import csr_matrix

    indices = np.asarray(indices)
    rows = np.repeat(np.arange(n), indices.shape[1])
    W = csr_matrix((np.ones(rows.size), (rows, indices.ravel())), shape=(n, n))
    W = ((W + W.T) > 0).astype(np.float64).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    return W


@metric
def time_morans_i(adata, *, time_key: str = "latent_time"):
    r"""Moran's I of the inferred time over the spatial neighbour graph.

    With :math:`W` the binary, symmetrised spatial kNN adjacency (no
    self-loops), :math:`z = t - \bar t` and :math:`S_0 = \sum_{ij} W_{ij}`,

    .. math::

        I = \frac{n}{S_0} \cdot \frac{z^\top W z}{z^\top z}

    (Moran, *Biometrika* 1950), as applied to inferred time in TopoVelo (Gu et
    al., *Nat Biotechnol* 2025) and called spatial time consistency by Huang
    et al. (bioRxiv 2026).

    Higher is better; roughly ``[-1, 1]``, with :math:`-1/(n-1)` expected
    without spatial structure.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry ``obs[time_key]`` and ``obsm['veloeval_spatial_knn']`` (see
        :func:`veloeval.build_spatial_neighbors`).
    time_key : str, default: "latent_time"
        Column holding the method's inferred time.  Cells where it is NaN are
        dropped, together with their edges.

    Returns
    -------
    MetricResult
        :math:`I`; ``detail`` gives its expectation without spatial structure.

    Notes
    -----
    Cannot detect a reversed time: :math:`I(t) = I(1 - t)`.  It is a
    coherence measure, not a correctness one, and the same caveat as for
    :func:`spatial_consistency` applies to methods that smooth over space.

    Examples
    --------
    .. code-block:: python

        import veloeval as ve
        from veloeval import metrics as M

        ve.build_spatial_neighbors(adata, spatial_key="spatial")
        res = M.time_morans_i(adata, time_key="latent_time")
        res.value
        res.detail  # expectation without spatial structure
    """
    if time_key not in adata.obs:
        raise NotApplicable(f"method infers no obs['{time_key}']")
    W = _knn_adjacency(get_neighbor_indices(adata, key=SPATIAL_KNN_KEY), adata.n_obs)

    t = adata.obs[time_key].to_numpy(dtype=np.float64)
    keep = np.isfinite(t)
    W, t = W[keep][:, keep], t[keep]
    n = t.size
    z = t - t.mean() if n else t
    s0 = W.sum()
    if n < 2 or s0 == 0 or z @ z == 0:
        raise NotApplicable("inferred time is constant or has no spatial neighbours")

    moran = n / s0 * (z @ (W @ z)) / (z @ z)
    return MetricResult(
        name="time_morans_i",
        value=float(moran),
        detail=f"expected {-1 / (n - 1):.3f} without spatial structure",
    )


@metric
def field_constancy(adata, *, vkey: str = "velocity"):
    r"""How close the velocity field is to one constant direction.

    The mean resultant length of the unit velocities, over cells whose
    velocity is nonzero:

    .. math::

        \mathrm{FC} = \Bigl\lVert \frac1n \sum_i \mathbf u_i \Bigr\rVert
        = \frac1n \sum_i \cos(\mathbf u_i, \bar{\mathbf u}),
        \qquad \mathbf u_i = \mathbf v_i / \lVert \mathbf v_i \rVert

    Range ``[0, 1]``: 1 for a constant field, about :math:`1/\sqrt n` for an
    isotropic random one.  A diagnostic, not ranked: ``DIRECTION`` maps it to
    ``None``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry a gene-space velocity in ``layers[vkey]``; genes with any
        NaN are dropped.
    vkey : str, default: "velocity"
        Velocity key prefix.

    Returns
    -------
    MetricResult
        ``per_cell`` is each cell's cosine with the mean direction, ``nan``
        for cells with zero velocity.  ``not_applicable`` with fewer than two
        nonzero velocities or outside gene space.

    Notes
    -----
    High is neither good nor bad: a linear trajectory is legitimately
    constant, a correct field on a branching one is less so.  Read it across
    methods on one dataset.  A method whose constancy stands well above the
    others' while its :func:`icvcoh` and :func:`velocity_consistency` are also
    high has probably collapsed towards a constant vector -- a decoder bias,
    say -- and those two coherence scores are then no evidence of quality.

    Unit vectors first, so a few cells with large velocities do not set the
    mean direction.  Unchanged by rescaling the field or reversing it, so it
    says nothing about direction.  Not invariant to rescaling single genes: a
    method that outputs velocity on standardised data scores differently.

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        res = M.field_constancy(adata)
        res.value     # 1 = every cell points the same way
        res.per_cell  # each cell's cosine with the mean direction
    """
    V = get_velocity(adata, vkey)
    norm = np.linalg.norm(V, axis=1)
    moving = norm > 0
    if moving.sum() < 2:
        raise NotApplicable("fewer than two cells have a nonzero velocity")
    U = V[moving] / norm[moving, None]
    mean = U.mean(axis=0)
    R = float(np.linalg.norm(mean))
    per_cell = np.full(adata.n_obs, np.nan)
    if R > 0:
        per_cell[moving] = U @ (mean / R)
    return R, per_cell
