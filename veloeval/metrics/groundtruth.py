"""Metrics scored against a truth that is independent of RNA.

Everything in :mod:`~veloeval.metrics.direction` and
:mod:`~veloeval.metrics.coherence` is scored against structure that was itself
read off the transcriptome, so a systematic error shared by the field is
invisible to them.  The metrics here use a signal the model never saw:

``phase_dir``   FUCCI fluorescence -- a protein-level, live-imageable readout
``truth_cos``   metabolic labelling -- a physical clock with real units
``gamma_corr``  labelling-derived degradation rates, per gene

They are the point of the benchmark, and they are also the ones with the
narrowest applicability: read the ``not_applicable`` details, not just the
numbers.
"""

from __future__ import annotations

import numpy as np

from .._math import nanmean, rowwise_cosine, spearman, wrap_angle
from ..access import (
    get_embedding,
    get_neighbor_indices,
    get_velocity,
    get_velocity_embedding,
    velocity_space,
)
from ..result import NotApplicable, metric

__all__ = ["phase_dir", "truth_cos", "gamma_corr"]


@metric
def phase_dir(
    adata,
    *,
    phase_key: str = "fucci_phase",
    basis: str = "umap",
    vkey: str = "velocity",
    period: float = 2 * np.pi,
    min_neighbors: int = 4,
):
    r"""FUCCI phase-gradient projection consistency.

    ``obs[phase_key]`` holds each cell's position on the cell cycle, derived
    from the two FUCCI protein reporters -- measured, cyclic, and independent
    of the RNA the model fits.  For each cell the local gradient of phase is
    fitted over its kNN neighbourhood by least squares in the embedding,

    .. math::

        g_i = \arg\min_g \sum_j \bigl( d_j \cdot g - \mathrm{wrap}(\varphi_j -
        \varphi_i) \bigr)^2, \qquad d_j = x_j - x_i

    and the metric is the cosine between the cell's velocity and :math:`g_i`.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry ``obs[phase_key]``, ``obsm['X_{basis}']``,
        ``obsm['{vkey}_{basis}']`` and ``uns['neighbors']['indices']``.
    phase_key : str, default: "fucci_phase"
        Column holding the FUCCI-derived cell-cycle position.  Absent ->
        ``not_applicable``.
    basis : str, default: "umap"
        Embedding the gradient and the velocity are read in.
    vkey : str, default: "velocity"
        Velocity key prefix.
    period : float, default: ``2 * np.pi``
        Period of the stored phase.  Pass ``1.0`` if it is on ``[0, 1)``.
    min_neighbors : int, default: 4
        Cells with fewer usable neighbours are left ``nan``.

    Returns
    -------
    MetricResult
        ``per_cell`` holds the per-cell cosine, ``nan`` where no gradient could
        be fitted.  ``not_applicable`` when the dataset has no phase column or
        no cell had a usable neighbourhood.

    Notes
    -----
    The wrap keeps the metric correct across the phase origin, so cells at the
    G1/M boundary are not scored backwards -- a naive difference would flip
    their sign.  Because the truth is a protein-level readout, this metric does
    not share the failure modes of the transcriptome-derived metrics, which is
    the whole point of including it.
    """
    if phase_key not in adata.obs:
        raise NotApplicable(f"dataset has no cell-cycle phase obs['{phase_key}']")

    phi = np.asarray(adata.obs[phase_key].values, dtype=np.float64)
    phi = phi * (2 * np.pi / period)  # work in radians internally

    X = get_embedding(adata, basis)
    V = get_velocity_embedding(adata, basis, vkey)
    indices = get_neighbor_indices(adata)

    per_cell = np.full(adata.n_obs, np.nan)
    for i in range(adata.n_obs):
        if np.isnan(phi[i]):
            continue
        nb = np.array([n for n in indices[i] if n != i and not np.isnan(phi[n])], dtype=int)
        if nb.size < min_neighbors:
            continue

        D = X[nb] - X[i]
        dphi = wrap_angle(phi[nb] - phi[i])
        if not np.any(np.linalg.norm(D, axis=1) > 0):
            continue

        g, *_ = np.linalg.lstsq(D, dphi, rcond=None)
        if np.linalg.norm(g) == 0:
            continue
        per_cell[i] = rowwise_cosine(V[i][None, :], g[None, :])[0]

    if np.all(np.isnan(per_cell)):
        raise NotApplicable("no cell had a usable phase neighbourhood")
    return nanmean(per_cell), per_cell


@metric
def truth_cos(adata, reference, *, vkey: str = "velocity", ref_vkey: str = "velocity"):
    """Metabolic-labelling ground-truth cosine.

    On tscRNA-seq data, hide the labelling channel and let a splicing-based
    method see only spliced/unspliced; then score its velocity against the
    velocity derived from the labelling, cell by cell, on shared genes.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        The method's run.  Its velocity space must be ``"gene"``.
    reference : anndata.AnnData
        Run carrying the labelling-derived velocity, on the **same cells in
        the same order**.
    vkey : str, default: "velocity"
        Velocity layer key in *adata*.
    ref_vkey : str, default: "velocity"
        Velocity layer key in *reference*.

    Returns
    -------
    MetricResult
        ``per_cell`` holds the per-cell cosine.  ``not_applicable`` when the
        velocity is not gene-space, the cell counts differ, or fewer than 10
        genes are shared and valid in both runs.

    Notes
    -----
    A method whose velocity lives in a learned latent space has no gene-wise
    correspondence to the reference, so the cosine would be meaningless;
    those return ``not_applicable`` rather than a number.  Scoring them needs a
    space-free measure -- distance correlation between the two neighbourhood
    geometries -- which is deliberately not implemented yet.
    """
    if velocity_space(adata) != "gene":
        raise NotApplicable(
            f"velocity is in {velocity_space(adata)!r} space; gene-wise cosine "
            "against the labelling reference is not defined (needs a "
            "space-free measure)"
        )
    if adata.n_obs != reference.n_obs:
        raise NotApplicable(
            f"cell counts differ ({adata.n_obs} vs {reference.n_obs}); "
            "the two runs must be on the same cells in the same order"
        )

    shared = adata.var_names.intersection(reference.var_names)
    if len(shared) < 10:
        raise NotApplicable(f"only {len(shared)} genes shared with the reference")

    ia = adata.var_names.get_indexer(shared)
    ib = reference.var_names.get_indexer(shared)
    V = get_velocity(adata, vkey, drop_nan_genes=False)[:, ia]
    R = get_velocity(reference, ref_vkey, drop_nan_genes=False)[:, ib]

    keep = ~(np.isnan(V).any(axis=0) | np.isnan(R).any(axis=0))
    if keep.sum() < 10:
        raise NotApplicable("fewer than 10 genes valid in both runs")

    per_cell = rowwise_cosine(V[:, keep], R[:, keep])
    return nanmean(per_cell), per_cell


@metric
def gamma_corr(
    adata,
    reference,
    *,
    gamma_key: str = "fit_gamma",
    ref_gamma_key: str = "fit_gamma",
):
    """Gene-level degradation-rate correlation.

    Spearman correlation between the degradation rates a splicing method infers
    and those measured from metabolic labelling.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        The method's run; must expose ``var[gamma_key]``.
    reference : anndata.AnnData
        Run carrying labelling-derived rates in ``var[ref_gamma_key]``.
    gamma_key : str, default: "fit_gamma"
        Per-gene degradation rate in *adata*.
    ref_gamma_key : str, default: "fit_gamma"
        Per-gene degradation rate in *reference*.

    Returns
    -------
    MetricResult
        ``not_applicable`` when either side exposes no per-gene rate or fewer
        than 10 genes are shared.

    Notes
    -----
    Only methods with an explicit rate parameter qualify.  Rate-free models
    (autoencoders regressing velocity directly) and models with no splicing
    rate concept are ``not_applicable`` by construction -- see the
    ``速率假设`` / rate-assumption column of the methods table.
    """
    if gamma_key not in adata.var:
        raise NotApplicable(f"method exposes no per-gene rate var['{gamma_key}']")
    if ref_gamma_key not in reference.var:
        raise NotApplicable(f"reference has no measured var['{ref_gamma_key}']")

    shared = adata.var_names.intersection(reference.var_names)
    if len(shared) < 10:
        raise NotApplicable(f"only {len(shared)} genes shared with the reference")

    ia = adata.var_names.get_indexer(shared)
    ib = reference.var_names.get_indexer(shared)
    a = adata.var[gamma_key].to_numpy(dtype=np.float64)[ia]
    b = reference.var[ref_gamma_key].to_numpy(dtype=np.float64)[ib]
    return spearman(a, b)
