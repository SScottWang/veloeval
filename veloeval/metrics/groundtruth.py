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
import pandas as pd

from .._math import nanmean, rowwise_cosine, spearman, wrap_angle
from ..access import (
    get_embedding,
    get_neighbor_indices,
    get_velocity,
    get_velocity_embedding,
    velocity_space,
)
from ..result import MetricResult, NotApplicable, metric

__all__ = ["phase_dir", "truth_cos", "gamma_corr"]


@metric
def phase_dir(
    adata,
    *,
    phase_key: str = "fucci_phase",
    basis: str = "umap",
    vkey: str = "velocity",
    period: float = 1.0,
    n_dims: int | None = None,
    min_neighbors: int = 4,
    min_r2: float | None = None,
    n_bins: int = 10,
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
    How well the plane fits is

    .. math::

        R^2_i = 1 - \frac{\sum_j (d_j \cdot g_i - \mathrm{wrap}(\varphi_j -
        \varphi_i))^2}{\sum_j \mathrm{wrap}(\varphi_j - \varphi_i)^2}.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry ``obs[phase_key]``, ``obsm['X_{basis}']``,
        ``obsm['{vkey}_{basis}']`` and ``obsm['veloeval_knn']``.
    phase_key : str, default: "fucci_phase"
        Column holding the FUCCI-derived cell-cycle position.  Absent ->
        ``not_applicable``.
    basis : str, default: "umap"
        Embedding the gradient and the velocity are read in.
    vkey : str, default: "velocity"
        Velocity key prefix.
    period : float, default: 1.0
        Period of the stored phase: ``1.0`` for a relative position on
        ``[0, 1)``, as FUCCI datasets usually ship it (dynamo's scEU-seq RPE1
        as ``obs['Cell_cycle_relativePos']``); ``2 * np.pi`` for radians.
    n_dims : int, optional
        Use only the first *n_dims* columns of both embeddings, e.g. the
        leading PCs with ``basis="pca"``.  ``None`` uses them all.
    min_neighbors : int, default: 4
        Cells with fewer usable neighbours are left ``nan``.
    min_r2 : float, optional
        Leave cells whose gradient fit has :math:`R^2_i` below this ``nan``.
        ``None`` scores every cell.
    n_bins : int, default: 10
        Equal-width phase bins for ``per_group``.

    Returns
    -------
    MetricResult
        ``value`` is the mean over scored cells.  ``per_cell`` holds the
        per-cell cosine, ``nan`` where no gradient could be fitted or the fit
        fell below *min_r2*.  ``per_group`` maps each phase bin (as a fraction
        of *period*) to its mean.  ``detail`` gives the median :math:`R^2_i`
        and the fraction of cells scored.  ``not_applicable`` when the
        dataset has no phase column or no cell had a usable neighbourhood.

    Notes
    -----
    The wrap keeps the metric correct across the phase origin, so cells at the
    G1/M boundary are not scored backwards -- a naive difference would flip
    their sign.  Because the truth is a protein-level readout, this metric does
    not share the failure modes of the transcriptome-derived metrics, which is
    the whole point of including it.

    Read ``per_group`` against a shuffled-velocity control on the same data:
    a single bin of a shuffled field can sit well away from 0 even when its
    overall mean does not.  Filtering on *min_r2* drops the regions where
    phase is noisiest, which can be exactly where a method fails, so the
    default keeps every cell.  Cosines shrink as dimensions are added, so
    compare values only within one *basis* and *n_dims*.
    """
    if phase_key not in adata.obs:
        raise NotApplicable(f"dataset has no cell-cycle phase obs['{phase_key}']")

    phi = np.asarray(adata.obs[phase_key].values, dtype=np.float64)
    phi = phi * (2 * np.pi / period)  # work in radians internally

    X = get_embedding(adata, basis)[:, :n_dims]
    V = get_velocity_embedding(adata, basis, vkey)[:, :n_dims]
    indices = get_neighbor_indices(adata)

    per_cell = np.full(adata.n_obs, np.nan)
    r2 = np.full(adata.n_obs, np.nan)
    for i in range(adata.n_obs):
        if np.isnan(phi[i]):
            continue
        nb = np.array(
            [n for n in indices[i] if n != i and not np.isnan(phi[n])], dtype=int
        )
        if nb.size < min_neighbors:
            continue

        D = X[nb] - X[i]
        dphi = wrap_angle(phi[nb] - phi[i])
        if not np.any(np.linalg.norm(D, axis=1) > 0) or not np.any(dphi):
            continue

        g, *_ = np.linalg.lstsq(D, dphi, rcond=None)
        if np.linalg.norm(g) == 0:
            continue
        r2[i] = 1 - np.sum((D @ g - dphi) ** 2) / np.sum(dphi**2)
        per_cell[i] = rowwise_cosine(V[i][None, :], g[None, :])[0]

    if min_r2 is not None:
        per_cell[~(r2 >= min_r2)] = np.nan
    if np.all(np.isnan(per_cell)):
        raise NotApplicable("no cell had a usable phase neighbourhood")

    frac = np.mod(phi / (2 * np.pi), 1.0)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.minimum(np.digitize(frac, edges) - 1, n_bins - 1)
    per_group = {}
    for b in range(n_bins):
        vals = per_cell[(which == b) & np.isfinite(per_cell)]
        if vals.size:
            per_group[f"{edges[b]:.2g}-{edges[b + 1]:.2g}"] = float(vals.mean())

    scored = np.isfinite(per_cell)
    return MetricResult(
        name="phase_dir",
        value=nanmean(per_cell),
        detail=f"median R2 {np.nanmedian(r2):.2f}; {scored.mean():.0%} of cells scored",
        per_cell=per_cell,
        per_group=per_group,
    )


@metric
def truth_cos(
    adata,
    reference,
    *,
    vkey: str = "velocity",
    ref_vkey: str = "velocity",
    genes=None,
):
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
    genes : sequence of str, optional
        Score only these genes.  ``None`` uses every gene shared by the two
        runs and valid in both.

    Returns
    -------
    MetricResult
        ``per_cell`` holds the per-cell cosine; ``detail`` the number of genes
        scored.  ``not_applicable`` when the velocity is not gene-space, the
        cell counts differ, or fewer than 10 genes are shared and valid in
        both runs.

    Notes
    -----
    A method whose velocity lives in a learned latent space has no gene-wise
    correspondence to the reference, so the cosine would be meaningless;
    those return ``not_applicable`` rather than a number.  Scoring them needs a
    space-free measure -- distance correlation between the two neighbourhood
    geometries -- which is deliberately not implemented yet.

    Methods that leave poorly fitted genes ``nan`` (scVelo's dynamical model)
    are scored on fewer genes than those that fill every gene, and the value
    moves with the gene set.  Pass the same *genes* to every method being
    compared.

    The value is only meaningful next to controls on the same data: a
    cell-shuffled velocity keeps each gene's average direction and scores
    above 0.  The reference is built outside veloeval; for one-shot labelling
    fit the new/total slope on unperturbed, steady-state cells.
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

    shared = _shared_genes(adata, reference, genes)
    ia = adata.var_names.get_indexer(shared)
    ib = reference.var_names.get_indexer(shared)
    V = get_velocity(adata, vkey, drop_nan_genes=False)[:, ia]
    R = get_velocity(reference, ref_vkey, drop_nan_genes=False)[:, ib]

    keep = ~(np.isnan(V).any(axis=0) | np.isnan(R).any(axis=0))
    if keep.sum() < 10:
        raise NotApplicable("fewer than 10 genes valid in both runs")

    per_cell = rowwise_cosine(V[:, keep], R[:, keep])
    return MetricResult(
        name="truth_cos",
        value=nanmean(per_cell),
        detail=f"{int(keep.sum())} genes",
        per_cell=per_cell,
    )


@metric
def gamma_corr(
    adata,
    reference,
    *,
    gamma_key: str = "fit_gamma",
    ref_gamma_key: str = "fit_gamma",
    genes=None,
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
    genes : sequence of str, optional
        Correlate only these genes.  ``None`` uses every gene shared by the
        two runs with a rate on both sides.

    Returns
    -------
    MetricResult
        ``detail`` gives the number of genes correlated.  ``not_applicable``
        when either side exposes no per-gene rate or fewer than 10 genes are
        shared.

    Notes
    -----
    Only methods with an explicit rate parameter qualify.  Rate-free models
    (autoencoders regressing velocity directly) and models with no splicing
    rate concept are ``not_applicable`` by construction -- see the
    rate-assumption column of the methods table.

    Splicing data fix a gene's rates only up to that gene's time scale, so
    across genes only scale-free quantities are comparable.  For scVelo pass
    ``var['fit_gamma'] / var['fit_beta']`` (dynamical) or
    ``var['velocity_gamma']`` (steady-state), both :math:`\\gamma/\\beta`,
    rather than the default ``fit_gamma``.  Labelling measures
    :math:`\\gamma` itself, so even a perfect method agrees only as far as
    :math:`\\beta` is constant across genes.  As with :func:`truth_cos`,
    compare methods on the same *genes*.
    """
    if gamma_key not in adata.var:
        raise NotApplicable(f"method exposes no per-gene rate var['{gamma_key}']")
    if ref_gamma_key not in reference.var:
        raise NotApplicable(f"reference has no measured var['{ref_gamma_key}']")

    shared = _shared_genes(adata, reference, genes)
    ia = adata.var_names.get_indexer(shared)
    ib = reference.var_names.get_indexer(shared)
    a = adata.var[gamma_key].to_numpy(dtype=np.float64)[ia]
    b = reference.var[ref_gamma_key].to_numpy(dtype=np.float64)[ib]
    n = int(np.sum(~(np.isnan(a) | np.isnan(b))))
    if n < 10:
        raise NotApplicable(f"only {n} genes carry a rate on both sides")
    return MetricResult(name="gamma_corr", value=spearman(a, b), detail=f"{n} genes")


def _shared_genes(adata, reference, genes):
    shared = adata.var_names.intersection(reference.var_names)
    if genes is not None:
        shared = shared.intersection(pd.Index(genes))
    if len(shared) < 10:
        raise NotApplicable(f"only {len(shared)} genes shared with the reference")
    return shared
