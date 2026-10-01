"""Metrics scored against a truth that is independent of RNA.

Everything in :mod:`~veloeval.metrics.direction` and
:mod:`~veloeval.metrics.coherence` is scored against structure that was itself
read off the transcriptome, so a systematic error shared by the field is
invisible to them.  The metrics here use a signal the model never saw:

``phase_dir``     FUCCI fluorescence -- a protein-level, live-imageable readout
``truth_cos``     metabolic labelling -- a physical clock with real units
``gamma_corr``    labelling-derived degradation rates, per gene
``lineage_fate``  clonal barcodes -- where a progenitor's sisters ended up
``rate_err``      live imaging -- how long one cell cycle takes, in hours

They are the point of the benchmark, and they are also the ones with the
narrowest applicability: read the ``not_applicable`` details, not just the
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._math import nanmean, rowwise_cosine, spearman, wrap_angle
from ..access import (
    GENES_KEY,
    get_embedding,
    get_labels,
    get_neighbor_indices,
    get_stages,
    get_velocity,
    get_velocity_embedding,
    velocity_space,
)
from ..result import MetricResult, MissingInput, NotApplicable, metric
from ._markov import absorption, velocity_transitions

__all__ = ["phase_dir", "truth_cos", "gamma_corr", "lineage_fate", "rate_err"]


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

    Examples
    --------
    .. code-block:: python

        import scvelo as scv
        from veloeval import metrics as M

        # relative FUCCI position on [0, 1), e.g. dynamo's scEU-seq RPE1
        adata.obs["fucci_phase"] = adata.obs["Cell_cycle_relativePos"]
        res = M.phase_dir(adata)
        res.detail     # "median R2 0.47; 100% of cells scored"
        res.per_group  # {"0-0.1": ..., ..., "0.9-1": ...}

        # in the first 5 PCs instead of UMAP
        scv.tl.velocity_embedding(adata, basis="pca")
        M.phase_dir(adata, basis="pca", n_dims=5)
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
        runs and valid in both -- restricted to ``var['veloeval_genes']`` if
        :func:`~veloeval.prepare` wrote it in reference mode.

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

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        from veloeval import metrics as M

        ref = ad.read_h5ad("labelling_reference.h5ad")  # same cells, same order
        genes = adata.var_names[adata.var["velocity_genes"]]
        res = M.truth_cos(adata, ref, genes=genes)
        res.detail  # number of genes scored
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
        two runs with a rate on both sides -- restricted to
        ``var['veloeval_genes']`` if :func:`~veloeval.prepare` wrote it in
        reference mode.

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

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        from veloeval import metrics as M

        ref = ad.read_h5ad("labelling_reference.h5ad")
        adata.var["gamma_over_beta"] = adata.var["fit_gamma"] / adata.var["fit_beta"]
        res = M.gamma_corr(adata, ref, gamma_key="gamma_over_beta", ref_gamma_key="gamma")
        res.detail
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
    if genes is None and GENES_KEY in adata.var:
        genes = adata.var_names[adata.var[GENES_KEY].to_numpy(dtype=bool)]
    shared = adata.var_names.intersection(reference.var_names)
    if genes is not None:
        shared = shared.intersection(pd.Index(genes))
    if len(shared) < 10:
        raise NotApplicable(f"only {len(shared)} genes shared with the reference")
    return shared


_ABSORBED_TOL = 1e-2


@metric
def lineage_fate(
    adata,
    *,
    clone_key: str,
    time_key: str,
    label_key: str,
    fates: tuple[str, str] = ("Neutrophil", "Monocyte"),
    progenitor: str = "undiff",
    vkey: str = "velocity",
    scale: float = 10.0,
    min_sisters: int = 1,
):
    r"""Clonal fate prediction: does the field send progenitors where their sisters went?

    The fate-prediction benchmark of LARRY (Weinreb et al., *Science* 2020).
    For each barcoded progenitor at the earliest time point, the observed fate
    bias is the fraction of its later clonal relatives that became fate
    :math:`a` rather than :math:`b`,

    .. math::

        p_i = \frac{n_a(c_i)}{n_a(c_i) + n_b(c_i)},

    and the prediction is :math:`\hat p_i = B_a(i) / (B_a(i) + B_b(i))`, with
    :math:`B_a(i)` the probability that a random walk on the method's velocity
    graph from cell :math:`i` is first absorbed in a cell annotated :math:`a`.
    Every annotated non-progenitor cell, of any type and time point, is
    absorbing, so the terminal states are fixed by the annotation and shared by
    all methods.  The score is the Pearson correlation of :math:`\hat p` and
    :math:`p`.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry scVelo's velocity graph ``{vkey}_graph`` **and**
        ``{vkey}_graph_neg``, and the three ``obs`` columns below.
    clone_key : str
        Column holding each cell's clone ID; ``nan`` for cells without a barcode
        (or with several).
    time_key : str
        Column holding the collection time; numeric, or an ordered
        ``Categorical``.  The earliest value is where progenitors are scored.
    label_key : str
        Column holding the cell-type annotation.
    fates : (str, str), default: ("Neutrophil", "Monocyte")
        The two fates whose balance is predicted.
    progenitor : str, default: "undiff"
        Annotation of the progenitors -- the transient cells.
    vkey : str, default: "velocity"
        Velocity key prefix.
    scale : float, default: 10.0
        Inverse temperature of the transitions,
        :math:`\propto \exp(\text{scale}\cos_{ij})` over every neighbour the
        graph scored, no self-transitions; scVelo's default.  ``0`` ignores the
        velocity direction altogether -- see Notes.
    min_sisters : int, default: 1
        Fewest later :math:`a`/:math:`b` relatives a progenitor's clone must
        have to be scored.  Larger values give a less noisy truth on fewer cells.

    Returns
    -------
    MetricResult
        ``per_cell`` holds :math:`\hat p` for the scored progenitors and
        ``nan`` elsewhere; ``detail`` gives how many progenitors and clones were
        scored, and how many were trapped (Notes).
        ``not_applicable`` with a single time point, fewer than 10 scorable
        progenitors or fewer than 10 that are not trapped, or a constant
        prediction or truth.

    Notes
    -----
    Report it next to its ``scale=0`` value.  Without the velocity direction
    the walk is a plain diffusion on the neighbour graph, and a progenitor
    closer to one branch still tends to end up there, so the floor is well
    above 0.  Above the floor the direction adds information; below it the
    field is doing harm.

    A field with sinks among the progenitors traps the walk: at ``scale=10``
    a step against the velocity is :math:`e^{20}` times less likely than one
    along it, so escaping takes more steps than double precision resolves and
    the absorption probabilities no longer sum to 1.  Such a progenitor gets
    no prediction, :math:`\hat p = 0.5`, and is still scored -- dropping it
    instead would reward a field for its sinks, since the progenitors it
    traps are the ones it gets wrong.  ``detail`` counts them.  A reversed
    field is the extreme case: on a synthetic LARRY-like tree every
    progenitor is trapped and the result is ``not_applicable`` rather than a
    low score -- itself a sign that the field points the wrong way, to be
    confirmed with :func:`~veloeval.metrics.cbdir`.  CellRank's solvers
    refuse the same systems (fate probabilities that do not sum to 1).

    The truth is a fraction over a few sisters and is noisy when they are few.
    Compare methods only within one dataset and one neighbour graph, and keep
    the graph connected: cells that can reach no absorbing cell are dropped.
    The sparse solve grows quickly with the number of cells and neighbours;
    subsample, keeping every barcoded cell.

    Weinreb et al. scored only their curated Neutrophil/Monocyte trajectory
    subset; here every barcoded progenitor at the first time point with
    enough sisters is scored, so values match theirs in magnitude only.

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        kw = dict(clone_key="clone", time_key="Time point",
                  label_key="Cell type annotation")
        res = M.lineage_fate(adata, **kw)
        floor = M.lineage_fate(adata, **kw, scale=0)  # no velocity direction
        res.value, floor.value
        res.detail  # progenitors and clones scored
    """
    a, b = fates
    labels = get_labels(adata, label_key).astype(str)
    missing = [x for x in (a, b, progenitor) if x not in set(labels)]
    if missing:
        raise ValueError(f"not found in obs['{label_key}']: {missing}")
    if clone_key not in adata.obs:
        raise MissingInput(f"obs['{clone_key}']")
    stages = get_stages(adata, time_key)
    if len(stages) < 2:
        raise NotApplicable(f"obs['{time_key}'] has a single time point")
    T = velocity_transitions(adata, vkey, scale)

    time = adata.obs[time_key]
    early = (time == stages[0]).to_numpy()
    later = (time.notna() & (time != stages[0])).to_numpy()
    clone = adata.obs[clone_key].to_numpy()
    barcoded = pd.notna(clone)
    to_a = pd.Series(clone[barcoded & later & (labels == a)]).value_counts()
    to_b = pd.Series(clone[barcoded & later & (labels == b)]).value_counts()
    cand = np.flatnonzero(barcoded & early & (labels == progenitor))
    n_a = pd.Series(clone[cand]).map(to_a).fillna(0).to_numpy()
    n_b = pd.Series(clone[cand]).map(to_b).fillna(0).to_numpy()
    keep = n_a + n_b >= max(min_sisters, 1)
    scored = cand[keep]
    observed = n_a[keep] / (n_a[keep] + n_b[keep])

    terminal = labels != progenitor
    rest = terminal & (labels != a) & (labels != b)
    B = absorption(T, terminal, [labels == a, labels == b, rest])[scored]
    with np.errstate(invalid="ignore", divide="ignore"):
        predicted = B[:, 0] / (B[:, 0] + B[:, 1])
    total = B.sum(axis=1)
    trapped = np.isfinite(total) & (np.abs(total - 1) > _ABSORBED_TOL)
    predicted[trapped] = 0.5
    n_trapped = int(trapped.sum())

    good = np.isfinite(predicted)
    if good.sum() < 10:
        raise NotApplicable(
            f"{int(good.sum())} of {len(scored)} progenitors at {stages[0]} with "
            f">= {min_sisters} '{a}'/'{b}' sisters have a finite prediction"
        )
    if good.sum() - n_trapped < 10:
        raise NotApplicable(
            f"the walk from {n_trapped} of {int(good.sum())} progenitors is trapped "
            "among progenitors, i.e. the field has sinks there"
        )
    if np.ptp(predicted[good]) == 0 or np.ptp(observed[good]) == 0:
        raise NotApplicable("predicted or observed fate bias is constant")

    per_cell = np.full(adata.n_obs, np.nan)
    per_cell[scored] = predicted
    n_clones = pd.Series(clone[scored[good]]).nunique()
    return MetricResult(
        name="lineage_fate",
        value=float(np.corrcoef(predicted[good], observed[good])[0, 1]),
        detail=f"{int(good.sum())} progenitors at {stages[0]} from {n_clones} clones"
        + (f"; {n_trapped} trapped, scored as 0.5" if n_trapped else ""),
        per_cell=per_cell,
    )


@metric
def rate_err(
    adata, *, period: float, period_key: str = "cycle_period", half_life: float = 1.0
):
    r"""Relative error of the inferred cell-cycle period against live imaging.

    .. math::

        \hat T = |P| \cdot \text{half\_life}, \qquad
        \mathrm{RateErr} = \frac{|\hat T - T^*|}{T^*}

    where :math:`P` is the period the method inferred in its own time unit and
    :math:`T^*` the measured one, in hours.  The comparison is VeloCycle's own
    validation (Lederer et al., *Nat Methods* 2024), which measured RPE1 cycles
    of 17.7 h by time-lapse imaging.

    Lower is better; range ``[0, inf)``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry the inferred period in ``uns[period_key]``.
    period : float
        Measured period :math:`T^*`, in hours.
    period_key : str, default: "cycle_period"
        ``uns`` key of the inferred period, written by the pipeline from
        VeloCycle's posterior.
    half_life : float, default: 1.0
        Hours per unit of the method's time.  VeloCycle's unit is the mean
        transcript half-life, which its paper takes as 1 h.

    Returns
    -------
    MetricResult
        The relative error; ``detail`` gives both periods, the signed error and
        the half-life used.  ``not_applicable`` without ``uns[period_key]`` or
        with a zero or non-finite period.

    Notes
    -----
    Cannot detect a reversed field -- the sign of the period is dropped; score
    direction with :func:`~veloeval.metrics.phase_corr`.

    Splicing-based velocity has no identifiable time unit, so only VeloCycle
    yields a value: this checks whether its absolute rate holds up, it does not
    rank methods.  The result scales linearly with *half_life*: a 20% error in
    the half-life assumption is a 20% error in the period.  The measured
    periods themselves vary between cells (s.d. / mean about 0.19 in RPE1), so
    errors below about 0.2 are within that spread.

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        res = M.rate_err(adata, period=17.7)  # RPE1, time-lapse imaging
        res.value
        res.detail  # "18.4 h inferred vs 17.7 h measured (+4%); half-life 1 h"
    """
    if not (period > 0 and half_life > 0):
        raise ValueError("period and half_life must be positive")
    if period_key not in adata.uns:
        raise NotApplicable(
            f"method infers no cell-cycle period with a time unit (uns['{period_key}'])"
        )
    inferred = abs(float(adata.uns[period_key])) * half_life
    if not np.isfinite(inferred) or inferred == 0:
        raise NotApplicable(f"inferred period is {inferred}")
    rel = (inferred - period) / period
    return MetricResult(
        name="rate_err",
        value=abs(rel),
        detail=(
            f"{inferred:.1f} h inferred vs {period:.1f} h measured ({rel:+.0%}); "
            f"half-life {half_life:g} h"
        ),
    )
