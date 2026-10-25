"""Direction correctness against known differentiation edges.

These metrics need curated ``cluster_edges`` -- a list of ``[source, target]``
cell-type pairs that are known to be a differentiation step.  A dataset without
them yields ``not_applicable``, which is a property of the dataset, not a
failure of the method.

CBDir is computed in a shared low-dimensional embedding, as in VeloAE (Qiao &
Huang, PNAS 2021) and the Genome Biology benchmark (2026), or with
``basis=None`` on the gene-space cosines of scVelo's velocity graph.  CBVCoh uses the same
embedding so that methods whose velocity lives in different native spaces --
including latent-space ones -- stay comparable; VeloAE itself computed it on
``layers``.  The basis must be identical across every method in a comparison.
"""

from __future__ import annotations

import numpy as np

from .._math import nanmean, rowwise_cosine
from ..access import (
    GENES_KEY,
    _dense,
    get_embedding,
    get_labels,
    get_neighbor_indices,
    get_stages,
    get_velocity_embedding,
    get_velocity_graph,
    velocity_space,
)
from ..result import MetricResult, MissingInput, NotApplicable, metric

__all__ = ["cbdir", "cbvcoh", "cbturn", "cto"]


_BOUNDARY_RULES = ("target", "non_source")


def _boundary_groups(labels, cluster_edges, indices, boundary="target"):
    """Yield ``(edge_name, [(cell, boundary_neighbours), ...])`` per edge.

    Grouped by edge rather than flattened because the outermost average is over
    edges: a source cluster feeding four targets contributes four edge scores,
    and flattening would weight it by how many boundary cells each target
    happens to have.
    """
    if boundary not in _BOUNDARY_RULES:
        raise ValueError(
            f"boundary must be one of {_BOUNDARY_RULES}, got {boundary!r}"
        )
    if not cluster_edges:
        raise NotApplicable("dataset has no curated cluster_edges")

    seen_any = False
    for src, tgt in cluster_edges:
        src_cells = np.where(labels == src)[0]
        tgt_cells = np.where(labels == tgt)[0]
        if src_cells.size == 0 or tgt_cells.size == 0:
            continue
        tgt_set = set(tgt_cells.tolist())
        members = []
        for i in src_cells:
            if boundary == "target":
                nbrs = [n for n in indices[i] if n in tgt_set and n != i]
            else:
                nbrs = [n for n in indices[i] if labels[n] != src]
            if nbrs:
                members.append((i, np.array(nbrs, dtype=int)))
        if members:
            seen_any = True
            yield f"{src} -> {tgt}", members

    if not seen_any:
        raise NotApplicable(
            "no cell had a kNN neighbour across any cluster edge; "
            "check that label values match cluster_edges"
        )


def _graph_cosines(adata, vkey):
    """Scorer reading the velocity graph's cosine for each (cell, neighbour) pair.

    ``nan`` for a pair the graph never scored.  Also returns a running count of
    pairs looked up and of pairs absent.
    """
    G = (
        get_velocity_graph(adata, vkey) + get_velocity_graph(adata, vkey, negative=True)
    ).tocsr()
    G.sort_indices()
    seen = {"pairs": 0, "absent": 0}

    def score(i, nbrs):
        cols = G.indices[G.indptr[i] : G.indptr[i + 1]]
        vals = G.data[G.indptr[i] : G.indptr[i + 1]]
        out = np.full(len(nbrs), np.nan)
        if cols.size:
            at = np.minimum(np.searchsorted(cols, nbrs), cols.size - 1)
            found = cols[at] == nbrs
            out[found] = vals[at[found]]
        seen["pairs"] += len(nbrs)
        seen["absent"] += int(np.isnan(out).sum())
        return out

    return score, seen


def _aggregate_over_edges(labels, cluster_edges, indices, n_obs, score, boundary):
    """Average *score* over neighbours, then over cells per edge, then over edges.

    Returns ``(value, per_cell, per_group)``.  ``per_cell`` is a cell's mean over
    the edges it takes part in, so a cell whose cluster is the source of several
    edges keeps all of them and the result does not depend on the order
    *cluster_edges* was written in.
    """
    per_group: dict[str, float] = {}
    total = np.zeros(n_obs, dtype=np.float64)
    seen = np.zeros(n_obs, dtype=np.int64)

    groups = _boundary_groups(labels, cluster_edges, indices, boundary)
    for name, members in groups:
        scores = []
        for i, boundary in members:
            s = nanmean(score(i, boundary))
            if np.isnan(s):
                continue
            scores.append(s)
            total[i] += s
            seen[i] += 1
        if scores:
            per_group[name] = float(np.mean(scores))

    if not per_group:
        raise NotApplicable("no cluster edge yielded a finite score")

    per_cell = np.divide(
        total, seen, out=np.full(n_obs, np.nan), where=seen > 0
    )
    return float(np.mean(list(per_group.values()))), per_cell, per_group


@metric
def cbdir(
    adata,
    *,
    label_key: str,
    cluster_edges,
    basis: str | None = "umap",
    vkey: str = "velocity",
    boundary: str = "target",
):
    r"""Cross-boundary direction correctness.

    For each edge ``A -> B`` and each cell ``i`` in ``A``, take the cosine
    between ``i``'s velocity and the displacement towards each kNN neighbour
    that lies in ``B``.  Averaged over neighbours, then over the boundary cells
    of that edge, then over edges -- so every edge carries equal weight
    regardless of how many cells sit on it.

    Higher is better; range ``[-1, 1]``.  A field pointing perfectly along
    every known edge scores 1, its reverse -1, an unstructured one ~0.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry ``obsm['veloeval_knn']`` and, with a *basis*,
        ``obsm['X_{basis}']`` and ``obsm['{vkey}_{basis}']``; with
        ``basis=None``, scVelo's ``{vkey}_graph`` and ``{vkey}_graph_neg`` --
        see :func:`veloeval.prepare`.
    label_key : str
        Column in ``adata.obs`` holding cell-type labels.  Its values must use
        the same spelling as *cluster_edges*.
    cluster_edges : list of [str, str]
        Known differentiation steps, e.g.
        ``[["Ductal", "Ngn3 low EP"], ["Ngn3 low EP", "Ngn3 high EP"]]``.
        ``None`` yields ``not_applicable``: a dataset without curated edges
        cannot be scored this way, which is a fact about the dataset.
    basis : str or None, default: "umap"
        Embedding both the positions and the velocity are read in.  ``None``
        reads the cosine of each (cell, neighbour) pair from the velocity graph
        instead, in gene space -- see Notes.  Must be the same for every method
        in a comparison.
    vkey : str, default: "velocity"
        Velocity key prefix.
    boundary : {"target", "non_source"}, default: "target"
        Which kNN neighbours of a source cell count as across the boundary.
        ``"target"`` keeps only those in ``B``, as in VeloAE and the Genome
        Biology benchmark.  ``"non_source"`` keeps every neighbour outside
        ``A``, so ``B`` no longer takes part: edges that share a source score
        identically, and upstream neighbours -- which a correct field points
        away from -- are mixed in with downstream ones.

    Returns
    -------
    MetricResult
        ``value`` is the mean over edges; ``per_group`` maps ``"A -> B"`` to
        that edge's score; ``per_cell`` holds each cell's mean over the edges it
        takes part in, ``nan`` where it had no cross-boundary neighbour.
        ``status`` is ``not_applicable`` without *cluster_edges* or when no cell
        has a neighbour across any edge; with ``basis=None`` also outside gene
        space, or when more than 10% of the boundary pairs are missing from the
        velocity graph, which then was built on other neighbours than
        ``obsm['veloeval_knn']``.

    Notes
    -----
    By default computed in the embedding, as in VeloAE (Qiao & Huang, *PNAS*
    2021) and the Genome Biology benchmark (2026).

    ``basis=None`` removes the dependence on the embedding.  The cosine is
    scVelo's: between the neighbour's displacement in ``Ms`` and the velocity,
    each first centred across genes -- a Pearson correlation.  The embedding
    smooths out most of the noise in single-neighbour displacements, gene
    space does not, so values run far lower: on scVelo's pancreas data 0.19
    (dynamical), 0.09 (deterministic) and 0.03 (shuffled control), against
    0.52, 0.50 and 0.18 in UMAP.  Compare methods only within one basis;
    record the basis with the result and never pool the two.  Reversing the
    field flips the sign exactly, which the UMAP projection, through
    :math:`\exp(10\cos)`, does not.  Not invariant to rescaling single genes.
    Only gene-space velocities are scored: a latent-space method's graph lives
    in its own space.  For the methods to share one set of neighbours and one
    ``Ms``, build their graphs with ``prepare(..., reference=ref)``.

    Examples
    --------
    .. code-block:: python

        import veloeval as ve
        from veloeval import metrics as M

        ve.prepare(adata, space="gene", basis="umap")
        edges = [
            ["Ngn3 low EP", "Ngn3 high EP"],
            ["Ngn3 high EP", "Pre-endocrine"],
            ["Pre-endocrine", "Beta"],
        ]
        res = M.cbdir(adata, label_key="clusters", cluster_edges=edges)
        res.value      # mean over edges
        res.per_group  # {"Ngn3 low EP -> Ngn3 high EP": ..., ...}
        res.per_cell   # score of each boundary cell, nan elsewhere
    """
    labels = get_labels(adata, label_key)
    indices = get_neighbor_indices(adata)
    if basis is None:
        space = velocity_space(adata)
        if space != "gene":
            raise NotApplicable(
                f"basis=None reads a gene-space velocity graph; velocity lives in "
                f"{space!r} space"
            )
        score, seen = _graph_cosines(adata, vkey)
        value, per_cell, per_group = _aggregate_over_edges(
            labels, cluster_edges, indices, adata.n_obs, score, boundary
        )
        if seen["absent"] > 0.1 * seen["pairs"]:
            raise NotApplicable(
                f"{seen['absent']} of {seen['pairs']} boundary pairs are not in the "
                "velocity graph, which was built on other neighbours than "
                "obsm['veloeval_knn']; use prepare(reference=...)"
            )
        return MetricResult(
            name="cbdir",
            value=value,
            per_cell=per_cell,
            per_group=per_group,
            detail=f"gene space; {seen['absent']} of {seen['pairs']} boundary pairs "
            "absent from the velocity graph",
        )

    X = get_embedding(adata, basis)
    V = get_velocity_embedding(adata, basis, vkey)

    def score(i, nbrs):
        disp = X[nbrs] - X[i]
        return rowwise_cosine(disp, np.broadcast_to(V[i], disp.shape))

    return _aggregate_over_edges(
        labels, cluster_edges, indices, adata.n_obs, score, boundary
    )


@metric
def cbvcoh(
    adata,
    *,
    label_key: str,
    cluster_edges,
    basis: str = "umap",
    vkey: str = "velocity",
    boundary: str = "target",
):
    """Cross-boundary velocity coherence.

    Same boundary cells as :func:`cbdir`, but compares ``i``'s velocity with
    the *velocity* of each boundary neighbour rather than with the displacement
    towards it.  Measures continuity of the field across the boundary.

    Higher is better; range ``[-1, 1]``.  Unchanged under ``v -> -v``: the edge
    direction plays no part, only which clusters are adjacent.

    Parameters
    ----------
    adata : anndata.AnnData
        As for :func:`cbdir`.
    label_key : str
        Column in ``adata.obs`` holding cell-type labels.
    cluster_edges : list of [str, str]
        Known differentiation steps.  ``None`` yields ``not_applicable``.
    basis : str, default: "umap"
        Embedding the velocity is read in.
    vkey : str, default: "velocity"
        Velocity key prefix.
    boundary : {"target", "non_source"}, default: "target"
        As for :func:`cbdir`.

    Returns
    -------
    MetricResult
        ``value`` is the mean over edges; ``per_group`` maps ``"A -> B"`` to
        that edge's score; ``per_cell`` holds each cell's mean over the edges it
        takes part in.

    See Also
    --------
    cbdir : whether the field points the right way across the same boundary.

    Notes
    -----
    Coherence is not correctness: a field that crosses the boundary smoothly in
    the *wrong* direction scores exactly as high here and low in :func:`cbdir`.
    Read the two together.

    VeloAE computed this on ``layers`` (gene space).  It is read in the
    embedding here so latent-space methods can be scored alongside the rest.
    Method rankings on this metric can differ between spaces, so compare
    methods only within one *basis*.

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        edges = [
            ["Ngn3 low EP", "Ngn3 high EP"],
            ["Ngn3 high EP", "Pre-endocrine"],
            ["Pre-endocrine", "Beta"],
        ]
        res = M.cbvcoh(adata, label_key="clusters", cluster_edges=edges)
        res.per_group

        # count every neighbour outside the source cluster, not just those in B
        M.cbvcoh(adata, label_key="clusters", cluster_edges=edges, boundary="non_source")
    """
    labels = get_labels(adata, label_key)
    indices = get_neighbor_indices(adata)
    V = get_velocity_embedding(adata, basis, vkey)

    def score(i, nbrs):
        return rowwise_cosine(V[nbrs], np.broadcast_to(V[i], V[nbrs].shape))

    return _aggregate_over_edges(
        labels, cluster_edges, indices, adata.n_obs, score, boundary
    )


# Below this mean squared spread the edge directions are too close to parallel
# for their differences to carry a signal; provisional until set on simulations.
_MIN_EDGE_SPREAD = 0.05


@metric
def cbturn(
    adata,
    reference=None,
    *,
    label_key: str,
    cluster_edges,
    xkey: str = "Ms",
    vkey: str = "velocity",
    n_null: int = 200,
    seed: int = 0,
):
    r"""Cross-boundary turning.

    Whether the velocity changes direction from one cluster edge to the next
    the way the data do.  For each edge :math:`e` (A -> B) take the boundary
    cells of :func:`cbdir` -- cells of A with a kNN neighbour in B -- and the
    mean of their unit velocities :math:`\bar u_e`; the edge's own direction is
    :math:`\hat d_e`, the unit difference of the B and A means of
    ``layers[xkey]``.  After removing what all edges share,

    .. math::

        \mathrm{CBTurn} = \frac{\sum_e \langle \bar u_e - \bar u,\;
        \hat d_e - \bar d \rangle}{\sum_e \lVert \hat d_e - \bar d \rVert^2},

    with :math:`\bar u, \bar d` the means over edges: the slope of the
    edge-specific velocity on the edge-specific direction.

    Parameters
    ----------
    adata : anndata.AnnData
        Velocity in gene or latent space, cell labels and the shared kNN.
    reference : anndata.AnnData, optional
        Where ``layers[xkey]`` is read, cells looked up by name -- the reference
        of :func:`~veloeval.prepare` in gene space, so that every method is
        scored against the same edge directions.  ``None`` reads *adata*'s own
        ``layers[xkey]``: in latent space the method's coordinates, comparable
        only within the method.
    label_key : str
        ``obs`` column holding the cell types.
    cluster_edges : list of [source, target]
        Known differentiation steps; at least two, not all parallel.
    xkey : str, default: ``"Ms"``
    vkey : str, default: ``"velocity"``
    n_null : int, default: 200
        Shuffles of the velocities across cells for the null; ``0`` skips it.
    seed : int, default: 0
        Seed of the shuffles.

    Returns
    -------
    MetricResult
        ``detail`` gives the number of edges, the global part
        :math:`\langle \bar u, \bar d \rangle` and the 97.5% quantile of the
        score with velocities shuffled across cells.  ``per_group`` holds each
        edge's term of the sum; they add up to the value.  ``not_applicable``
        in embedding or scalar space, with fewer than two edges scored, or when
        the edge directions are nearly parallel.

    Notes
    -----
    Every constant field -- one vector for all cells -- scores 0 up to rounding, and so
    does any field whose boundary cells point the same way on every edge; a
    reversed field scores the negative.  :math:`\langle \bar u_e, \hat d_e
    \rangle` averaged over edges, a centroid version of :func:`cbdir`, is the
    global part plus the turning term, so a field can score well on
    :func:`cbdir` from the global part alone; this metric keeps only the
    turning term.  Being a slope and not a correlation, a field that is 99%
    constant scores about 1% of the field it is mixed from.  It is not bounded
    by 1: the mean of unit velocities is shorter than 1 unless they agree, and
    the ceiling depends on the dataset, so compare methods within a dataset.

    The velocity is read on ``var['veloeval_genes']`` if present, otherwise
    on ``var['{vkey}_genes']`` (if present) with finite velocity.  Not in an
    embedding: projecting velocity through the neighbour graph turns it
    along the data on its own -- on pancreas a constant gene-space field
    scores 0.42 in the UMAP, next to 0.46 for scVelo's dynamical model.

    Examples
    --------
    .. code-block:: python

        import veloeval as ve
        from veloeval import metrics as M

        ve.prepare(adata, reference=ref)
        res = M.cbturn(adata, ref, label_key="clusters", cluster_edges=edges)
        res.value, res.detail
    """
    space = velocity_space(adata)
    if space not in ("gene", "latent"):
        raise NotApplicable(
            f"velocity is in {space!r} space; projected velocity turns with the "
            "neighbour graph, so turning is read in gene or latent space only"
        )
    labels = get_labels(adata, label_key)
    indices = get_neighbor_indices(adata)
    V, X = _turning_inputs(adata, reference, xkey, vkey)

    norm = np.linalg.norm(V, axis=1, keepdims=True)
    U = np.divide(V, norm, out=np.zeros_like(V), where=norm > 0)
    names, cells, D = [], [], []
    for name, members in _boundary_groups(labels, cluster_edges, indices):
        src, tgt = name.split(" -> ")
        d = X[labels == tgt].mean(axis=0) - X[labels == src].mean(axis=0)
        if np.linalg.norm(d) == 0:
            continue
        names.append(name)
        cells.append(np.array([i for i, _ in members]))
        D.append(d / np.linalg.norm(d))
    if len(names) < 2:
        raise NotApplicable(f"{len(names)} cluster edge scored; turning needs two")
    D = np.array(D)
    Dc = D - D.mean(axis=0)
    spread = float((Dc * Dc).sum(axis=1).mean())
    if spread < _MIN_EDGE_SPREAD:
        raise NotApplicable(
            f"edge directions nearly parallel (mean squared spread {spread:.3f})"
        )
    denom = float((Dc * Dc).sum())

    def edge_means(U):
        return np.array([U[c].mean(axis=0) for c in cells])

    B = edge_means(U)
    terms = ((B - B.mean(axis=0)) * Dc).sum(axis=1) / denom
    value = float(terms.sum())
    detail = f"{len(names)} edges; global part {B.mean(axis=0) @ D.mean(axis=0):.3f}"
    if n_null > 0:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_null):
            Bp = edge_means(U[rng.permutation(len(U))])
            null.append(((Bp - Bp.mean(axis=0)) * Dc).sum() / denom)
        detail += f"; shuffled q97.5 {np.quantile(null, 0.975):.3f}"
    return MetricResult(
        name="cbturn",
        value=value,
        detail=detail,
        per_group=dict(zip(names, terms.tolist())),
    )


def _turning_inputs(adata, reference, xkey, vkey):
    """Velocity and ``layers[xkey]`` on the same features, cells in *adata*'s order."""
    if vkey not in adata.layers:
        raise MissingInput(f"layers['{vkey}']")
    source = adata if reference is None else reference
    if xkey not in source.layers:
        where = "reference" if reference is not None else "adata"
        raise MissingInput(f"{where} layers['{xkey}']")

    if GENES_KEY in adata.var:
        keep = adata.var[GENES_KEY].to_numpy(dtype=bool)
    elif f"{vkey}_genes" in adata.var:
        keep = adata.var[f"{vkey}_genes"].to_numpy(dtype=bool)
    else:
        keep = np.ones(adata.n_vars, dtype=bool)
    if reference is not None:
        keep &= adata.var_names.isin(reference.var_names)
        missing = adata.obs_names.difference(reference.obs_names)
        if len(missing):
            raise NotApplicable(
                f"{len(missing)} cells are not in the reference, e.g. {missing[0]!r}"
            )
    V = _dense(adata.layers[vkey][:, keep])
    finite = ~np.isnan(V).any(axis=0)
    features = adata.var_names[keep][finite]
    if len(features) < 2:
        raise NotApplicable(f"{len(features)} features with finite velocity")
    if reference is None:
        X = adata[:, features].layers[xkey]
    else:
        X = reference[adata.obs_names, features].layers[xkey]
    return V[:, finite], _dense(X)


def _ordered_stages(adata, stage_key: str):
    """Consecutive ``[earlier, later]`` pairs of the stages in ``obs[stage_key]``."""
    stages = get_stages(adata, stage_key)
    if len(stages) < 2:
        raise NotApplicable(f"obs['{stage_key}'] has fewer than two stages")
    return np.asarray(adata.obs[stage_key].values), list(zip(stages[:-1], stages[1:]))


def _fraction_later(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pairs ``(i in a, j in b)`` with ``a_i < b_j``; ties count as wrong."""
    return float(np.searchsorted(np.sort(a), b, side="left").sum()) / (a.size * b.size)


@metric
def cto(
    adata,
    *,
    label_key: str | None = None,
    cluster_edges=None,
    stage_key: str | None = None,
    time_key: str = "latent_time",
    fallback_key: str | None = None,
):
    r"""Cluster temporal ordering (CTO).

    For every known step ``A -> B``, the fraction of cell pairs
    :math:`(i \in A, j \in B)` whose inferred time is ordered correctly,
    averaged over steps:

    .. math::

        \mathrm{CTO} = \frac{1}{|E|} \sum_{(A, B) \in E}
        \frac{1}{|A|\,|B|} \sum_{i \in A} \sum_{j \in B} \mathbf{1}\{t_i < t_j\}

    This is equation (3) of the Genome Biology benchmark (2026).  The pairwise
    fraction goes back to the Time Accuracy Score of VeloVAE (Gu et al.), which
    differs in counting ties as correct and in also averaging indirect
    ancestor-descendant pairs.

    The steps come from one of two sources:

    - ``cluster_edges`` over ``obs[label_key]`` -- curated cell-type edges, as
      in VeloVAE;
    - ``stage_key`` -- an experimentally measured time label (e.g. collection
      day); every pair of consecutive stages is one step, as in the Genome
      Biology benchmark.

    Higher is better; range ``[0, 1]``.  0.5 is chance.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry ``obs[time_key]``.
    label_key : str, optional
        Column in ``adata.obs`` holding cell-type labels.  Required with
        *cluster_edges*.
    cluster_edges : list of [str, str], optional
        Known differentiation steps.  Only these direct edges are scored.
    stage_key : str, optional
        Column holding the measured time label: numeric, or an ordered
        ``Categorical`` for strings (anything else fails, since string sort
        order is not time order).  Mutually exclusive with *cluster_edges*.
    time_key : str, default: "latent_time"
        Column holding the method's own per-cell time.
    fallback_key : str or None, default: None
        Column read when *time_key* is absent, e.g. ``"velocity_pseudotime"``
        to reproduce the Genome Biology benchmark.  Off by default; see Notes.

    Returns
    -------
    MetricResult
        ``per_group`` holds the pairwise fraction of each step.  ``detail``
        names the time column that was scored and whether it was the fallback.
        ``not_applicable`` when neither time column exists, when neither steps
        source is available, or when no step has timed cells on both sides.

    Notes
    -----
    Only a method's own time is scored by default; a method that emits none is
    ``not_applicable``.  The Genome Biology benchmark instead fills such
    methods in with ``scvelo.tl.velocity_pseudotime``, but that is a diffusion
    pseudotime on the symmetrised velocity graph: velocity only picks the root
    and end cells, and the ordering comes from the manifold.  On pancreas a
    pure-noise velocity field scores about as well that way as scVelo's own
    latent time, so the fallback is opt-in and ``detail`` records when it was
    used.  Methods with gene-specific times should aggregate to one time (e.g.
    ``scvelo.tl.latent_time``) before scoring.

    Ties count as wrong, so a constant time scores 0.  Cells with ``nan`` time
    are dropped rather than set to 0.

    On a cyclic stage order (G1 -> S -> G2M -> G1) the closing step can be
    satisfied by no linear time; score cyclic processes with
    :func:`~veloeval.metrics.phase_corr` instead.

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        edges = [
            ["Ngn3 low EP", "Ngn3 high EP"],
            ["Ngn3 high EP", "Pre-endocrine"],
            ["Pre-endocrine", "Beta"],
        ]
        res = M.cto(
            adata, label_key="clusters", cluster_edges=edges, time_key="latent_time"
        )
        res.value

        # or order whole stages: numeric days or an ordered Categorical
        M.cto(adata, stage_key="day", time_key="latent_time")
    """
    if stage_key is not None and cluster_edges:
        raise ValueError("pass either cluster_edges or stage_key, not both")
    if stage_key is not None:
        labels, steps = _ordered_stages(adata, stage_key)
    elif cluster_edges:
        labels, steps = get_labels(adata, label_key), cluster_edges
    else:
        raise NotApplicable("dataset has no curated cluster_edges or stage labels")

    if time_key in adata.obs:
        used, detail = time_key, f"time: obs['{time_key}']"
    elif fallback_key and fallback_key in adata.obs:
        used = fallback_key
        detail = f"time: obs['{fallback_key}'] (fallback; no obs['{time_key}'])"
    else:
        tried = " or ".join(f"obs['{k}']" for k in (time_key, fallback_key) if k)
        raise NotApplicable(f"method has no per-cell time ({tried})")

    t = np.asarray(adata.obs[used].values, dtype=np.float64)
    per_group: dict[str, float] = {}
    for src, tgt in steps:
        a = t[labels == src]
        b = t[labels == tgt]
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if a.size and b.size:
            per_group[f"{src} -> {tgt}"] = _fraction_later(a, b)

    if not per_group:
        raise NotApplicable("no step had timed cells on both sides")
    return MetricResult(
        name="cto",
        value=float(np.mean(list(per_group.values()))),
        detail=detail,
        per_group=per_group,
    )
