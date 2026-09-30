"""Agreement between methods: do their fields send cells to the same places?

Agreement is consensus, not correctness.  It is the one metric that takes
several runs, because a consensus is only defined over a set of methods.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .._math import nanmean
from ..access import get_velocity_graph
from ..result import MetricResult, MissingInput, failed, missing_input, not_applicable

__all__ = ["agreement"]

_NAME = "agreement"


def _transition_matrix(adata, vkey, scale, self_transitions):
    """``scv.utils.get_transition_matrix(adata, vgraph=velocity_graph)``, as CZ
    Biohub computed it: positive cosines only, row-normalised."""
    import anndata as ad
    import pandas as pd
    import scvelo as scv

    sub = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names))
    sub.uns[f"{vkey}_graph"] = get_velocity_graph(adata, vkey)
    T = scv.utils.get_transition_matrix(
        sub, vkey=vkey, scale=scale, self_transitions=self_transitions
    ).tocsr()
    T.data[~np.isfinite(T.data)] = 0
    T.eliminate_zeros()
    T.sort_indices()
    return T


def _candidates(adata, vkey):
    """Neighbours each cell's cosines were computed against, either sign."""
    graph = get_velocity_graph(adata, vkey)
    try:
        graph = graph + get_velocity_graph(adata, vkey, negative=True)
    except MissingInput:
        pass
    graph = graph.tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def _support_overlap(graphs):
    """Per cell, |intersection| / |union| of the candidate neighbour sets."""
    n = graphs[0].shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        sets = [set(G.indices[G.indptr[i] : G.indptr[i + 1]]) for G in graphs]
        union = set.union(*sets)
        if union:
            out[i] = len(set.intersection(*sets)) / len(union)
    return out


def _agreement_core(Ts):
    """Per-cell A2 (``(m, n)``) and A1 (``(m, m, n)``) for row-normalised CSRs."""
    m, n = len(Ts), Ts[0].shape[0]
    a2 = np.full((m, n), np.nan)
    a1 = np.full((m, m, n), np.nan)
    for i in range(n):
        sup = [T.indices[T.indptr[i] : T.indptr[i + 1]] for T in Ts]
        if any(s.size == 0 for s in sup):
            continue
        cols = np.unique(np.concatenate(sup))
        M = np.zeros((m, cols.size))
        for k, T in enumerate(Ts):
            M[k, np.searchsorted(cols, sup[k])] = T.data[T.indptr[i] : T.indptr[i + 1]]
        norm = np.linalg.norm(M, axis=1)
        med = np.median(M, axis=0)
        med_norm = np.linalg.norm(med)
        if med_norm > 0:
            a2[:, i] = (M @ med) / (norm * med_norm)
        a1[:, :, i] = (M @ M.T) / np.outer(norm, norm)
    return a2, a1


def agreement(
    runs: Mapping,
    *,
    vkey: str = "velocity",
    scale: float = 10.0,
    self_transitions: bool = True,
    min_methods: int = 5,
    min_overlap: float = 0.9,
) -> dict[str, MetricResult]:
    r"""Agreement of each method's transitions with the median across methods.

    For each cell :math:`i`, each method's transition row :math:`p_{M,i}` --
    scVelo's transition matrix built from that run's velocity graph -- is
    compared with the element-wise median over all methods, the method itself
    included (CZ Biohub, bioRxiv 2024, "A2"):

    .. math::

        \mathrm{A2}_{M} = \frac1n \sum_i \cos\bigl(p_{M,i},\
            \operatorname{median}_{M'} p_{M',i}\bigr)

    and each pair of methods with each other ("A1"),
    :math:`\cos(p_{M,i}, p_{M',i})`.  Transition probabilities are
    non-negative, so both lie in ``[0, 1]``.

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    runs : mapping of str to anndata.AnnData
        Method name to run, all on one dataset.  Each must carry scVelo's
        velocity graph ``{vkey}_graph`` and ideally ``{vkey}_graph_neg``, with
        the same cells in the same order.
    vkey : str, default: "velocity"
        Velocity key prefix.
    scale : float, default: 10.0
        Inverse temperature of the transition matrix.  scVelo's default, and
        what CZ Biohub used; :func:`~veloeval.metrics.ees` uses 30.
    self_transitions : bool, default: True
        Put scVelo's self-transition probability on the diagonal.  scVelo's
        default, and what CZ Biohub used.
    min_methods : int, default: 5
        Fewer scorable runs than this and every run is ``not_applicable``: the
        median includes the method itself, which dominates a small consensus.
        CZ Biohub compared five methods.
    min_overlap : float, default: 0.9
        Minimum median, over cells, of the fraction of candidate neighbours all
        runs share.  Below it the runs' graphs were built on different
        neighbourhoods, and the metric would compare graphs, not velocities.

    Returns
    -------
    dict of str to MetricResult
        One result per run, all named ``"agreement"``.  ``value`` is the mean
        A2 over cells, ``per_cell`` the per-cell A2, ``per_group`` maps every
        other method to the mean A1 against it.  A run without a velocity
        graph is ``missing_input`` and left out of the median.

    Notes
    -----
    Detects a reversed field only when it is in the minority: when most
    methods are reversed together, A2 rewards the reversed majority.  It
    measures consensus, not correctness.

    Every score depends on the set of methods -- adding one changes them all --
    so compare only within one dataset and one method set.

    Transition probabilities live on cells, not genes, so this is the one
    metric that compares gene-space and latent-space methods directly.

    All runs must share one neighbour graph: ``scv.tl.velocity_graph`` uses
    each run's own ``obsp['distances']``, so a wrapper that rebuilt the
    neighbours (in a latent space, say) is scored against different
    neighbours.  Rebuild those velocity graphs on the shared neighbour graph
    in the pipeline.  Without ``{vkey}_graph_neg`` the check sees only the
    positive-cosine neighbours and is stricter than it needs to be.

    Values are not guaranteed to match the CRM 2026 benchmark, which does not
    state its *scale* or self-transition setting.

    Requires scvelo (``pip install veloeval[prepare]``); without it every
    status is ``missing_input``.

    Examples
    --------
    .. code-block:: python

        import anndata as ad
        from veloeval import metrics as M

        runs = {
            m: ad.read_h5ad(f"2.velocity/pancreas/{m}/seed_42/velocity.h5ad")
            for m in ["scvelo_dynamical", "scvelo_stochastic", "velovi",
                      "unitvelo", "deepvelo"]
        }
        res = M.agreement(runs)
        res["velovi"].value      # mean A2 against the median
        res["velovi"].per_group  # mean A1 against each other method
    """
    names = list(runs)
    try:
        import scvelo  # noqa: F401
    except ImportError:
        return {m: missing_input(_NAME, "scvelo (pip install veloeval[prepare])")
                for m in names}

    try:
        return _agreement(
            runs, names, vkey, scale, self_transitions, min_methods, min_overlap
        )
    except Exception as exc:  # noqa: BLE001 - deliberate: status carries it
        return {m: failed(_NAME, exc) for m in names}


def _agreement(runs, names, vkey, scale, self_transitions, min_methods, min_overlap):
    out, Ts, graphs, scored = {}, [], [], []
    for m in names:
        try:
            Ts.append(_transition_matrix(runs[m], vkey, scale, self_transitions))
            graphs.append(_candidates(runs[m], vkey))
        except MissingInput as exc:
            out[m] = missing_input(_NAME, *exc.keys)
            continue
        scored.append(m)

    def all_not_applicable(detail):
        return {m: out.get(m) or not_applicable(_NAME, detail) for m in names}

    if scored:
        first = runs[scored[0]].obs_names
        if any(not runs[m].obs_names.equals(first) for m in scored[1:]):
            return all_not_applicable("runs do not share cells in the same order")
    if len(scored) < min_methods:
        return all_not_applicable(
            f"{len(scored)} runs with a velocity graph; needs {min_methods}"
        )

    overlap = float(np.nanmedian(_support_overlap(graphs)))
    if not overlap >= min_overlap:
        return all_not_applicable(
            f"median neighbour overlap {overlap:.2f} < {min_overlap}; rebuild the "
            "velocity graphs on one shared neighbour graph"
        )

    a2, a1 = _agreement_core(Ts)
    detail = f"{len(scored)} methods; median neighbour overlap {overlap:.2f}"
    for k, m in enumerate(scored):
        per_group = {
            other: nanmean(a1[k, j])
            for j, other in enumerate(scored)
            if j != k
        }
        out[m] = MetricResult(
            name=_NAME,
            value=nanmean(a2[k]),
            per_cell=a2[k],
            per_group=per_group,
            detail=detail,
        )
    return {m: out[m] for m in names}
