"""Markov chain on a method's velocity graph, for the fate metrics."""

from __future__ import annotations

import numpy as np

from ..access import get_velocity_graph


def velocity_transitions(adata, vkey: str, scale: float):
    r"""Row-stochastic transitions :math:`\propto \exp(\text{scale}\cos_{ij})`.

    Over every neighbour the velocity graph scored, either sign, without
    self-transitions -- ``scvelo.tl.transition_matrix(scale=scale,
    self_transitions=False)`` when ``{vkey}_graph_neg`` is present, and the
    rows :func:`~veloeval.metrics.ees` reads.  ``scale=0`` is the uniform walk
    on the same neighbours.  Both graphs are required: without the negative
    one a cell whose cosines are all negative has no way out.
    """
    from scipy.sparse import diags

    C = get_velocity_graph(adata, vkey) + get_velocity_graph(adata, vkey, negative=True)
    C = C.tocsr().astype(np.float64)
    C = C - diags(C.diagonal())
    C.eliminate_zeros()
    C.data = np.exp(scale * C.data)
    rowsum = np.asarray(C.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.where(rowsum > 0, 1 / rowsum, 0)
    return (diags(inv) @ C).tocsr()


def absorption(T, absorbing, targets):
    """Probability that the first absorbing cell reached is in each target set.

    *absorbing* is a boolean mask; *targets* a list of boolean masks within it.
    Returns ``(n_obs, len(targets))``; NaN for cells that can reach no absorbing
    cell.  Solved exactly with a sparse LU factorisation.
    """
    from scipy.sparse import identity
    from scipy.sparse.linalg import splu

    absorbing = np.asarray(absorbing, dtype=bool)
    reach = absorbing.copy()
    while True:
        new = reach | (T @ reach.astype(np.float64) > 0)
        if (new == reach).all():
            break
        reach = new

    out = np.full((T.shape[0], len(targets)), np.nan)
    for k, t in enumerate(targets):
        out[absorbing, k] = np.asarray(t, dtype=np.float64)[absorbing]
    live = reach & ~absorbing
    if live.any():
        T_live = T[live]
        A = identity(int(live.sum()), format="csc") - T_live[:, live].tocsc()
        rhs = np.column_stack([T_live @ np.asarray(t, dtype=np.float64) for t in targets])
        out[live] = splu(A).solve(rhs)
    return out
