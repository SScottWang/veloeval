"""Negative controls: does the method stay quiet where nothing is happening?

On a population with no ongoing differentiation, a trustworthy method should
produce a diffuse, low-confidence field.  A confident one there is a
hallucination -- and the conventional accuracy metrics cannot see it, because
they only ever ask "is the arrow pointing the right way" on data where there is
a right way.

All four read scVelo's velocity graph (``scvelo.tl.velocity_graph``), which is
not computed for you.
"""

from __future__ import annotations

import numpy as np

from ..access import get_labels, get_velocity_graph
from ..result import MissingInput, NotApplicable, metric

__all__ = ["sts", "sts_abs", "ees", "nte"]

_REFERENCES = ("all", "groups")


def _confidence(adata, vkey):
    """Each cell's best cosine to a neighbour displacement, scVelo's ``confidence``."""
    graph = get_velocity_graph(adata, vkey)
    return graph.max(axis=1).toarray().ravel().astype(np.float64)


def _group_mask(adata, label_key, groups):
    """Cells whose ``obs[label_key]`` is in *groups*; every cell when *groups* is None."""
    if groups is None:
        return np.ones(adata.n_obs, dtype=bool)
    if label_key is None:
        raise ValueError("groups needs label_key")
    wanted = [groups] if isinstance(groups, str) else list(groups)
    labels = get_labels(adata, label_key).astype(str)
    missing = sorted(set(wanted) - set(labels))
    if missing:
        raise ValueError(f"groups not found in obs['{label_key}']: {missing}")
    return np.isin(labels, wanted)


def _on_groups(per_cell, mask):
    out = np.full(per_cell.shape, np.nan)
    out[mask] = per_cell[mask]
    if np.all(np.isnan(out)):
        raise NotApplicable("no scored cell has a value")
    return float(np.nanmean(out)), out


def _transition_rows(adata, vkey, scale):
    r"""Rows of ``scvelo.tl.transition_matrix(self_transitions=False)``.

    Row :math:`c` is proportional to :math:`\exp(\text{scale}\cos_{cj})` over
    every neighbour in the graph, negative cosines included.  Without
    ``{vkey}_graph_neg`` only the positive cosines count, as
    :math:`\exp(\text{scale}\cos_{cj}) - 1`, which is also what scVelo does.
    """
    T = get_velocity_graph(adata, vkey).astype(np.float64)
    T.data = np.expm1(scale * T.data)
    try:
        neg = get_velocity_graph(adata, vkey, negative=True).astype(np.float64)
    except MissingInput:
        pass
    else:
        neg.data = np.expm1(scale * neg.data)
        T = (T + neg).tocsr()
        T.data += 1
    for i in range(T.shape[0]):
        p = T.data[T.indptr[i] : T.indptr[i + 1]]
        p = p[p > 0]
        yield p / p.sum() if p.size else p


@metric
def sts(
    adata,
    *,
    vkey: str = "velocity",
    label_key: str | None = None,
    groups: str | list[str] | None = None,
    reference: str = "all",
):
    r"""Self-transition score (STS).

    scVelo's self-transition probability (Bergen et al., 2020), averaged over
    cells; equation (5) of the Genome Biology benchmark (2026):

    .. math::

        \hat\pi_i = \max_{j \in \mathcal N(i)}
            \cos(\mathbf x_j - \mathbf x_i,\ \mathbf v_i), \qquad
        \mathrm{STS} = \frac{1}{|G|} \sum_{i \in G}
            \operatorname{clip}(Q_{0.98}(\hat\pi) - \hat\pi_i,\ 0,\ 1)

    where :math:`G` is the scored cells.

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry scVelo's velocity graph ``{vkey}_graph``.
    vkey : str, default: "velocity"
        Velocity key prefix.
    label_key : str, optional
        Column in ``adata.obs`` holding cluster labels.  Required with *groups*.
    groups : str or list of str, optional
        Clusters to score -- the negative-control population.  ``None`` scores
        every cell.  A name absent from ``obs[label_key]`` fails.
    reference : {"all", "groups"}, default: "all"
        Cells the 98th percentile :math:`Q_{0.98}` is taken over.  ``"all"``
        keeps every cell's value identical to scVelo's; ``"groups"`` takes it
        over the scored cells only.

    Returns
    -------
    MetricResult
        ``per_cell`` holds each scored cell's self-transition probability and
        ``nan`` elsewhere.  With ``reference="all"`` the values equal scVelo's
        ``obs['{vkey}_self_transition']``.

    Notes
    -----
    Interpret on a **negative control** -- a population with no ongoing
    differentiation, where a trustworthy field points at no neighbour.

    The reference is taken from the run itself, so STS is unchanged when
    every cell's confidence shifts by the same amount: a field that is
    confident everywhere can score like one that is quiet everywhere.  Report
    :func:`sts_abs` next to it.  With ``reference="all"`` on a mixed dataset
    the line is set by the differentiating cells, which raises the score of a
    quiet subset; with ``"groups"`` it is set by the subset itself.

    :math:`\mathbf x` is whatever space the graph was built in (expression for
    gene-space methods, the method's latent representation otherwise), and
    cosines run higher in fewer dimensions.
    """
    if reference not in _REFERENCES:
        raise ValueError(f"reference must be one of {_REFERENCES}, got {reference!r}")
    mask = _group_mask(adata, label_key, groups)
    conf = _confidence(adata, vkey)
    line = np.percentile(conf[mask] if reference == "groups" else conf, 98)
    return _on_groups(np.clip(line - conf, 0, 1), mask)


@metric
def sts_abs(
    adata,
    *,
    vkey: str = "velocity",
    label_key: str | None = None,
    groups: str | list[str] | None = None,
):
    r"""Absolute self-transition score.

    :func:`sts` with the reference fixed at 1 instead of a 98th percentile,
    :math:`\frac{1}{|G|}\sum_{i \in G} (1 - \hat\pi_i)`, so a field that is
    confident everywhere scores low.

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry scVelo's velocity graph ``{vkey}_graph``.
    vkey : str, default: "velocity"
        Velocity key prefix.
    label_key : str, optional
        Column in ``adata.obs`` holding cluster labels.  Required with *groups*.
    groups : str or list of str, optional
        Clusters to score.  ``None`` scores every cell.

    Returns
    -------
    MetricResult
        ``per_cell`` holds :math:`1 - \hat\pi_i` for scored cells and ``nan``
        elsewhere.

    Notes
    -----
    Same space caveat as :func:`sts`: compare runs whose graphs live in spaces
    of similar dimension.
    """
    mask = _group_mask(adata, label_key, groups)
    return _on_groups(1.0 - _confidence(adata, vkey), mask)


@metric
def ees(
    adata,
    *,
    vkey: str = "velocity",
    scale: float = 30.0,
    mass: float = 0.95,
    label_key: str | None = None,
    groups: str | list[str] | None = None,
):
    r"""Effective entropy score (EES).

    The effective number of places each cell's transitions go, averaged over
    cells; equation (6) of the Genome Biology benchmark (2026):

    .. math::

        \mathrm{EES} = \frac{1}{|G|} \sum_{c \in G}
            \exp\Bigl(-\sum_{i=1}^{m_c} \tilde p_{c,i} \log \tilde p_{c,i}\Bigr)

    where :math:`m_c` is the fewest largest transition probabilities of cell
    :math:`c` whose sum reaches *mass*, :math:`\tilde p_{c,i}` those
    probabilities renormalised, and :math:`G` the scored cells.  The
    exponentiated entropy is the Hill number of order 1.

    Higher is better; range ``[1, max m_c]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry scVelo's velocity graph ``{vkey}_graph``, and ideally
        ``{vkey}_graph_neg``.
    vkey : str, default: "velocity"
        Velocity key prefix.
    scale : float, default: 30.0
        Inverse temperature of the transition matrix: row :math:`c` is
        proportional to :math:`\exp(\text{scale}\cos_{cj})`, as in
        ``scvelo.tl.transition_matrix(scale=scale, self_transitions=False)``.
        The benchmark's value.
    mass : float, default: 0.95
        Probability mass the retained transitions must reach.
    label_key : str, optional
        Column in ``adata.obs`` holding cluster labels.  Required with *groups*.
    groups : str or list of str, optional
        Clusters to score -- the negative-control population.  ``None`` scores
        every cell.

    Returns
    -------
    MetricResult
        ``per_cell`` holds each scored cell's effective number of transitions
        and ``nan`` elsewhere.

    Notes
    -----
    Interpret on a **negative control**, where a trustworthy field spreads a
    cell's transitions over many neighbours; 1 means it commits to one.

    The value moves with *scale* and with how many neighbours the graph gives
    each cell, so compare methods on one dataset with one graph, never raw
    values across datasets -- the benchmark ranks methods within each dataset
    for this reason.  Like :func:`sts`, cosines run higher, and EES lower, in
    fewer dimensions.
    """
    if not 0 < mass <= 1:
        raise ValueError(f"mass must be in (0, 1], got {mass}")
    mask = _group_mask(adata, label_key, groups)

    per_cell = np.full(adata.n_obs, np.nan)
    for c, p in enumerate(_transition_rows(adata, vkey, scale)):
        if not mask[c] or p.size == 0:
            continue
        p = np.sort(p)[::-1]
        m = np.searchsorted(np.cumsum(p), mass, side="left") + 1
        q = p[:m] / p[:m].sum()
        per_cell[c] = np.exp(-np.sum(q * np.log(q)))
    return _on_groups(per_cell, mask)


@metric
def nte(
    adata,
    *,
    vkey: str = "velocity",
    scale: float = 30.0,
    label_key: str | None = None,
    groups: str | list[str] | None = None,
):
    r"""Normalised transition entropy (NTE).

    Shannon entropy of each cell's transition distribution over its
    neighbours, divided by the entropy of a uniform one,

    .. math::

        \mathrm{NTE} = \frac{1}{|G|} \sum_{c \in G}
            \frac{-\sum_j P_{cj} \log P_{cj}}{\log k_c},

    where :math:`k_c` is the number of neighbours of cell :math:`c`.  The
    Genome Biology benchmark's code computes it next to :func:`ees` but its
    paper does not report it.

    Higher is better; range ``[0, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry scVelo's velocity graph ``{vkey}_graph``, and ideally
        ``{vkey}_graph_neg``.
    vkey : str, default: "velocity"
        Velocity key prefix.
    scale : float, default: 30.0
        Inverse temperature of the transition matrix; see :func:`ees`.
    label_key : str, optional
        Column in ``adata.obs`` holding cluster labels.  Required with *groups*.
    groups : str or list of str, optional
        Clusters to score.  ``None`` scores every cell.

    Returns
    -------
    MetricResult
        ``per_cell`` holds each scored cell's normalised entropy and ``nan``
        elsewhere, including cells with fewer than two neighbours.

    Notes
    -----
    Unlike :func:`ees`, the value does not grow with the number of
    neighbours, but it still moves with *scale*.
    """
    mask = _group_mask(adata, label_key, groups)

    per_cell = np.full(adata.n_obs, np.nan)
    for c, p in enumerate(_transition_rows(adata, vkey, scale)):
        if not mask[c] or p.size < 2:
            continue
        per_cell[c] = -np.sum(p * np.log(p)) / np.log(p.size)
    return _on_groups(per_cell, mask)
