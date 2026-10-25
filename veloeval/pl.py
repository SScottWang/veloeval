"""Plots of metric results.  Needs matplotlib (``pip install veloeval[plot]``)."""

from __future__ import annotations

import numpy as np

from .access import get_embedding, get_labels, get_velocity
from .metrics import field_constancy as _field_constancy
from .metrics.direction import _edge_means, _edge_turning

__all__ = ["field_constancy", "cbturn", "cbturn_parts"]


def field_constancy(
    adata,
    *,
    vkey: str = "velocity",
    label_key: str | None = None,
    bins: int = 36,
    ax=None,
    title: str | None = None,
    legend: bool = True,
):
    r"""Where each cell's velocity points relative to the field's mean direction.

    A half-disc: a cell at angle :math:`\theta` has cosine
    :math:`\cos\theta` with the mean unit velocity, 0° (right) pointing the
    same way and 180° (left) the opposite.  Each wedge's area is proportional
    to its number of cells, stacked by cell type with *label_key*.  The arrow is
    :func:`~veloeval.metrics.field_constancy` :math:`R` on the scale of the
    dashed circle, :math:`R = 1`; it is also the mean of every cell's
    :math:`\cos\theta`.  The grey band is where directions drawn at random in
    as many features as the velocity has would fall (±2 sd).  With a thousand
    genes it is a few degrees either side of 90°, so cells in it point no more
    along the mean than noise would.

    Parameters
    ----------
    adata : anndata.AnnData
        Gene-space velocity, as for :func:`~veloeval.metrics.field_constancy`.
    vkey : str, default: ``"velocity"``
    label_key : str, optional
        ``obs`` column to stack the wedges by, coloured with
        ``uns['{label_key}_colors']`` when present.
    bins : int, default: 36
        Wedges over 0–180°.
    ax : matplotlib.projections.polar.PolarAxes, optional
        Drawn into if given, otherwise a new figure.
    title : str, optional
        Placed above, followed by the value of :math:`R`.
    legend : bool, default: True
        Legend at the side: the cell types and the random band.

    Returns
    -------
    matplotlib.projections.polar.PolarAxes

    Examples
    --------
    .. code-block:: python

        import veloeval as ve

        ve.pl.field_constancy(adata, label_key="clusters", title="scVelo")
    """
    import matplotlib.pyplot as plt

    res = _field_constancy(adata, vkey=vkey)
    if res.status != "ok":
        raise ValueError(f"field_constancy is {res.status}: {res.detail}")
    R = res.value
    n_features = get_velocity(adata, vkey).shape[1]
    scored = np.isfinite(res.per_cell)
    theta = np.arccos(np.clip(res.per_cell[scored], -1, 1))

    if label_key is None:
        groups = {"cells": np.ones(scored.sum(), dtype=bool)}
        colors = {"cells": "C0"}
    else:
        labels = get_labels(adata, label_key)[scored]
        names = _categories(adata, label_key, labels)
        groups = {n: labels == n for n in names}
        colors = _label_colors(adata, label_key)

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4.5, 2.8))
    ax.set_theta_zero_location("E")
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    edges = np.linspace(0, np.pi, bins + 1)
    width = edges[1] - edges[0]
    counts = np.array([np.histogram(theta[m], edges)[0] for m in groups.values()])
    total = counts.sum(axis=0).max()
    # Area of a wedge grows with radius squared, so stack in squared radius.
    radius = np.sqrt(np.cumsum(counts, axis=0) / total) * 0.9
    inner = np.vstack([np.zeros(bins), radius[:-1]])
    for (name, _), r0, r1 in zip(groups.items(), inner, radius):
        ax.bar(edges[:-1], r1 - r0, width=width, bottom=r0, align="edge",
               color=colors[name], edgecolor="white", linewidth=0.3, label=name)

    sd = 2 / np.sqrt(n_features)
    ax.fill_between(np.linspace(np.pi / 2 - sd, np.pi / 2 + sd, 20), 0, 1,
                    color="0.6", alpha=0.35, linewidth=0, zorder=0,
                    label="random directions (±2 sd)")
    ax.plot(np.linspace(0, np.pi, 100), np.ones(100), ls="--", color="k", lw=0.8)
    ax.annotate("", xy=(0, R), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2))

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180]))
    ax.grid(color="0.85", lw=0.5)
    head = f"{title}\n" if title else ""
    ax.set_title(f"{head}R = {R:.2f}", fontsize=10, pad=12)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.12, 0.35), frameon=False,
                  fontsize=7)
    return ax


def cbturn(
    adata,
    reference=None,
    *,
    label_key: str,
    cluster_edges,
    basis: str = "umap",
    xkey: str = "Ms",
    vkey: str = "velocity",
    vmax: float | None = None,
    ax=None,
    title: str | None = None,
    legend: bool = True,
):
    r"""CBTurn edge by edge, on the differentiation topology.

    Each cell type is a node at the median of its cells in ``obsm['X_{basis}']``,
    sized by its number of cells, and each of *cluster_edges* an arrow from
    source to target, as in scVelo's PAGA plot.  The arrow's colour is the
    edge's turning part (see :func:`cbturn_parts`): blue where the velocity on
    the edge turns with it, red where against it, grey where nothing on the
    edge sets it apart from the others.  The embedding only places the
    nodes: the scores are computed in gene or latent space, and an arrow's
    angle in the plot says nothing about the velocity.

    Parameters
    ----------
    adata, reference, label_key, cluster_edges, xkey, vkey
        As for :func:`~veloeval.metrics.cbturn`.
    basis : str, default: ``"umap"``
        Embedding that places the nodes.
    vmax : float, optional
        Turning part at full colour.  ``None`` uses the largest an edge
        could reach, :math:`\lVert \hat d_e - \bar d \rVert^2` -- velocity on
        every edge exactly along it -- so that colours compare across methods.
    ax : matplotlib.axes.Axes, optional
        Drawn into if given, otherwise a new figure.
    title : str, optional
        Placed above, followed by the value of CBTurn.
    legend : bool, default: True
        Cell types at the side.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    .. code-block:: python

        ve.pl.cbturn(adata, ref, label_key="clusters", cluster_edges=edges)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import FancyArrowPatch

    names, U, cells, D = _edge_turning(
        adata, reference, label_key, cluster_edges, xkey, vkey
    )
    glob, turn, value = _edge_parts(U, cells, D)
    emb = get_embedding(adata, basis)[:, :2]
    labels = get_labels(adata, label_key)
    colors = _label_colors(adata, label_key)
    if vmax is None:
        Dc = D - D.mean(axis=0)
        vmax = float((Dc * Dc).sum(axis=1).max())
    cmap = LinearSegmentedColormap.from_list(
        "turn", ["#67001f", "#d6604d", "#c8c8c8", "#4393c3", "#053061"]
    )
    norm = Normalize(-vmax, vmax)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(*emb.T, c=[colors.get(x, "0.8") for x in labels], s=4, lw=0,
               alpha=0.12, rasterized=True)
    nodes = sorted({n for e in names for n in e.split(" -> ")},
                   key=list(colors).index)
    where = {n: np.median(emb[labels == n], axis=0) for n in nodes}
    size = {n: 60 + 240 * np.sqrt((labels == n).mean()) for n in nodes}
    for name, t in zip(names, turn):
        src, tgt = name.split(" -> ")
        ax.add_patch(FancyArrowPatch(
            where[src], where[tgt], arrowstyle="-|>", mutation_scale=16,
            color=cmap(norm(t)), lw=2.6, zorder=2,
            shrinkA=np.sqrt(size[src]) / 2 + 1, shrinkB=np.sqrt(size[tgt]) / 2 + 1,
        ))
    for n in nodes:
        ax.scatter(*where[n], s=size[n], color=colors.get(n, "0.5"),
                   edgecolor="white", linewidth=1, zorder=3, label=n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = plt.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.5,
                       pad=0.02, location="left")
    bar.set_label("turning part on the edge", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    head = f"{title}\n" if title else ""
    ax.set_title(f"{head}CBTurn = {_fmt(value)}", fontsize=10)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                  fontsize=7, markerscale=0.5)
    return ax


def cbturn_parts(
    runs,
    reference=None,
    *,
    label_key: str,
    cluster_edges,
    xkey: str = "Ms",
    vkey: str = "velocity",
    ax=None,
    title: str | None = None,
):
    r"""Split a centroid CBDir into a global and a turning part.

    The agreement of each edge's mean boundary velocity with its direction,
    :math:`\langle \bar u_e, \hat d_e \rangle` -- a version of
    :func:`~veloeval.metrics.cbdir` aimed at the target cluster's mean --
    splits into a turning part :math:`\langle \bar u_e - \bar u,\;
    \hat d_e - \bar d \rangle`, what the edge's velocity shares with what sets
    the edge apart from the others, and a global part, the rest.  Averaged
    over edges the global part is :math:`\langle \bar u, \bar d \rangle`, one
    direction shared by every edge, and the turning part is
    :func:`~veloeval.metrics.cbturn` up to a factor of the dataset.  A
    constant field has no turning part on any edge.

    Given one AnnData, a bar per edge, the turning part in the target
    cluster's colour.  Given several by name, a bar per run with both parts
    averaged over edges and CBTurn at the end.  The global part is grey.

    Parameters
    ----------
    runs : anndata.AnnData or dict of str to anndata.AnnData
        One prepared AnnData, or one per method, drawn top to bottom.
    reference, label_key, cluster_edges, xkey, vkey
        As for :func:`~veloeval.metrics.cbturn`.
    ax : matplotlib.axes.Axes, optional
        Drawn into if given, otherwise a new figure.
    title : str, optional
        Placed above; for one AnnData, followed by the value of CBTurn.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    .. code-block:: python

        ve.pl.cbturn_parts(adata, ref, label_key="clusters", cluster_edges=edges)
        runs = {"scVelo": a, "ConstVelo": c}
        ve.pl.cbturn_parts(runs, ref, label_key="clusters", cluster_edges=edges)
    """
    import matplotlib.pyplot as plt

    single = not isinstance(runs, dict)
    rows = []
    for name, adata in ({"": runs} if single else runs).items():
        names, U, cells, D = _edge_turning(
            adata, reference, label_key, cluster_edges, xkey, vkey
        )
        glob, turn, value = _edge_parts(U, cells, D)
        if single:
            colors = _label_colors(adata, label_key)
            rows = [
                (e.replace(" -> ", " → "), g, t, colors.get(e.split(" -> ")[1], "C0"))
                for e, g, t in zip(names, glob, turn)
            ]
        else:
            rows.append((name, glob.mean(), turn.mean(), "C0", value))

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 0.6 + 0.4 * len(rows)))
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        glob, turn, color = row[1:4]
        ax.barh(yi, glob, color="0.75", height=0.6)
        ax.barh(yi, turn, left=glob, color=color, height=0.6)
        if not single:
            ax.text(max(glob, glob + turn, 0) + 0.01, yi, f"CBTurn {_fmt(row[4])}",
                    va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("grey: global part; colour: turning part", fontsize=8)
    ax.margins(x=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if single:
        head = f"{title}\n" if title else ""
        ax.set_title(f"{head}CBTurn = {_fmt(value)}", fontsize=10)
    elif title:
        ax.set_title(title, fontsize=10)
    return ax


def _edge_parts(U, cells, D):
    """Per edge the global and turning parts of its centroid CBDir, and CBTurn."""
    B = _edge_means(U, cells)
    Dc, Bc = D - D.mean(axis=0), B - B.mean(axis=0)
    turn = (Bc * Dc).sum(axis=1)
    glob = (B * D).sum(axis=1) - turn
    return glob, turn, float(turn.sum() / (Dc * Dc).sum())


def _fmt(x: float) -> str:
    # round() then + 0.0 turns -0.0 into 0.0, so rounding noise prints as 0.00.
    return f"{round(x, 2) + 0.0:.2f}"


def _label_colors(adata, label_key):
    """Colour per label: ``uns['{label_key}_colors']`` follows every category."""
    col = adata.obs[label_key]
    if hasattr(col, "cat"):
        names = list(col.cat.categories)
    else:
        names = sorted(set(get_labels(adata, label_key).tolist()))
    palette = adata.uns.get(f"{label_key}_colors")
    if palette is not None and len(palette) == len(names):
        return dict(zip(names, palette))
    return {n: f"C{i % 10}" for i, n in enumerate(names)}


def _categories(adata, label_key, labels):
    col = adata.obs[label_key]
    if hasattr(col, "cat"):
        present = set(labels.tolist())
        return [c for c in col.cat.categories if c in present]
    return sorted(set(labels.tolist()))
