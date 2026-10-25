"""Plots of metric results.  Needs matplotlib (``pip install veloeval[plot]``)."""

from __future__ import annotations

import numpy as np

from .access import get_labels, get_velocity
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
    :math:`\cos\theta`.  The grey band is where unit vectors drawn at random
    in the same number of features fall (±2 sd): in gene space nearly all at
    90°, so cells there point nowhere in particular.

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
        Cell-type legend, with *label_key*.

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
                    color="0.6", alpha=0.35, linewidth=0, zorder=0)
    ax.plot(np.linspace(0, np.pi, 100), np.ones(100), ls="--", color="k", lw=0.8)
    ax.annotate("", xy=(0, R), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2))

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180]))
    ax.grid(color="0.85", lw=0.5)
    ax.text(np.pi / 2 + sd + 0.03, 0.55, f"random,\n{n_features}\nfeatures",
            ha="right", va="center", fontsize=6.5, color="0.4")
    head = f"{title}\n" if title else ""
    ax.set_title(f"{head}R = {R:.2f}", fontsize=10, pad=12)
    if label_key is not None and legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False,
                  fontsize=7, ncol=4)
    return ax


def cbturn(
    adata,
    reference=None,
    *,
    label_key: str,
    cluster_edges,
    xkey: str = "Ms",
    vkey: str = "velocity",
    velocity_scale: float | None = None,
    ax=None,
    title: str | None = None,
):
    r"""Each cluster edge's own direction against the velocity on it.

    What :func:`~veloeval.metrics.cbturn` regresses, in the plane that holds
    most of the spread of the edge directions.  For every edge, a grey arrow
    for :math:`\hat d_e - \bar d`, its direction minus the mean over edges, and
    a coloured one, in the target cluster's colour, for
    :math:`\bar u_e - \bar u`, the mean unit velocity of its boundary cells
    minus the mean over edges.  The coloured arrows are lengthened by
    *velocity_scale*, printed in the title: at that scale, matching arrows
    are a slope of 1 over the scale.  A constant field leaves no coloured
    arrow, a reversed one turns them all round.  The fraction of the spread of the
    edge directions the plane holds is printed below; the velocity can also
    have parts outside it.

    Parameters
    ----------
    adata, reference, label_key, cluster_edges, xkey, vkey
        As for :func:`~veloeval.metrics.cbturn`.
    velocity_scale : float, optional
        Factor on the coloured arrows.  ``None`` picks the whole number that
        brings their root-mean-square length nearest the grey arrows'; pass
        one value to compare methods.
    ax : matplotlib.axes.Axes, optional
        Drawn into if given, otherwise a new figure.
    title : str, optional
        Placed above, followed by the value of CBTurn.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    .. code-block:: python

        ve.pl.cbturn(adata, ref, label_key="clusters", cluster_edges=edges)
    """
    import matplotlib.pyplot as plt

    names, U, cells, D = _edge_turning(
        adata, reference, label_key, cluster_edges, xkey, vkey
    )
    B = _edge_means(U, cells)
    Dc, Bc = D - D.mean(axis=0), B - B.mean(axis=0)
    value = float((Bc * Dc).sum() / (Dc * Dc).sum())
    _, sv, Vt = np.linalg.svd(Dc, full_matrices=False)
    plane = Vt[:2].T
    held = float((sv[:2] ** 2).sum() / (sv**2).sum())
    d2, b2 = Dc @ plane, Bc @ plane

    longest_d = np.linalg.norm(d2, axis=1).max()
    rms_d, rms_b = np.sqrt((d2**2).sum()), np.sqrt((b2**2).sum())
    if velocity_scale is None:
        velocity_scale = (
            max(1.0, float(np.round(rms_d / rms_b))) if rms_b > 1e-9 * rms_d else 1.0
        )
    b2 = b2 * velocity_scale

    colors = _label_colors(adata, label_key)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    lim = 1.35 * max(longest_d, np.linalg.norm(b2, axis=1).max())
    for name, d, b in zip(names, d2, b2):
        color = colors.get(name.split(" -> ")[1], "C0")
        ax.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(
            arrowstyle="-|>", color="0.8", lw=4, shrinkA=0, shrinkB=0))
        if np.linalg.norm(b) > 1e-6 * longest_d:
            ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.8, shrinkA=0, shrinkB=0))
        ax.text(*(d * 1.08), name.replace(" -> ", " → "), fontsize=6.5,
                ha="left" if d[0] >= 0 else "right", va="center", color="0.3")
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.axvline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    head = f"{title}\n" if title else ""
    ax.set_title(f"{head}CBTurn = {_fmt(value)}, velocity × {velocity_scale:g}",
                 fontsize=10)
    ax.text(0.5, -0.02, "grey: edge direction; colour: velocity; both minus "
            f"their mean over edges\nplane holds {held:.0%} of the edge "
            "directions' spread", transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color="0.4")
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
):
    r"""Split a centroid CBDir into a global and a turning part, per run.

    Averaged over edges, the cosine-like score
    :math:`\langle \bar u_e, \hat d_e \rangle` -- a version of
    :func:`~veloeval.metrics.cbdir` aimed at the target cluster's mean -- is
    exactly

    .. math::

        \langle \bar u, \bar d \rangle \;+\;
        \tfrac{1}{E} \sum_e \langle \bar u_e - \bar u,\; \hat d_e - \bar d \rangle,

    one direction shared by every edge plus what changes from edge to edge.
    Each run is a bar: the global part in grey, the turning part in colour,
    stacked; :func:`~veloeval.metrics.cbturn` is printed at the end.  A
    constant field has no coloured part.

    Parameters
    ----------
    runs : dict of str to anndata.AnnData
        One prepared AnnData per method, by name, drawn top to bottom.
    reference, label_key, cluster_edges, xkey, vkey
        As for :func:`~veloeval.metrics.cbturn`.
    ax : matplotlib.axes.Axes, optional
        Drawn into if given, otherwise a new figure.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    .. code-block:: python

        runs = {"scVelo": a, "ConstVelo": c}
        ve.pl.cbturn_parts(runs, ref, label_key="clusters", cluster_edges=edges)
    """
    import matplotlib.pyplot as plt

    rows = []
    for name, adata in runs.items():
        _, U, cells, D = _edge_turning(
            adata, reference, label_key, cluster_edges, xkey, vkey
        )
        B = _edge_means(U, cells)
        Dc, Bc = D - D.mean(axis=0), B - B.mean(axis=0)
        glob = float(B.mean(axis=0) @ D.mean(axis=0))
        turn = float((Bc * Dc).sum(axis=1).mean())
        rows.append((name, glob, turn, float((Bc * Dc).sum() / (Dc * Dc).sum())))

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 0.5 + 0.45 * len(rows)))
    y = np.arange(len(rows))[::-1]
    for yi, (_, glob, turn, value) in zip(y, rows):
        ax.barh(yi, glob, color="0.75", height=0.6,
                label="global" if yi == y[0] else None)
        ax.barh(yi, turn, left=glob, color="C0", height=0.6,
                label="turning" if yi == y[0] else None)
        end = max(glob, glob + turn, 0)
        ax.text(end + 0.01, yi, f"CBTurn {_fmt(value)}", va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("centroid CBDir = global + turning")
    ax.margins(x=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


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
