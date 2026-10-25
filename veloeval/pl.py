"""Plots of metric results.  Needs matplotlib (``pip install veloeval[plot]``)."""

from __future__ import annotations

import numpy as np

from .access import get_labels, get_velocity
from .metrics import field_constancy as _field_constancy

__all__ = ["field_constancy"]


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
        palette = adata.uns.get(f"{label_key}_colors")
        if palette is not None and len(palette) == len(names):
            colors = dict(zip(names, palette))
        else:
            colors = {n: f"C{i % 10}" for i, n in enumerate(names)}

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


def _categories(adata, label_key, labels):
    col = adata.obs[label_key]
    if hasattr(col, "cat"):
        present = set(labels.tolist())
        return [c for c in col.cat.categories if c in present]
    return sorted(set(labels.tolist()))
