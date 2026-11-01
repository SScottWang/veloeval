"""Draw ConstVelo's time against scVelo's latent time on pancreas.

2000 HVGs, 30 PCs, 30 neighbours; ``ve.constvelo`` from Ductal to Beta and
scVelo's dynamical model with ``scvelo.tl.latent_time``.  Writes
``constvelo_time_pancreas.png`` next to the docs' other static files and prints
the CTO of both times::

    python docs/_scripts/constvelo_time_pancreas.py [raw_pancreas.h5ad]

Without a path, ``scvelo.datasets.pancreas()`` downloads the data.
"""

from __future__ import annotations

import pathlib
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import scvelo as scv  # noqa: E402

import veloeval as ve  # noqa: E402
from veloeval import metrics as M  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "_static"
LABEL = "clusters"
ORDER = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine",
         "Alpha", "Beta", "Delta", "Epsilon"]
EDGES = [[ORDER[i], ORDER[i + 1]] for i in range(3)] + [
    ["Pre-endocrine", t] for t in ORDER[4:]
]


def tidy(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(path=None):
    scv.settings.verbosity = 0
    ref = ad.read_h5ad(path) if path else scv.datasets.pancreas()
    scv.pp.filter_genes(ref, min_shared_counts=20)
    scv.pp.normalize_per_cell(ref)
    sc.pp.log1p(ref)
    sc.pp.highly_variable_genes(ref, n_top_genes=2000, subset=True)
    sc.pp.pca(ref, n_comps=30)
    sc.pp.neighbors(ref, n_neighbors=30, n_pcs=30)
    scv.pp.moments(ref, n_neighbors=None, n_pcs=None)

    dyn = ref.copy()
    scv.tl.recover_dynamics(dyn, n_jobs=4)
    scv.tl.velocity(dyn, mode="dynamical")
    scv.tl.velocity_graph(dyn, n_jobs=4)
    scv.tl.latent_time(dyn)
    ref.obs["latent_time"] = dyn.obs["latent_time"]
    c = ve.constvelo(ref, start="Ductal", end="Beta")

    kw = dict(label_key=LABEL, cluster_edges=EDGES)
    cto = {k: M.cto(ref, **kw, time_key=k).value
           for k in ("constvelo_time", "latent_time")}
    print({k: round(v, 3) for k, v in cto.items()})

    lab = ref.obs[LABEL].to_numpy()
    palette = dict(zip(ref.obs[LABEL].cat.categories, ref.uns[f"{LABEL}_colors"]))
    cells = [palette[x] for x in lab]

    X = np.asarray(ref.layers["Ms"], dtype=np.float64)
    Wt = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:2]
    Z = (X - X.mean(0)) @ Wt.T
    a = Wt @ c
    t = ref.obs["constvelo_time"].to_numpy()
    start = Z[lab == "Ductal"].mean(0)
    span = t[lab == "Beta"].mean() - t[lab == "Ductal"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(wspace=0.15))
    ax = axes[0]
    ax.scatter(*Z.T, c=cells, s=3, lw=0, alpha=0.6, rasterized=True)
    ax.annotate("", xy=start + a * span, xytext=start,
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="k"))
    u = a / np.linalg.norm(a)
    n = np.array([-u[1], u[0]])
    t2, zn = Z @ a, Z @ n
    for q in np.quantile(t2, [0.1, 0.3, 0.5, 0.7, 0.9]):
        near = np.abs(t2 - q) < 0.03 * np.ptp(t2)
        lo, hi = zn[near].min(), zn[near].max()
        p0 = u * q / np.linalg.norm(a)
        ax.plot(*np.array([p0 + n * lo, p0 + n * hi]).T, color="0.4", lw=0.8, ls=":")
    tidy(ax, "time = distance travelled along c\n(PCA of Ms; dotted: equal time)",
         "PC1 of Ms", "PC2 of Ms")

    P = ref.obsm["X_pca"][:, :2]
    for ax, key, name in ((axes[1], "constvelo_time", "ConstVelo time"),
                          (axes[2], "latent_time", "scVelo latent_time")):
        v = ref.obs[key].to_numpy()
        rank = np.argsort(np.argsort(v)) / (len(v) - 1)
        o = np.argsort(rank)
        dots = ax.scatter(*P[o].T, c=rank[o], cmap="viridis", s=3, lw=0,
                          rasterized=True)
        tidy(ax, f"{name}, as a rank\nCTO = {cto[key]:.3f}", "PC1", "PC2")
    bar = fig.colorbar(dots, ax=axes[1:], shrink=0.7, pad=0.01)
    bar.set_label("rank of the time", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=palette[k], label=k)
               for k in ORDER]
    axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=7)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "constvelo_time_pancreas.png", dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
