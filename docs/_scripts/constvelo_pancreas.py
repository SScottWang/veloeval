"""Draw the ConstVelo figure of the docs on scVelo's pancreas data.

One velocity, ``ve.constvelo`` from Ductal to Beta on the k = 30 reference,
scored on references whose kNN differ (k = 15, 30, 50; 30 PCs each) and drawn
on the dataset's own UMAP, held fixed.  Writes ``constvelo_pancreas.png`` and
``constvelo_pancreas.csv`` next to the docs' other static files and prints the
checks.  Needs scvelo and matplotlib::

    python docs/_scripts/constvelo_pancreas.py [raw_pancreas.h5ad]

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
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scvelo as scv  # noqa: E402

import veloeval as ve  # noqa: E402
from veloeval import metrics as M  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "_static"
KS = (15, 30, 50)
N_PCS = 30
LABEL = "clusters"
EDGES = [
    ["Ductal", "Ngn3 low EP"],
    ["Ngn3 low EP", "Ngn3 high EP"],
    ["Ngn3 high EP", "Pre-endocrine"],
    ["Pre-endocrine", "Alpha"],
    ["Pre-endocrine", "Beta"],
    ["Pre-endocrine", "Delta"],
    ["Pre-endocrine", "Epsilon"],
]
SCALES = (0.1, 10.0, 0.125, 8.0)
N_ARROWS = 800


def preprocess(raw):
    adata = raw.copy()
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=N_PCS)
    return adata


def reference(base, k):
    ref = base.copy()
    sc.pp.neighbors(ref, n_neighbors=k, n_pcs=N_PCS)
    scv.pp.moments(ref, n_neighbors=None, n_pcs=None)
    return ref


def method(base, c, scale=1.0):
    m = ad.AnnData(obs=base.obs[[LABEL]].copy(), var=pd.DataFrame(index=base.var_names))
    m.layers["velocity"] = np.tile((scale * c).astype(np.float32), (m.n_obs, 1))
    m.var["velocity_genes"] = True
    m.uns[f"{LABEL}_colors"] = base.uns[f"{LABEL}_colors"]
    return m


def scores(m):
    kw = dict(label_key=LABEL, cluster_edges=EDGES)
    res = {
        "CBDir (UMAP)": M.cbdir(m, **kw),
        "CBDir (gene)": M.cbdir(m, **kw, basis=None),
        "CBVCoh (UMAP)": M.cbvcoh(m, **kw),
        "ICVCoh (UMAP)": M.icvcoh(m, label_key=LABEL),
    }
    for name, r in res.items():
        if r.status != "ok":
            print(f"  {name}: {r.status} ({r.detail})")
    return {name: r.value for name, r in res.items()}


def main(path=None):
    scv.settings.verbosity = 0
    raw = ad.read_h5ad(path) if path else scv.datasets.pancreas()
    base = preprocess(raw)
    refs = {k: reference(base, k) for k in KS}

    c = ve.constvelo(refs[30].copy(), start="Ductal", end="Beta")
    runs, rows = {}, []
    for k in KS:
        m = method(base, c)
        ve.prepare(m, reference=refs[k])
        runs[k] = m
        rows.append({"k": k, "n_pcs": N_PCS, **scores(m)})
    table = pd.DataFrame(rows)

    print("check 1, rescaled V (largest change of any score):")
    for s in SCALES:
        diffs = []
        for k in KS:
            m = method(base, c, s)
            ve.prepare(m, reference=refs[k])
            got = scores(m)
            ref_row = table.loc[table.k == k].iloc[0]
            diffs += [abs(got[n] - ref_row[n]) for n in got]
        print(f"  x{s:g}: {max(diffs):.3g}")

    loadings = base.varm["PCs"][:, :2]
    pca_arrows = runs[30].layers["velocity"] @ loadings
    print("check 2, largest difference between PCA arrows:",
          float(np.abs(pca_arrows - pca_arrows[0]).max()))

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "constvelo_pancreas.csv", index=False)
    print(table.to_string(index=False))
    draw(base, runs, pca_arrows[0], table)


def draw(base, runs, pca_arrow, table):
    rng = np.random.default_rng(0)
    pick = rng.choice(base.n_obs, N_ARROWS, replace=False)
    cats = base.obs[LABEL].cat.categories
    colors = dict(zip(cats, base.uns[f"{LABEL}_colors"]))
    point = base.obs[LABEL].map(colors).to_numpy()
    pcs, umap = base.obsm["X_pca"][:, :2], base.obsm["X_umap"]

    span = np.ptp(umap, axis=0).max()
    lengths = np.linalg.norm(runs[30].obsm["velocity_umap"], axis=1)
    umap_scale = 0.04 * span / np.median(lengths)
    pca_scale = 0.06 * np.ptp(pcs, axis=0).max() / np.linalg.norm(pca_arrow)

    fig = plt.figure(figsize=(12, 16))
    grid = fig.add_gridspec(4, 12, height_ratios=[1, 1, 1, 0.7], hspace=0.3,
                            wspace=2.5, top=0.9, bottom=0.07)
    axes = np.array([[fig.add_subplot(grid[r, 4 * j:4 * j + 4]) for j in range(3)]
                     for r in range(3)])
    for j, k in enumerate(KS):
        m = runs[k]
        ax = axes[0, j]
        ax.scatter(*pcs.T, c=point, s=2, lw=0)
        ax.quiver(*pcs[pick].T, *np.tile(pca_arrow * pca_scale, (N_ARROWS, 1)).T,
                  angles="xy", scale_units="xy", scale=1, width=0.003)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("PC1")
        if j == 0:
            ax.set_ylabel("reference PCA\nPC2")

        ax = axes[1, j]
        ax.scatter(*umap.T, c=point, s=2, lw=0)
        ax.quiver(*umap[pick].T, *(m.obsm["velocity_umap"][pick] * umap_scale).T,
                  angles="xy", scale_units="xy", scale=1, width=0.003)
        if j == 0:
            ax.set_ylabel("dataset UMAP (fixed)\nper-cell arrows")

        ax = axes[2, j]
        scv.pl.velocity_embedding_stream(
            m, basis="umap", color=LABEL, ax=ax, show=False, legend_loc="none",
            title="", size=8, alpha=0.4,
        )
        if j == 0:
            ax.set_ylabel("dataset UMAP (fixed)\nstreamlines")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], ls="", marker="o", color=colors[n], label=n)
               for n in cats]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.915),
               ncol=len(cats), frameon=False)

    names = [n for n in table.columns if n not in ("k", "n_pcs")]
    panels = [fig.add_subplot(grid[3, 3 * j:3 * j + 3]) for j in range(len(names))]
    for ax, name in zip(panels, names):
        ax.plot(range(len(KS)), table[name], "o-", color="k")
        for x, y in enumerate(table[name]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xticks(range(len(KS)), [f"k = {k}" for k in KS])
        ax.set_title(name)
        ax.margins(x=0.2, y=0.3)
    fig.text(
        0.5, 0.02,
        f"Arrows on the same {N_ARROWS} random cells in every panel; one scale "
        "factor per row.  PCA arrow: V @ loadings[:, :2] (PCA on log1p X, "
        "centred, genes not scaled).  UMAP arrows: scVelo velocity_graph -> "
        "velocity_embedding, all genes as velocity genes.",
        ha="center", va="bottom", fontsize=8, wrap=True,
    )
    fig.suptitle(
        "ConstVelo on pancreas (idealised): every cell has velocity "
        "c = mean(Ms[Beta]) - mean(Ms[Ductal]), unit norm, fixed across columns;\n"
        f"columns change only the kNN graph ({N_PCS} PCs), the UMAP layout is "
        "the dataset's own",
        y=0.965,
    )
    fig.savefig(OUT / "constvelo_pancreas.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
