"""Draw the ConstVelo figure of the docs on scVelo's pancreas data.

``ve.constvelo`` from Ductal to Beta, with 30 PCs and 30 neighbours, drawn in
PCA, as per-cell arrows on the dataset's UMAP and as streamlines.  Writes
``constvelo_pancreas.png`` next to the docs' other static files.  Needs scvelo
and matplotlib::

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

OUT = pathlib.Path(__file__).resolve().parents[1] / "_static"
LABEL = "clusters"
N_ARROWS = 400


def main(path=None):
    scv.settings.verbosity = 0
    raw = ad.read_h5ad(path) if path else scv.datasets.pancreas()
    scv.pp.filter_genes(raw, min_shared_counts=20)
    scv.pp.normalize_per_cell(raw)
    sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=2000, subset=True)

    ref = raw.copy()
    c = ve.constvelo(ref, start="Ductal", end="Beta")

    m = ad.AnnData(obs=ref.obs[[LABEL]].copy(), var=pd.DataFrame(index=ref.var_names))
    m.layers["velocity"] = ref.layers["velocity"]
    m.var["velocity_genes"] = True
    m.uns[f"{LABEL}_colors"] = ref.uns[f"{LABEL}_colors"]
    del ref.layers["velocity"]
    ve.prepare(m, reference=ref)

    draw(m, c @ ref.varm["PCs"][:, :2])


def draw(m, pca_arrow):
    pick = np.random.default_rng(0).choice(m.n_obs, N_ARROWS, replace=False)
    cats = m.obs[LABEL].cat.categories
    colors = dict(zip(cats, m.uns[f"{LABEL}_colors"]))
    point = m.obs[LABEL].map(colors).to_numpy()
    pcs, umap = m.obsm["X_pca"][:, :2], m.obsm["X_umap"]
    arrows = m.obsm["velocity_umap"]

    pca_scale = 0.05 * np.ptp(pcs, axis=0).max() / np.linalg.norm(pca_arrow)
    umap_scale = 0.04 * np.ptp(umap, axis=0).max() / np.median(
        np.linalg.norm(arrows, axis=1)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    ax.scatter(*pcs.T, c=point, s=3, lw=0)
    ax.quiver(*pcs[pick].T, *np.tile(pca_arrow * pca_scale, (N_ARROWS, 1)).T,
              angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.set(title="PCA", xlabel="PC1", ylabel="PC2")

    ax = axes[1]
    ax.scatter(*umap.T, c=point, s=3, lw=0)
    ax.quiver(*umap[pick].T, *(arrows[pick] * umap_scale).T,
              angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.set(title="UMAP, per-cell arrows", xlabel="UMAP1", ylabel="UMAP2")

    ax = axes[2]
    scv.pl.velocity_embedding_stream(
        m, basis="umap", color=LABEL, ax=ax, show=False, legend_loc="none",
        title="", size=10, alpha=0.5,
    )
    ax.set_title("UMAP, streamlines", fontsize=axes[0].title.get_fontsize())
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [plt.Line2D([], [], ls="", marker="o", color=colors[n], label=n)
               for n in cats]
    fig.legend(handles=handles, loc="lower center", ncol=len(cats), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "constvelo_pancreas.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
