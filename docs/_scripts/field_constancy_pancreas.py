"""Draw the field-constancy figure of the docs on scVelo's pancreas data.

scVelo's dynamical and deterministic velocities and ConstVelo (Ductal to Beta),
30 PCs and 30 neighbours, each drawn with ``ve.pl.field_constancy``.  Writes
``field_constancy_pancreas.png`` next to the docs' other static files::

    python docs/_scripts/field_constancy_pancreas.py [raw_pancreas.h5ad]

Without a path, ``scvelo.datasets.pancreas()`` downloads the data.
"""

from __future__ import annotations

import pathlib
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scanpy as sc  # noqa: E402
import scvelo as scv  # noqa: E402

import veloeval as ve  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "_static"


def main(path=None):
    scv.settings.verbosity = 0
    adata = ad.read_h5ad(path) if path else scv.datasets.pancreas()
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    scv.pp.moments(adata, n_neighbors=None, n_pcs=None)

    runs = {}
    det = adata.copy()
    scv.tl.velocity(det, mode="deterministic")
    dyn = adata.copy()
    scv.tl.recover_dynamics(dyn, n_jobs=4)
    scv.tl.velocity(dyn, mode="dynamical")
    const = adata.copy()
    ve.constvelo(const, start="Ductal", end="Beta")
    runs = {"scVelo dynamical": dyn, "scVelo deterministic": det, "ConstVelo": const}

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"},
                             figsize=(14, 3.6))
    for ax, (name, run) in zip(axes, runs.items()):
        ve.set_velocity_space(run, "gene")
        ve.pl.field_constancy(run, label_key="clusters", ax=ax, title=name,
                              legend=ax is axes[1])
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "field_constancy_pancreas.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
