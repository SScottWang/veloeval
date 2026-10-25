"""Draw the field-constancy figure of the docs on scVelo's pancreas data.

scVelo's dynamical velocity, 30 PCs and 30 neighbours, drawn with
``ve.pl.field_constancy``.  Writes
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

    scv.tl.recover_dynamics(adata, n_jobs=4)
    scv.tl.velocity(adata, mode="dynamical")
    ve.set_velocity_space(adata, "gene")
    ax = ve.pl.field_constancy(adata, label_key="clusters", title="scVelo dynamical")
    fig = ax.figure
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "field_constancy_pancreas.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
