"""Draw the CBTurn figures of the docs on scVelo's pancreas data.

scVelo's dynamical and deterministic velocities, ConstVelo (Ductal to Beta) and
the reversed dynamical velocity, each put on one reference (2000 HVGs, 30 PCs,
30 neighbours) with ``ve.prepare(reference=...)``.  Writes
``cbturn_pancreas_topology.png``, ``cbturn_pancreas_edges.png`` and
``cbturn_pancreas_parts.png`` next to the docs' other static files::

    python docs/_scripts/cbturn_pancreas.py [raw_pancreas.h5ad]

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
EDGES = [
    ["Ductal", "Ngn3 low EP"],
    ["Ngn3 low EP", "Ngn3 high EP"],
    ["Ngn3 high EP", "Pre-endocrine"],
    ["Pre-endocrine", "Alpha"],
    ["Pre-endocrine", "Beta"],
    ["Pre-endocrine", "Delta"],
    ["Pre-endocrine", "Epsilon"],
]


def method(ref, V, genes):
    m = ad.AnnData(obs=ref.obs[[LABEL]].copy(), var=pd.DataFrame(index=ref.var_names))
    m.layers["velocity"] = np.asarray(V, dtype=np.float32)
    m.var["velocity_genes"] = genes
    m.uns[f"{LABEL}_colors"] = ref.uns[f"{LABEL}_colors"]
    ve.prepare(m, reference=ref)
    return m


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

    det = ref.copy()
    scv.tl.velocity(det, mode="deterministic")
    dyn = ref.copy()
    scv.tl.recover_dynamics(dyn, n_jobs=4)
    scv.tl.velocity(dyn, mode="dynamical")
    const = ref.copy()
    ve.constvelo(const, start="Ductal", end="Beta")

    dyn_genes = dyn.var["velocity_genes"].to_numpy(bool)
    runs = {
        "scVelo dynamical": method(ref, dyn.layers["velocity"], dyn_genes),
        "scVelo deterministic": method(
            ref, det.layers["velocity"], det.var["velocity_genes"].to_numpy(bool)
        ),
        "ConstVelo": method(ref, const.layers["velocity"], np.ones(ref.n_vars, bool)),
        "reversed dynamical": method(ref, -dyn.layers["velocity"], dyn_genes),
    }
    kw = dict(label_key=LABEL, cluster_edges=EDGES)
    OUT.mkdir(exist_ok=True)

    names = ["scVelo dynamical", "ConstVelo", "reversed dynamical"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, name in zip(axes, names):
        ve.pl.cbturn(runs[name], ref, ax=ax, title=name, legend=ax is axes[-1], **kw)
    fig.savefig(OUT / "cbturn_pancreas_topology.png", dpi=100, bbox_inches="tight")

    ax = ve.pl.cbturn_parts(runs["scVelo dynamical"], ref, title="scVelo dynamical",
                            **kw)
    ax.figure.savefig(OUT / "cbturn_pancreas_edges.png", dpi=110, bbox_inches="tight")

    ax = ve.pl.cbturn_parts(runs, ref, **kw)
    ax.figure.savefig(OUT / "cbturn_pancreas_parts.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
