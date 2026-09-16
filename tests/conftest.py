"""Synthetic fixtures whose correct metric values are known analytically.

Every metric here is tested on a field whose answer we can write down: a
perfect field must score 1, its reverse -1, and an unstructured one ~0.  A
metric that passes these has at least the right sign convention and
normalisation, which is the class of bug that silently flips a benchmark's
conclusions.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def knn_indices(X: np.ndarray, k: int = 6) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(X))
    return NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X, return_distance=False)


def make_adata(X_emb, V_emb, labels=None, V_gene=None, obs=None, k=6):
    """Assemble a minimal AnnData carrying everything the metrics read."""
    n = len(X_emb)
    if V_gene is None:
        V_gene = np.asarray(V_emb, dtype=np.float64)

    adata = ad.AnnData(
        X=np.asarray(V_gene, dtype=np.float32),
        obs=pd.DataFrame(obs or {}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(np.shape(V_gene)[1])]),
    )
    adata.layers["velocity"] = np.asarray(V_gene, dtype=np.float64)
    adata.obsm["X_umap"] = np.asarray(X_emb, dtype=np.float64)
    adata.obsm["velocity_umap"] = np.asarray(V_emb, dtype=np.float64)
    adata.uns["neighbors"] = {"indices": knn_indices(np.asarray(X_emb), k)}
    if labels is not None:
        adata.obs["clusters"] = pd.Categorical(labels)
    return adata


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def linear():
    """Two clusters on a line: A at x<0, B at x>0.  Truth points +x."""
    n = 60
    x = np.linspace(-1, 1, n)
    X = np.column_stack([x, np.zeros(n)])
    labels = np.where(x < 0, "A", "B")
    V = np.tile(np.array([1.0, 0.0]), (n, 1))
    return make_adata(X, V, labels=labels)


@pytest.fixture
def edges():
    return [["A", "B"]]


@pytest.fixture
def gene_space(rng):
    """40 cells x 20 genes of gene-space velocity.

    The ground-truth metrics compare gene by gene and refuse to run on fewer
    than 10 shared genes, so they need a fixture wider than the 2-D ones.
    """
    V = rng.normal(size=(40, 20))
    return make_adata(np.zeros((40, 2)), np.zeros((40, 2)), V_gene=V)


@pytest.fixture
def cycle():
    """Cells on a circle, phase = angle, velocity tangential and forward."""
    n = 120
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    V = np.column_stack([-np.sin(theta), np.cos(theta)])  # d/dtheta
    return make_adata(X, V, obs={"fucci_phase": theta})
