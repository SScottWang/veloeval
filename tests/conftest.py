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

import veloeval as ve


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
    adata.obsm[ve.KNN_KEY] = knn_indices(np.asarray(X_emb), k)
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
def ybranch():
    """A feeds both B and C, with the kNN written out by hand.

    Seven cells and an explicit neighbour table, because these fixtures exist to
    pin down how the edge scores are *aggregated* and a geometric kNN would make
    the expected numbers depend on tie-breaking.  Laid out so that

    * ``A -> B`` has boundary cells {0, 1} and ``A -> C`` has {1, 2} -- cell 1 is
      on both edges, cells 0 and 2 on one each, so averaging over cells and
      averaging over edges give different answers;
    * the two edges score differently under both ``cbdir`` (via the geometry)
      and ``cbvcoh`` (via the branch velocities).

    Worked out by hand, with ``v_A = (1, 0)``::

        cbdir   A -> B = mean(mean(1, 0.7071), 0.7071)  = 0.780330
                A -> C = mean(0, mean(0, -1))           = -0.25
    """
    X = np.array(
        [
            [0.0, 0.0],  # 0  A
            [0.0, 1.0],  # 1  A
            [0.0, 2.0],  # 2  A
            [1.0, 0.0],  # 3  B
            [1.0, 1.0],  # 4  B
            [0.0, 3.0],  # 5  C
            [-1.0, 2.0],  # 6  C
        ]
    )
    V = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    labels = np.array(["A", "A", "A", "B", "B", "C", "C"])
    adata = make_adata(X, V, labels=labels, k=3)
    adata.obsm[ve.KNN_KEY] = np.array(
        [
            [0, 3, 4],  # A cell on the A -> B boundary only
            [1, 3, 5],  # A cell on both boundaries
            [2, 5, 6],  # A cell on the A -> C boundary only
            [3, 0, 4],
            [4, 1, 3],
            [5, 2, 6],
            [6, 2, 5],
        ]
    )
    return adata


@pytest.fixture
def ybranch_edges():
    return [["A", "B"], ["A", "C"]]


@pytest.fixture
def gene_space(rng):
    """40 cells x 20 genes of gene-space velocity.

    The ground-truth metrics compare gene by gene and refuse to run on fewer
    than 10 shared genes, so they need a fixture wider than the 2-D ones.
    """
    V = rng.normal(size=(40, 20))
    return make_adata(np.zeros((40, 2)), np.zeros((40, 2)), V_gene=V)


@pytest.fixture
def scattered_genes(rng):
    """40 cells at distinct positions with 20-gene velocity, kNN without self.

    ``gene_space`` puts every cell at the origin, so its neighbour table is
    arbitrary and may or may not contain the cell itself; comparing against
    scVelo needs a table that follows the no-self convention.
    """
    X = rng.normal(size=(40, 2))
    adata = make_adata(X, X, V_gene=rng.normal(size=(40, 20)), k=7)
    adata.obsm[ve.KNN_KEY] = adata.obsm[ve.KNN_KEY][:, 1:]
    return adata


@pytest.fixture
def cycle():
    """Cells on a circle, phase = angle as a fraction of a turn, velocity forward."""
    n = 120
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    V = np.column_stack([-np.sin(theta), np.cos(theta)])  # d/dtheta
    return make_adata(X, V, obs={"fucci_phase": theta / (2 * np.pi)})
