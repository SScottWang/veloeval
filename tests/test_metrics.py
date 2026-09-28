"""Sign, scale and status behaviour of every metric on known-answer inputs."""

from __future__ import annotations

import numpy as np
import pytest

import veloeval as ve
from veloeval import metrics as M

# --------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------

def test_cbdir_perfect_reversed_random(linear, edges, rng):
    good = M.cbdir(linear, label_key="clusters", cluster_edges=edges)
    assert good.status == "ok"
    assert good.value == pytest.approx(1.0, abs=1e-6)

    rev = linear.copy()
    rev.obsm["velocity_umap"] = -rev.obsm["velocity_umap"]
    assert M.cbdir(rev, label_key="clusters", cluster_edges=edges).value == pytest.approx(
        -1.0, abs=1e-6
    )

    noisy = linear.copy()
    noisy.obsm["velocity_umap"] = rng.normal(size=noisy.obsm["velocity_umap"].shape)
    assert abs(M.cbdir(noisy, label_key="clusters", cluster_edges=edges).value) < 0.4


def test_cbdir_is_scale_invariant(linear, edges):
    base = M.cbdir(linear, label_key="clusters", cluster_edges=edges).value
    scaled = linear.copy()
    scaled.obsm["velocity_umap"] = scaled.obsm["velocity_umap"] * 1e4
    assert M.cbdir(scaled, label_key="clusters", cluster_edges=edges).value == pytest.approx(
        base, abs=1e-9
    )


def test_cbvcoh_coherent_field(linear, edges):
    assert M.cbvcoh(linear, label_key="clusters", cluster_edges=edges).value == pytest.approx(
        1.0, abs=1e-6
    )


def test_cto_orders_clusters(linear, edges):
    linear.obs["latent_time"] = np.linspace(0, 1, linear.n_obs)
    assert M.cto(linear, label_key="clusters", cluster_edges=edges).value == 1.0

    linear.obs["latent_time"] = np.linspace(1, 0, linear.n_obs)
    assert M.cto(linear, label_key="clusters", cluster_edges=edges).value == 0.0


# --------------------------------------------------------------------------
# Coherence
# --------------------------------------------------------------------------

def test_icvcoh_aligned_vs_random(linear, rng):
    assert M.icvcoh(linear, label_key="clusters").value == pytest.approx(1.0, abs=1e-6)

    noisy = linear.copy()
    noisy.layers["velocity"] = rng.normal(size=(noisy.n_obs, 2))
    assert abs(M.icvcoh(noisy, label_key="clusters").value) < 0.4


def test_velocity_consistency_needs_no_labels(cycle):
    res = M.velocity_consistency(cycle)
    assert res.status == "ok"
    assert res.value > 0.9  # neighbours on a smooth circle are nearly parallel


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def test_phase_dir_forward_and_backward(cycle):
    fwd = M.phase_dir(cycle)
    assert fwd.status == "ok"
    assert fwd.value == pytest.approx(1.0, abs=0.05)

    back = cycle.copy()
    back.obsm["velocity_umap"] = -back.obsm["velocity_umap"]
    assert M.phase_dir(back).value == pytest.approx(-1.0, abs=0.05)


def test_phase_dir_handles_the_wraparound(cycle):
    """Cells straddling phase 0 must not be scored backwards."""
    per_cell = M.phase_dir(cycle).per_cell
    phi = cycle.obs["fucci_phase"].to_numpy()
    near_origin = (phi < 0.2) | (phi > 2 * np.pi - 0.2)
    assert np.nanmean(per_cell[near_origin]) == pytest.approx(1.0, abs=0.05)


def test_phase_dir_not_applicable_without_phase(linear):
    res = M.phase_dir(linear)
    assert res.status == "not_applicable"
    assert "fucci_phase" in res.detail


def test_truth_cos_against_itself_is_one(gene_space):
    assert M.truth_cos(gene_space, gene_space.copy()).value == pytest.approx(1.0, abs=1e-6)


def test_truth_cos_opposite_is_minus_one(gene_space):
    flipped = gene_space.copy()
    flipped.layers["velocity"] = -flipped.layers["velocity"]
    assert M.truth_cos(gene_space, flipped).value == pytest.approx(-1.0, abs=1e-6)


def test_truth_cos_needs_enough_shared_genes(linear):
    res = M.truth_cos(linear, linear.copy())
    assert res.status == "not_applicable"
    assert "shared" in res.detail


def test_truth_cos_rejects_latent_space(gene_space):
    latent = gene_space.copy()
    ve.set_velocity_space(latent, "latent")
    res = M.truth_cos(latent, gene_space.copy())
    assert res.status == "not_applicable"
    assert "latent" in res.detail


def test_gamma_corr(gene_space, rng):
    a, b = gene_space.copy(), gene_space.copy()
    g = rng.random(a.n_vars)
    a.var["fit_gamma"] = g
    b.var["fit_gamma"] = g * 3 + 1  # monotone -> rank correlation 1
    assert M.gamma_corr(a, b).value == pytest.approx(1.0, abs=1e-9)


def test_gamma_corr_not_applicable_for_rate_free_methods(gene_space, rng):
    ref = gene_space.copy()
    ref.var["fit_gamma"] = rng.random(ref.n_vars)
    res = M.gamma_corr(gene_space.copy(), ref)  # method exposes no gamma
    assert res.status == "not_applicable"


# --------------------------------------------------------------------------
# Temporal
# --------------------------------------------------------------------------

def test_tsc_monotone_and_reversed(linear):
    linear.obs["latent_time"] = np.linspace(0, 1, linear.n_obs)
    linear.obs["stage"] = np.linspace(0, 10, linear.n_obs)
    assert M.tsc(linear, time_key="latent_time", true_time_key="stage").value == pytest.approx(
        1.0, abs=1e-9
    )

    linear.obs["stage"] = np.linspace(10, 0, linear.n_obs)
    assert M.tsc(linear, time_key="latent_time", true_time_key="stage").value == pytest.approx(
        -1.0, abs=1e-9
    )


def test_tsc_not_applicable_without_measured_axis(linear):
    linear.obs["latent_time"] = np.linspace(0, 1, linear.n_obs)
    r = M.tsc(linear, time_key="latent_time", true_time_key="nope")
    assert r.status == "not_applicable"


# --------------------------------------------------------------------------
# The shared neighbourhood
# --------------------------------------------------------------------------

@pytest.fixture
def blobs(rng):
    """Three overlapping clusters, 80 genes, PCA already run to 30 components.

    More than 50 genes because below that scanpy ignores X_pca and n_pcs and
    measures distance in X directly.  Overlapping so that changing n_pcs
    actually changes the neighbourhood.
    """
    import anndata as ad
    import scanpy as sc

    centres = rng.normal(scale=1.5, size=(3, 80))
    X = np.repeat(centres, 40, axis=0) + rng.normal(scale=1.0, size=(120, 80))
    adata = ad.AnnData(X=X.astype(np.float32))
    sc.pp.pca(adata, n_comps=30)
    return adata


def test_build_neighbor_indices_follows_scanpy_n_neighbors(blobs):
    # scanpy's n_neighbors counts the cell itself: 10 means 9 other cells.
    idx = ve.build_neighbor_indices(blobs, n_neighbors=10, n_pcs=5)

    assert idx.shape == (blobs.n_obs, 9)
    assert not any(i in row for i, row in enumerate(idx)), "cell is its own neighbour"
    np.testing.assert_array_equal(idx, blobs.obsm[ve.KNN_KEY])


def test_build_neighbor_indices_matches_exact_knn_on_the_same_pcs(blobs):
    from sklearn.neighbors import NearestNeighbors

    idx = ve.build_neighbor_indices(blobs, n_neighbors=8, n_pcs=5)

    rep = np.asarray(blobs.obsm["X_pca"])[:, :5]
    exact = NearestNeighbors(n_neighbors=8).fit(rep).kneighbors(rep)[1]
    for mine, theirs in zip(idx, exact):
        assert set(mine) == set(theirs[1:])


def test_build_neighbor_indices_honours_n_pcs(blobs):
    five = ve.build_neighbor_indices(blobs, n_neighbors=8, n_pcs=5)
    thirty = ve.build_neighbor_indices(blobs, n_neighbors=8, n_pcs=30)
    changed = sum(set(a) != set(b) for a, b in zip(five, thirty))
    assert changed > blobs.n_obs // 2, f"only {changed} cells moved"


def test_build_neighbor_indices_reuses_the_methods_graph_by_default(blobs):
    import scanpy as sc

    sc.pp.neighbors(blobs, n_neighbors=6, n_pcs=5)  # what the method's pipeline did
    idx = ve.build_neighbor_indices(blobs, n_neighbors=30, n_pcs=30)

    assert idx.shape == (blobs.n_obs, 5), "the method's k, not ours"
    D = blobs.obsp["distances"]
    for i in (0, 57, 119):
        assert set(idx[i]) == set(D[i].indices)
    assert blobs.uns["veloeval"]["prepared"]["neighbors"]["source"] == "reused"


def test_build_neighbor_indices_overwrite_leaves_the_methods_graph_alone(blobs):
    """scv.tl.velocity_graph reads obsp['distances'], so overwrite builds into
    a separate slot instead of replacing it."""
    import scanpy as sc

    sc.pp.neighbors(blobs, n_neighbors=6, n_pcs=5)
    before = blobs.obsp["distances"].copy()

    idx = ve.build_neighbor_indices(blobs, n_neighbors=10, n_pcs=5, overwrite=True)

    assert idx.shape == (blobs.n_obs, 9)
    assert (blobs.obsp["distances"] != before).nnz == 0
    assert blobs.uns["neighbors"]["params"]["n_neighbors"] == 6
    assert blobs.uns["veloeval"]["prepared"]["neighbors"]["source"] == "built"


def test_build_neighbor_indices_handles_scanpys_approximate_layout(blobs):
    """Above 8192 cells scanpy stores the cell itself (at distance 0) plus one
    extra neighbour per row.  Reproduced by hand here rather than with 8192
    cells: the rows must still come out as n_neighbors - 1 non-self cells."""
    from scipy.sparse import csr_matrix

    n, k = blobs.n_obs, 4
    rep = np.asarray(blobs.obsm["X_pca"])[:, :5]
    d = np.linalg.norm(rep[:, None] - rep[None], axis=2)
    order = np.argsort(d, axis=1)[:, : k + 1]  # self + k others
    rows = np.repeat(np.arange(n), k + 1)
    D = csr_matrix((d[rows, order.ravel()], (rows, order.ravel())), shape=(n, n))
    blobs.obsp["distances"] = D
    blobs.uns["neighbors"] = {"distances_key": "distances", "params": {"n_neighbors": k}}

    idx = ve.build_neighbor_indices(blobs)

    assert idx.shape == (n, k - 1)
    assert not any(i in row for i, row in enumerate(idx))
    for i in range(n):
        assert set(idx[i]) == set(order[i, 1:k])


def test_build_neighbor_indices_does_not_break_a_later_scanpy_call(blobs):
    """Regression: the indices used to go into uns['neighbors'], which left a
    record with no 'params' and made scanpy's own Neighbors.__init__ raise."""
    import scanpy as sc

    ve.build_neighbor_indices(blobs, n_neighbors=8, n_pcs=5)
    sc.pp.neighbors(blobs, n_neighbors=6, n_pcs=5)  # must not raise


def test_get_neighbor_indices_still_reads_the_legacy_slot(blobs):
    idx = ve.build_neighbor_indices(blobs, n_neighbors=5, n_pcs=5)
    legacy = blobs.copy()
    del legacy.obsm[ve.KNN_KEY]
    legacy.uns["neighbors"] = {"indices": idx}

    from veloeval.access import get_neighbor_indices

    np.testing.assert_array_equal(get_neighbor_indices(legacy), idx)


def test_build_neighbor_indices_accepts_a_latent_space(blobs):
    blobs.obsm["X_latent"] = np.asarray(blobs.X)[:, :3]
    idx = ve.build_neighbor_indices(blobs, n_neighbors=6, n_pcs=None, use_rep="X_latent")
    assert idx.shape == (blobs.n_obs, 5)


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------

def test_sts_and_ees_on_known_matrices(linear):
    n = linear.n_obs

    stay = linear.copy()
    stay.obsp["T_fwd"] = np.eye(n)
    assert M.sts(stay).value == pytest.approx(1.0)

    uniform = linear.copy()
    uniform.obsp["T_fwd"] = np.full((n, n), 1.0 / n)
    assert M.ees(uniform).value == pytest.approx(1.0, abs=1e-9)

    confident = linear.copy()
    T = np.zeros((n, n))
    T[:, 0] = 0.99
    T[:, 1] = 0.01
    confident.obsp["T_fwd"] = T
    assert M.ees(confident).value < 0.2


# --------------------------------------------------------------------------
# Status semantics -- the reason this library exists
# --------------------------------------------------------------------------

def test_the_four_statuses_are_distinguishable(linear, edges):
    # not_applicable: dataset has no curated edges
    assert M.cbdir(linear, label_key="clusters", cluster_edges=None).status == (
        "not_applicable"
    )

    # missing_input: upstream never wrote the projection
    broken = linear.copy()
    del broken.obsm["velocity_umap"]
    res = M.cbdir(broken, label_key="clusters", cluster_edges=edges)
    assert res.status == "missing_input"
    assert "velocity_umap" in res.detail

    # failed: something genuinely raised
    corrupt = linear.copy()
    del corrupt.obsm[ve.KNN_KEY]
    corrupt.uns["neighbors"] = {"indices": "not an array"}
    assert M.cbdir(corrupt, label_key="clusters", cluster_edges=edges).status == "failed"

    # ok
    assert M.cbdir(linear, label_key="clusters", cluster_edges=edges).status == "ok"


def test_not_applicable_without_labels(linear):
    assert M.icvcoh(linear, label_key=None).status == "not_applicable"


def test_no_metric_ever_raises(gene_space):
    """A stripped AnnData must produce statuses, not a traceback."""
    empty = gene_space.copy()
    del empty.layers["velocity"]
    del empty.obsm["velocity_umap"]

    calls = [
        lambda a: M.cbdir(a, label_key="clusters", cluster_edges=[["A", "B"]]),
        lambda a: M.cbvcoh(a, label_key="clusters", cluster_edges=[["A", "B"]]),
        lambda a: M.cto(a, label_key="clusters", cluster_edges=[["A", "B"]]),
        lambda a: M.icvcoh(a, label_key="clusters"),
        lambda a: M.velocity_consistency(a),
        lambda a: M.tsc(a, time_key="latent_time", true_time_key="stage"),
        lambda a: M.sts(a),
        lambda a: M.ees(a),
        lambda a: M.phase_dir(a),
        lambda a: M.truth_cos(a, gene_space),
        lambda a: M.gamma_corr(a, gene_space),
    ]
    for call in calls:
        assert call(empty).status != "ok"


def test_direction_covers_every_exported_metric():
    exported = {n for n in M.__all__ if n != "DIRECTION"}
    assert exported == set(M.DIRECTION)


def test_version_is_importable():
    assert ve.__version__ == "0.0.1"
