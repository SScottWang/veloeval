"""Sign, scale and status behaviour of every metric on known-answer inputs."""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
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
    scaled_value = M.cbdir(scaled, label_key="clusters", cluster_edges=edges).value
    assert scaled_value == pytest.approx(base, abs=1e-9)


def test_cbvcoh_coherent_field(linear, edges):
    value = M.cbvcoh(linear, label_key="clusters", cluster_edges=edges).value
    assert value == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("fn", [M.cbdir, M.cbvcoh])
def test_edge_order_does_not_change_the_answer(fn, ybranch, ybranch_edges):
    """Regression: per_cell used to be overwritten edge by edge.

    A cell in A is a boundary cell of both A -> B and A -> C, so the last edge
    written won and the score depended on how cluster_edges happened to be
    ordered.
    """
    forward = fn(ybranch, label_key="clusters", cluster_edges=ybranch_edges)
    reverse = fn(ybranch, label_key="clusters", cluster_edges=ybranch_edges[::-1])

    assert forward.value == pytest.approx(reverse.value, abs=1e-12)
    np.testing.assert_allclose(forward.per_cell, reverse.per_cell, equal_nan=True)
    assert forward.per_group == reverse.per_group


@pytest.mark.parametrize("fn", [M.cbdir, M.cbvcoh])
def test_per_group_has_one_entry_per_edge(fn, ybranch, ybranch_edges):
    r = fn(ybranch, label_key="clusters", cluster_edges=ybranch_edges)
    assert set(r.per_group) == {"A -> B", "A -> C"}
    assert r.value == pytest.approx(np.mean(list(r.per_group.values())))


def test_cbdir_matches_hand_computed_edge_scores(ybranch, ybranch_edges):
    r = M.cbdir(ybranch, label_key="clusters", cluster_edges=ybranch_edges)
    half = np.sqrt(0.5)

    assert r.per_group["A -> B"] == pytest.approx(np.mean([np.mean([1.0, half]), half]))
    assert r.per_group["A -> C"] == pytest.approx(np.mean([0.0, np.mean([0.0, -1.0])]))
    assert r.per_cell[0] == pytest.approx(np.mean([1.0, half]))
    assert r.per_cell[1] == pytest.approx(np.mean([half, 0.0]))
    assert r.per_cell[2] == pytest.approx(-0.5)
    assert np.isnan(r.per_cell[3:]).all(), "only source-cluster cells are scored"


def test_cbdir_weights_edges_equally_not_cells(ybranch, ybranch_edges):
    """value is the mean of the edge scores, not of the cell scores.

    The two differ whenever the edges carry different numbers of boundary
    cells, which is the normal case: on pancreas one edge holds 63% of them.
    """
    r = M.cbdir(ybranch, label_key="clusters", cluster_edges=ybranch_edges)
    edge_mean = np.mean(list(r.per_group.values()))
    cell_mean = np.nanmean(r.per_cell)

    assert r.value == pytest.approx(edge_mean)
    assert r.per_group["A -> B"] > r.per_group["A -> C"]
    # Not a tautology: the fixture is built so the two aggregations disagree.
    assert abs(edge_mean - cell_mean) > 1e-6


def test_cbdir_per_cell_averages_every_edge_a_cell_joins(ybranch, ybranch_edges):
    r_both = M.cbdir(ybranch, label_key="clusters", cluster_edges=ybranch_edges)
    r_b = M.cbdir(ybranch, label_key="clusters", cluster_edges=[["A", "B"]])
    r_c = M.cbdir(ybranch, label_key="clusters", cluster_edges=[["A", "C"]])

    shared = ~np.isnan(r_b.per_cell) & ~np.isnan(r_c.per_cell)
    assert shared.sum() > 0, "fixture must have cells on both branches"
    np.testing.assert_allclose(
        r_both.per_cell[shared],
        (r_b.per_cell[shared] + r_c.per_cell[shared]) / 2,
        atol=1e-12,
    )


def test_cbdir_non_source_boundary_matches_hand_computed(ybranch, ybranch_edges):
    """Every neighbour outside A counts, so B vs C no longer matters.

    Non-A neighbours: cell 0 -> {3, 4}, cell 1 -> {3, 5}, cell 2 -> {5, 6}.
    """
    r = M.cbdir(
        ybranch, label_key="clusters", cluster_edges=ybranch_edges,
        boundary="non_source",
    )
    half = np.sqrt(0.5)
    expected = np.mean([np.mean([1.0, half]), np.mean([half, 0.0]), np.mean([0.0, -1.0])])

    assert r.per_group == pytest.approx({"A -> B": expected, "A -> C": expected})
    assert r.value == pytest.approx(expected)


@pytest.mark.parametrize("fn", [M.cbdir, M.cbvcoh])
def test_unknown_boundary_rule_fails_loudly(fn, ybranch, ybranch_edges):
    r = fn(ybranch, label_key="clusters", cluster_edges=ybranch_edges, boundary="B")
    assert r.status == "failed"
    assert "boundary must be one of" in r.detail


def test_cbvcoh_ignores_the_direction_of_the_field(ybranch, ybranch_edges):
    rev = ybranch.copy()
    rev.obsm["velocity_umap"] = -rev.obsm["velocity_umap"]
    fwd = M.cbvcoh(ybranch, label_key="clusters", cluster_edges=ybranch_edges)
    bwd = M.cbvcoh(rev, label_key="clusters", cluster_edges=ybranch_edges)
    assert bwd.per_group == pytest.approx(fwd.per_group)


def test_cto_orders_clusters(linear, edges):
    linear.obs["latent_time"] = np.linspace(0, 1, linear.n_obs)
    assert M.cto(linear, label_key="clusters", cluster_edges=edges).value == 1.0

    linear.obs["latent_time"] = np.linspace(1, 0, linear.n_obs)
    assert M.cto(linear, label_key="clusters", cluster_edges=edges).value == 0.0


def test_cto_is_the_pairwise_fraction(linear, edges, rng):
    t = rng.integers(0, 5, linear.n_obs).astype(float)
    t[:3] = np.nan
    linear.obs["latent_time"] = t
    labels = linear.obs["clusters"].to_numpy()

    expect = {}
    for src, tgt in edges:
        a, b = t[labels == src], t[labels == tgt]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        expect[f"{src} -> {tgt}"] = np.mean(a[:, None] < b[None, :])

    r = M.cto(linear, label_key="clusters", cluster_edges=edges)
    assert r.per_group == pytest.approx(expect)
    assert r.value == pytest.approx(np.mean(list(expect.values())))


def test_cto_counts_ties_as_wrong(linear, edges):
    linear.obs["latent_time"] = 0.0
    assert M.cto(linear, label_key="clusters", cluster_edges=edges).value == 0.0


def test_cto_stage_key_follows_category_order(linear):
    n = linear.n_obs
    linear.obs["day"] = pd.Categorical(
        np.repeat(["d10", "d2", "d5"], [n // 3, n // 3, n - 2 * (n // 3)]),
        categories=["d2", "d5", "d10"],
        ordered=True,
    )
    linear.obs["latent_time"] = linear.obs["day"].cat.codes.astype(float)
    r = M.cto(linear, stage_key="day")
    assert list(r.per_group) == ["d2 -> d5", "d5 -> d10"]
    assert r.value == 1.0


def test_cto_falls_back_only_when_asked(linear, edges):
    linear.obs["velocity_pseudotime"] = np.linspace(0, 1, linear.n_obs)
    kw = dict(label_key="clusters", cluster_edges=edges)
    assert M.cto(linear, **kw).status == "not_applicable"

    r = M.cto(linear, **kw, fallback_key="velocity_pseudotime")
    assert r.value == 1.0
    assert "fallback" in r.detail

    linear.obs["latent_time"] = np.linspace(1, 0, linear.n_obs)
    r = M.cto(linear, **kw, fallback_key="velocity_pseudotime")
    assert r.value == 0.0
    assert "fallback" not in r.detail


def test_cto_rejects_two_step_sources(linear, edges):
    linear.obs["latent_time"] = 0.0
    linear.obs["day"] = "d0"
    r = M.cto(
        linear, label_key="clusters", cluster_edges=edges, stage_key="day"
    )
    assert r.status == "failed"


# --------------------------------------------------------------------------
# Coherence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("basis", ["umap", None])
def test_icvcoh_aligned_vs_random(linear, rng, basis):
    r = M.icvcoh(linear, label_key="clusters", basis=basis)
    assert r.value == pytest.approx(1.0, abs=1e-6)

    noisy = linear.copy()
    noisy.layers["velocity"] = rng.normal(size=(noisy.n_obs, 2))
    noisy.obsm["velocity_umap"] = rng.normal(size=(noisy.n_obs, 2))
    assert abs(M.icvcoh(noisy, label_key="clusters", basis=basis).value) < 0.4


def test_icvcoh_reads_the_embedding_by_default(linear, rng):
    only_gene_noisy = linear.copy()
    only_gene_noisy.layers["velocity"] = rng.normal(size=(linear.n_obs, 2))
    r_emb = M.icvcoh(only_gene_noisy, label_key="clusters")
    r_gene = M.icvcoh(only_gene_noisy, label_key="clusters", basis=None)
    assert r_emb.value == pytest.approx(1.0, abs=1e-6)
    assert abs(r_gene.value) < 0.4


def test_icvcoh_gene_space_refuses_latent_velocity(linear):
    latent = linear.copy()
    ve.set_velocity_space(latent, "latent")
    assert M.icvcoh(latent, label_key="clusters", basis=None).status == "not_applicable"
    assert M.icvcoh(latent, label_key="clusters").status == "ok"


def test_velocity_consistency_needs_no_labels(cycle):
    pytest.importorskip("scvelo")
    res = M.velocity_consistency(cycle)
    assert res.status == "ok"
    assert res.value > 0.9  # neighbours on a smooth circle are nearly parallel


def _scvelo_confidence(adata, indices):
    """scVelo's own velocity_confidence on the given neighbour table."""
    import scvelo as scv
    from scipy.sparse import csr_matrix

    n, k = indices.shape
    ref = adata.copy()
    ref.obsp["distances"] = csr_matrix(
        (np.ones(n * k), indices.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n)
    )
    ref.obs["velocity_confidence_transition"] = 0.0
    scv.tl.velocity_confidence(ref)
    return ref.obs["velocity_confidence"].to_numpy()


def test_velocity_consistency_is_scvelo_velocity_confidence(scattered_genes):
    pytest.importorskip("scvelo")
    r = M.velocity_consistency(scattered_genes)
    expected = _scvelo_confidence(scattered_genes, scattered_genes.obsm[ve.KNN_KEY])

    np.testing.assert_allclose(r.per_cell, expected, atol=1e-12)
    assert r.value >= 0, "scVelo clips negative confidence to 0"


def test_velocity_consistency_without_scvelo_is_missing_input(
    scattered_genes, monkeypatch
):
    monkeypatch.setitem(sys.modules, "scvelo", None)
    r = M.velocity_consistency(scattered_genes)
    assert r.status == "missing_input"
    assert "scvelo" in r.detail


def test_velocity_consistency_drops_self_from_the_legacy_slot(scattered_genes):
    pytest.importorskip("scvelo")
    knn = scattered_genes.obsm[ve.KNN_KEY]
    legacy = scattered_genes.copy()
    del legacy.obsm[ve.KNN_KEY]
    legacy.uns["neighbors"] = {
        "indices": np.column_stack([np.arange(legacy.n_obs), knn])
    }
    np.testing.assert_allclose(
        M.velocity_consistency(legacy).per_cell,
        M.velocity_consistency(scattered_genes).per_cell,
        atol=1e-12,
    )


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
    near_origin = (phi < 0.03) | (phi > 0.97)
    assert np.nanmean(per_cell[near_origin]) == pytest.approx(1.0, abs=0.05)


def test_phase_dir_reads_radians_with_period(cycle):
    radians = cycle.copy()
    radians.obs["fucci_phase"] = radians.obs["fucci_phase"] * 2 * np.pi
    assert M.phase_dir(radians, period=2 * np.pi).per_cell == pytest.approx(
        M.phase_dir(cycle).per_cell
    )


def test_phase_dir_breaks_down_by_phase_bin(cycle):
    r = M.phase_dir(cycle, n_bins=4)
    assert list(r.per_group) == ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
    assert all(v == pytest.approx(1.0, abs=0.05) for v in r.per_group.values())
    assert "median R2" in r.detail


def test_phase_dir_min_r2_drops_noisy_neighbourhoods(cycle, rng):
    noisy = cycle.copy()
    phase = noisy.obs["fucci_phase"].to_numpy().copy()
    phase[::2] = rng.uniform(0, 1, phase[::2].size)
    noisy.obs["fucci_phase"] = phase
    everyone = M.phase_dir(noisy)
    strict = M.phase_dir(noisy, min_r2=0.5)
    assert np.isfinite(strict.per_cell).sum() < np.isfinite(everyone.per_cell).sum()


def test_phase_dir_n_dims_ignores_trailing_columns(cycle, rng):
    padded = cycle.copy()
    for key in ("X_umap", "velocity_umap"):
        extra = rng.normal(size=(padded.n_obs, 3))
        padded.obsm[key] = np.hstack([padded.obsm[key], extra])
    assert M.phase_dir(padded, n_dims=2).per_cell == pytest.approx(
        M.phase_dir(cycle).per_cell
    )


def test_phase_dir_not_applicable_without_phase(linear):
    res = M.phase_dir(linear)
    assert res.status == "not_applicable"
    assert "fucci_phase" in res.detail


def test_truth_cos_against_itself_is_one(gene_space):
    value = M.truth_cos(gene_space, gene_space.copy()).value
    assert value == pytest.approx(1.0, abs=1e-6)


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


def test_truth_cos_scores_only_the_chosen_genes(gene_space, rng):
    ref = gene_space.copy()
    half = gene_space.n_vars // 2
    ref.layers["velocity"] = np.asarray(ref.layers["velocity"]).copy()
    ref.layers["velocity"][:, half:] = rng.normal(size=(ref.n_obs, ref.n_vars - half))
    chosen = gene_space.var_names[:half]
    res = M.truth_cos(gene_space, ref, genes=chosen)
    assert res.value == pytest.approx(1.0, abs=1e-6)
    assert res.detail == f"{half} genes"
    assert M.truth_cos(gene_space, ref).detail == f"{gene_space.n_vars} genes"


def test_gamma_corr_scores_only_the_chosen_genes(gene_space, rng):
    a, b = gene_space.copy(), gene_space.copy()
    g = rng.random(a.n_vars)
    a.var["fit_gamma"] = g
    b.var["fit_gamma"] = np.where(np.arange(a.n_vars) < 10, g, rng.random(a.n_vars))
    res = M.gamma_corr(a, b, genes=a.var_names[:10])
    assert res.value == pytest.approx(1.0, abs=1e-9)
    assert res.detail == "10 genes"
    assert M.gamma_corr(a, b).value < 0.99


def test_gamma_corr_not_applicable_for_rate_free_methods(gene_space, rng):
    ref = gene_space.copy()
    ref.var["fit_gamma"] = rng.random(ref.n_vars)
    res = M.gamma_corr(gene_space.copy(), ref)  # method exposes no gamma
    assert res.status == "not_applicable"


def _cos_graph(X, V, k=15):
    """scVelo-style graph: cos(x_j - x_i, v_i) over kNN; + in graph, - in graph_neg."""
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(len(X)), k)
    D = X[idx.ravel()] - X[rows]
    c = np.einsum("ij,ij->i", D, V[rows]) / (
        np.linalg.norm(D, axis=1) * np.linalg.norm(V[rows], axis=1) + 1e-12
    )
    n = len(X)
    pos = csr_matrix((np.where(c > 0, c, 0), (rows, idx.ravel())), shape=(n, n))
    neg = csr_matrix((np.where(c < 0, c, 0), (rows, idx.ravel())), shape=(n, n))
    pos.eliminate_zeros()
    neg.eliminate_zeros()
    return pos, neg


def _larry_like(rng, n_clones=400, sisters=6):
    """Day-2 progenitors in a stem (x in [0,1]); Neu branch up-right, Mo down-right,
    Meg to the left.  Neu-vs-Mo bias rises with y; 20% of clones go Meg."""
    import anndata as ad

    X, lab, t, clone, V = [], [], [], [], []

    def add(xy, label, v):
        X.append(xy)
        lab.append(label)
        V.append(v)

    for c in range(n_clones):
        y = rng.uniform(-0.5, 0.5)
        x = rng.uniform(0, 1)
        bias = 1 / (1 + np.exp(-6 * y))
        meg = rng.random() < 0.2
        add([x, y], "undiff", [-1, 0] if meg else [1, 1.6 * (2 * bias - 1)])
        t.append(2)
        clone.append(c)
        for _ in range(sisters):
            s = rng.uniform(1.2, 3.0)
            if meg:
                add([-s, rng.normal(0, 0.08)], "Meg", [-1, 0])
            elif rng.random() < bias:
                add([s, 0.8 * (s - 1) + rng.normal(0, 0.08)], "Neutrophil", [1, 0.8])
            else:
                add([s, -0.8 * (s - 1) + rng.normal(0, 0.08)], "Monocyte", [1, -0.8])
            t.append(rng.choice([4, 6]))
            clone.append(c)
    for _ in range(1500):  # unbarcoded cells
        s = rng.uniform(0, 3)
        if s < 1.1:
            y = rng.uniform(-0.5, 0.5)
            add([s, y], "undiff", [1, 1.6 * (2 / (1 + np.exp(-6 * y)) - 1)])
        else:
            sign = rng.choice([1, -1])
            label = "Neutrophil" if sign > 0 else "Monocyte"
            add([s, sign * 0.8 * (s - 1) + rng.normal(0, 0.08)], label, [1, sign * 0.8])
        t.append(rng.choice([2, 4, 6]))
        clone.append(np.nan)
    obs = pd.DataFrame({"cell_type": lab, "time_point": t, "clone": clone})
    obs.index = obs.index.astype(str)
    return ad.AnnData(obs=obs), np.array(X, float), np.array(V, float)


@pytest.fixture(scope="module")
def larry():
    return _larry_like(np.random.default_rng(0))


def _field(larry, V=None):
    a, X, V0 = larry
    a = a.copy()
    graphs = _cos_graph(X, V0 if V is None else V)
    a.uns["velocity_graph"], a.uns["velocity_graph_neg"] = graphs
    return a


LARRY = dict(clone_key="clone", time_key="time_point", label_key="cell_type")


def test_velocity_transitions_match_scvelo_and_ees(larry):
    scv = pytest.importorskip("scvelo")
    import anndata as ad

    from veloeval.metrics._markov import velocity_transitions
    from veloeval.metrics.negative import _transition_rows

    a = _field(larry)
    T = velocity_transitions(a, "velocity", 10.0)
    for i, row in enumerate(_transition_rows(a, "velocity", 10.0)):
        np.testing.assert_allclose(np.sort(T[i].data), np.sort(row), atol=1e-12)

    sub = ad.AnnData(obs=pd.DataFrame(index=a.obs_names))
    sub.uns = {k: a.uns[k] for k in ("velocity_graph", "velocity_graph_neg")}
    ref = scv.tl.transition_matrix(sub, scale=10, self_transitions=False)
    assert abs(T - ref).max() < 1e-12


def test_absorption_matches_a_dense_solve():
    from scipy.sparse import csr_matrix

    from veloeval.metrics._markov import absorption

    rng = np.random.default_rng(3)
    n = 60
    P = rng.random((n, n)) * (rng.random((n, n)) < 0.2)
    np.fill_diagonal(P, 0)
    P[50:] = 0  # cells 50-59: an island with no absorbing cell
    P[50:, 50:] = rng.random((10, 10))
    P /= P.sum(axis=1, keepdims=True)
    P[:50, 50:] = 0
    P[:50] /= P[:50].sum(axis=1, keepdims=True)
    absorbing = np.zeros(n, bool)
    absorbing[:10] = True
    targets = [np.arange(n) < 4, (np.arange(n) >= 4) & (np.arange(n) < 10)]

    B = absorption(csr_matrix(P), absorbing, targets)
    live = np.arange(10, 50)
    A = np.eye(40) - P[np.ix_(live, live)]
    R = P[np.ix_(live, np.arange(10))]
    hit = np.column_stack([t[:10] for t in targets]).astype(float)
    dense = np.linalg.solve(A, R @ hit)
    np.testing.assert_allclose(B[live], dense, atol=1e-12)
    np.testing.assert_allclose(B[live].sum(axis=1), 1, atol=1e-12)
    assert np.isnan(B[50:]).all()
    np.testing.assert_array_equal(B[:10], np.column_stack(targets)[:10].astype(float))


def test_lineage_fate_against_its_floor(larry):
    true = M.lineage_fate(_field(larry), **LARRY)
    floor = M.lineage_fate(_field(larry), **LARRY, scale=0)
    rand = M.lineage_fate(
        _field(larry, np.random.default_rng(1).normal(size=larry[2].shape)), **LARRY
    )
    assert (true.value, floor.value, rand.value) == pytest.approx(
        (0.835, 0.695, 0.425), abs=5e-4
    )
    assert true.detail == "319 progenitors at 2 from 319 clones"
    assert np.isfinite(true.per_cell).sum() == 319

    rev = M.lineage_fate(_field(larry, -larry[2]), **LARRY)
    assert rev.status == "not_applicable"
    assert "trapped" in rev.detail


def test_lineage_fate_scores_trapped_progenitors_as_no_prediction(larry):
    a, X, V = larry
    V = V.copy()
    centre = np.array([0.5, 0.0])
    sink = (a.obs["cell_type"] == "undiff").to_numpy()
    sink &= np.linalg.norm(X - centre, axis=1) < 0.25
    V[sink] = centre - X[sink]
    res = M.lineage_fate(_field(larry, V), **LARRY)
    trapped = int(re.search(r"; (\d+) trapped, scored as 0\.5$", res.detail).group(1))
    assert 50 < trapped < 150  # which rows cross the tolerance depends on the LU backend
    assert np.isfinite(res.per_cell).sum() == 319
    assert (res.per_cell == 0.5).sum() == trapped
    assert res.value < M.lineage_fate(_field(larry), **LARRY).value


def test_lineage_fate_leaves_adata_alone(larry):
    a = _field(larry)
    keys = (list(a.obs), list(a.obsm), list(a.uns), list(a.obsp))
    M.lineage_fate(a, **LARRY)
    assert (list(a.obs), list(a.obsm), list(a.uns), list(a.obsp)) == keys


def test_lineage_fate_statuses(larry):
    a = _field(larry)
    no_neg = a.copy()
    del no_neg.uns["velocity_graph_neg"]
    assert M.lineage_fate(no_neg, **LARRY).status == "missing_input"
    assert M.lineage_fate(a, **{**LARRY, "clone_key": "nope"}).status == "missing_input"

    one = a.copy()
    one.obs["time_point"] = 2
    assert M.lineage_fate(one, **LARRY).status == "not_applicable"
    assert M.lineage_fate(a, **LARRY, min_sisters=50).status == "not_applicable"

    typo = M.lineage_fate(a, **LARRY, fates=("Neutrophill", "Monocyte"))
    assert typo.status == "failed" and "Neutrophill" in typo.detail

    days = a.copy()
    days.obs["time_point"] = "day" + days.obs["time_point"].astype(str)
    assert M.lineage_fate(days, **LARRY).status == "failed"
    days.obs["time_point"] = pd.Categorical(
        days.obs["time_point"], categories=["day2", "day4", "day6"], ordered=True
    )
    assert M.lineage_fate(days, **LARRY).value == M.lineage_fate(a, **LARRY).value


def test_rate_err(linear):
    linear.uns["cycle_period"] = 17.7
    assert M.rate_err(linear, period=17.7).value == 0
    linear.uns["cycle_period"] = -17.7
    assert M.rate_err(linear, period=17.7).value == 0

    linear.uns["cycle_period"] = 17.7
    res = M.rate_err(linear, period=17.7, half_life=1.2)
    assert res.value == pytest.approx(0.2)
    assert res.detail == "21.2 h inferred vs 17.7 h measured (+20%); half-life 1.2 h"

    assert M.rate_err(linear, period=-1).status == "failed"
    linear.uns["cycle_period"] = 0.0
    assert M.rate_err(linear, period=17.7).status == "not_applicable"
    del linear.uns["cycle_period"]
    assert M.rate_err(linear, period=17.7).status == "not_applicable"


# --------------------------------------------------------------------------
# Temporal
# --------------------------------------------------------------------------

def test_tsc_monotone_and_reversed(linear):
    linear.obs["latent_time"] = np.linspace(0, 1, linear.n_obs)
    linear.obs["stage"] = np.linspace(0, 10, linear.n_obs)
    forward = M.tsc(linear, time_key="latent_time", true_time_key="stage").value
    assert forward == pytest.approx(1.0, abs=1e-9)

    linear.obs["stage"] = np.linspace(10, 0, linear.n_obs)
    reversed_tsc = M.tsc(linear, time_key="latent_time", true_time_key="stage").value
    assert reversed_tsc == pytest.approx(-1.0, abs=1e-9)


def _days(n, ordered):
    labels = np.repeat(["day2", "day5", "day10"], [n // 3, n // 3, n - 2 * (n // 3)])
    if ordered:
        return pd.Categorical(labels, categories=["day2", "day5", "day10"], ordered=True)
    return labels


def test_string_stages_need_a_declared_order(linear, edges):
    linear.obs["latent_time"] = np.linspace(0, 1, linear.n_obs)
    raw = _days(linear.n_obs, False)
    for unordered in (raw, pd.Categorical(raw)):
        linear.obs["day"] = unordered
        r = M.tsc(linear, time_key="latent_time", true_time_key="day")
        assert r.status == "failed" and "ordered" in r.detail
        assert M.cto(linear, stage_key="day").status == "failed"

    linear.obs["day"] = _days(linear.n_obs, True)
    tsc = M.tsc(linear, time_key="latent_time", true_time_key="day")
    assert tsc.value > 0.8
    cto = M.cto(linear, stage_key="day")
    assert list(cto.per_group) == ["day2 -> day5", "day5 -> day10"]
    assert cto.value == 1.0


def test_numeric_stages_sort_by_value(linear):
    n = linear.n_obs
    linear.obs["latent_time"] = np.linspace(0, 1, n)
    days = np.repeat([2, 5, 10], [n // 3, n // 3, n - 2 * (n // 3)])
    for col in (days, pd.Categorical(days)):
        linear.obs["day"] = col
        assert M.tsc(linear, time_key="latent_time", true_time_key="day").value > 0.8
        assert list(M.cto(linear, stage_key="day").per_group) == ["2 -> 5", "5 -> 10"]


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

def _with_confidence(adata, conf):
    """A velocity graph whose row maxima are *conf* (each cell -> next cell)."""
    from scipy.sparse import csr_matrix

    n = adata.n_obs
    a = adata.copy()
    a.uns["velocity_graph"] = csr_matrix(
        (conf, (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n)
    )
    return a


def test_sts_follows_the_self_transition_formula(linear, rng):
    conf = rng.uniform(0, 0.9, linear.n_obs)
    r = M.sts(_with_confidence(linear, conf))
    expect = np.clip(np.percentile(conf, 98) - conf, 0, 1)
    assert r.per_cell == pytest.approx(expect)
    assert r.value == pytest.approx(expect.mean())


def test_sts_is_scvelos_self_transition(rng):
    scv = pytest.importorskip("scvelo")
    import anndata as ad
    import scanpy as sc

    S = rng.gamma(2, 1, (150, 30))
    a = ad.AnnData(S.copy())
    a.layers["Ms"] = S
    a.layers["velocity"] = rng.normal(size=S.shape)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=15)
    scv.tl.velocity_graph(a, n_jobs=1, backend="threading")

    expect = a.obs["velocity_self_transition"].to_numpy()
    np.testing.assert_allclose(M.sts(a).per_cell, expect, atol=1e-6)


def test_sts_abs_sees_a_uniform_shift_that_sts_does_not(linear, rng):
    noise = rng.normal(0, 0.03, linear.n_obs)
    quiet = _with_confidence(linear, np.clip(0.1 + noise, 0, 1))
    loud = _with_confidence(linear, np.clip(0.8 + noise, 0, 1))
    assert M.sts(quiet).value == pytest.approx(M.sts(loud).value, abs=1e-3)
    assert M.sts_abs(quiet).value == pytest.approx(0.9, abs=0.01)
    assert M.sts_abs(loud).value == pytest.approx(0.2, abs=0.01)


def test_sts_scores_only_the_chosen_groups(linear, rng):
    conf = rng.uniform(0, 0.9, linear.n_obs)
    a = _with_confidence(linear, conf)
    labels = a.obs["clusters"].to_numpy().astype(str)
    everyone = np.clip(np.percentile(conf, 98) - conf, 0, 1)

    one = M.sts(a, label_key="clusters", groups="B")
    assert one.value == pytest.approx(everyone[labels == "B"].mean())
    assert np.isnan(one.per_cell[labels != "B"]).all()

    two = M.sts(a, label_key="clusters", groups=["A", "B"])
    assert two.value == pytest.approx(everyone[np.isin(labels, ["A", "B"])].mean())

    own = M.sts(a, label_key="clusters", groups="B", reference="groups")
    c = conf[labels == "B"]
    assert own.value == pytest.approx(np.clip(np.percentile(c, 98) - c, 0, 1).mean())

    absolute = M.sts_abs(a, label_key="clusters", groups=["B"])
    assert absolute.value == pytest.approx(1 - c.mean())


@pytest.mark.parametrize(
    "kw",
    [
        dict(groups="B"),
        dict(label_key="clusters", groups=["B", "typo"]),
        dict(label_key="clusters", groups="B", reference="median"),
    ],
)
def test_sts_rejects_bad_group_arguments(linear, rng, kw):
    a = _with_confidence(linear, rng.uniform(0, 0.9, linear.n_obs))
    assert M.sts(a, **kw).status == "failed"


def _with_cosines(adata, cos):
    """A velocity graph where cell i's neighbours are the next ``cos.shape[1]`` cells."""
    from scipy.sparse import csr_matrix

    n, k = cos.shape
    rows = np.repeat(np.arange(n), k)
    cols = (rows + np.tile(np.arange(1, k + 1), n)) % n
    a = adata.copy()
    v = cos.ravel()
    for key, keep in (("velocity_graph", v > 0), ("velocity_graph_neg", v < 0)):
        a.uns[key] = csr_matrix((v[keep], (rows[keep], cols[keep])), shape=(n, n))
    return a


def test_ees_counts_equally_likely_neighbours(linear):
    n = linear.n_obs
    flat = _with_cosines(linear, np.full((n, 8), 0.3))
    assert M.ees(flat).value == pytest.approx(8.0)

    cos = np.full((n, 8), -0.5)
    cos[:, 0] = 0.9
    sharp = _with_cosines(linear, cos)
    assert M.ees(sharp).value == pytest.approx(1.0)


def test_ees_is_scvelos_transition_matrix(linear, rng):
    scv = pytest.importorskip("scvelo")
    n = linear.n_obs
    a = _with_cosines(linear, rng.uniform(-1, 1, (n, 10)))
    T = scv.tl.transition_matrix(a, scale=30, self_transitions=False).tocsr()

    def hill(p, mass=0.95):
        p = np.sort(p / p.sum())[::-1]
        m = np.searchsorted(np.cumsum(p), mass, side="left") + 1
        q = p[:m] / p[:m].sum()
        return np.exp(-np.sum(q * np.log(q)))

    expect = [hill(T.data[T.indptr[i] : T.indptr[i + 1]]) for i in range(n)]
    assert M.ees(a).per_cell == pytest.approx(expect, rel=1e-4)


def test_ees_scores_only_the_chosen_groups(linear, rng):
    a = _with_cosines(linear, rng.uniform(-1, 1, (linear.n_obs, 10)))
    labels = a.obs["clusters"].to_numpy().astype(str)
    everyone = M.ees(a).per_cell
    one = M.ees(a, label_key="clusters", groups="B")
    assert one.value == pytest.approx(everyone[labels == "B"].mean())
    assert np.isnan(one.per_cell[labels != "B"]).all()
    assert M.ees(a, mass=0).status == "failed"


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
        lambda a: M.sts_abs(a),
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
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", ve.__version__)


# --------------------------------------------------------------------------
# Cyclic time
# --------------------------------------------------------------------------

def _cyclic(t, phase):
    import anndata as ad

    return ad.AnnData(
        obs=pd.DataFrame(
            {"latent_time": t, "fucci_phase": phase},
            index=[f"c{i}" for i in range(len(t))],
        )
    )


@pytest.fixture
def skewed_phase(rng):
    """800 cells piled up early in the cycle, as FUCCI cells pile up in G1."""
    return rng.beta(2, 5, 800)


def test_fisher_lee_closed_form_is_the_pairwise_definition(rng):
    from veloeval._math import fisher_lee

    a, b = rng.uniform(0, 2 * np.pi, (2, 60))
    i, j = np.triu_indices(60, 1)
    sa, sb = np.sin(a[i] - a[j]), np.sin(b[i] - b[j])
    naive = (sa * sb).sum() / np.sqrt((sa**2).sum() * (sb**2).sum())
    assert fisher_lee(a, b) == pytest.approx(naive, abs=1e-10)


@pytest.mark.parametrize("cut", [0.0, 0.5, 0.8])
def test_phase_corr_ignores_where_a_correct_order_cuts_the_cycle(skewed_phase, cut):
    t = ((skewed_phase - cut) % 1) ** 3 * 7 + 2
    assert M.phase_corr(_cyclic(t, skewed_phase)).value > 0.99


def test_tsc_depends_on_the_cut_which_is_why_phase_corr_exists(skewed_phase):
    t = ((skewed_phase - 0.5) % 1) ** 3 * 7 + 2
    a = _cyclic(t, skewed_phase)
    assert M.tsc(a, time_key="latent_time", true_time_key="fucci_phase").value < 0.5


def test_phase_corr_reversed_random_and_radians(skewed_phase, rng):
    phase = skewed_phase
    assert M.phase_corr(_cyclic((0.3 - phase) % 1, phase)).value < -0.99
    assert abs(M.phase_corr(_cyclic(rng.uniform(size=phase.size), phase)).value) < 0.1
    radians = (2 * np.pi * phase + 1.0) % (2 * np.pi)
    assert M.phase_corr(_cyclic(radians, phase)).value > 0.99


def test_phase_corr_drops_nan_cells(skewed_phase):
    t = skewed_phase.copy()
    t[:50] = np.nan
    r = M.phase_corr(_cyclic(t, skewed_phase))
    assert r.status == "ok"
    assert r.detail == "750 cells"


def test_phase_corr_not_applicable(skewed_phase):
    n = skewed_phase.size
    constant = _cyclic(np.ones(n), skewed_phase)
    two_valued = _cyclic((np.arange(n) % 2).astype(float), skewed_phase)
    assert M.phase_corr(constant).status == "not_applicable"
    assert M.phase_corr(two_valued).status == "not_applicable"
    a = _cyclic(skewed_phase, skewed_phase)
    assert M.phase_corr(a, time_key="absent").status == "not_applicable"
    assert M.phase_corr(a, phase_key="absent").status == "not_applicable"


# --------------------------------------------------------------------------
# Spatial
# --------------------------------------------------------------------------

def _hex_grid(n_rows=8, n_cols=8):
    r, c = np.divmod(np.arange(n_rows * n_cols), n_cols)
    return np.column_stack([c + 0.5 * (r % 2), r * np.sqrt(3) / 2])


def test_build_spatial_neighbors_takes_the_hexagonal_ring():
    import anndata as ad

    xy = _hex_grid()
    a = ad.AnnData(obs=pd.DataFrame(index=[f"s{i}" for i in range(len(xy))]))
    a.obsm["spatial"] = xy
    idx = ve.build_spatial_neighbors(a)

    assert idx.shape == (len(xy), 6)
    assert (idx != np.arange(len(xy))[:, None]).all()
    interior = 3 * 8 + 3
    d = np.linalg.norm(xy[idx[interior]] - xy[interior], axis=1)
    assert d == pytest.approx(np.ones(6))
    assert a.obsm[ve.SPATIAL_KNN_KEY] is idx
    assert a.uns["veloeval"]["prepared"]["spatial_neighbors"] == {
        "spatial_key": "spatial",
        "n_neighbors": 6,
    }


def test_build_spatial_neighbors_survives_coincident_spots():
    import anndata as ad

    xy = np.vstack([_hex_grid(), _hex_grid()[:5]])
    a = ad.AnnData(obs=pd.DataFrame(index=[f"s{i}" for i in range(len(xy))]))
    a.obsm["spatial"] = xy
    idx = ve.build_spatial_neighbors(a)
    assert idx.shape == (len(xy), 6)
    assert (idx != np.arange(len(xy))[:, None]).all()


def test_prepare_builds_the_spatial_graph_only_when_asked(blobs):
    blobs.obsm["spatial"] = blobs.obsm["X_pca"][:, :2]
    plain = blobs.copy()
    ve.prepare(plain)
    assert ve.SPATIAL_KNN_KEY not in plain.obsm

    ve.prepare(blobs, spatial_key="spatial")
    assert blobs.obsm[ve.SPATIAL_KNN_KEY].shape == (blobs.n_obs, 6)


@pytest.fixture
def spots(rng):
    """64 hexagonal spots with 20-gene velocity and the spatial kNN built."""
    import anndata as ad

    xy = _hex_grid()
    a = ad.AnnData(np.zeros((len(xy), 20)))
    a.layers["velocity"] = rng.normal(size=(len(xy), 20))
    a.obsm["spatial"] = xy
    ve.build_spatial_neighbors(a)
    return a


def test_spatial_consistency_uniform_field_scores_one(spots, rng):
    spots.layers["velocity"] = np.tile(rng.normal(size=20), (spots.n_obs, 1))
    assert M.spatial_consistency(spots).value == pytest.approx(1.0)


def test_spatial_consistency_cannot_see_a_reversed_field(spots):
    flipped = spots.copy()
    flipped.layers["velocity"] = -flipped.layers["velocity"]
    np.testing.assert_allclose(
        M.spatial_consistency(flipped).per_cell,
        M.spatial_consistency(spots).per_cell,
    )


def test_spatial_consistency_is_scvelo_velocity_confidence(spots):
    pytest.importorskip("scvelo")
    expected = _scvelo_confidence(spots, spots.obsm[ve.SPATIAL_KNN_KEY])
    r = M.spatial_consistency(spots)
    np.testing.assert_allclose(r.per_cell, expected, atol=1e-12)


def test_spatial_consistency_needs_the_spatial_graph(spots):
    del spots.obsm[ve.SPATIAL_KNN_KEY]
    r = M.spatial_consistency(spots)
    assert r.status == "missing_input"
    assert "spatial_key" in r.detail


def _ring(t):
    import anndata as ad

    n = len(t)
    obs = pd.DataFrame({"latent_time": t}, index=[f"s{i}" for i in range(n)])
    a = ad.AnnData(obs=obs)
    i = np.arange(n)
    a.obsm[ve.SPATIAL_KNN_KEY] = np.column_stack([(i - 1) % n, (i + 1) % n])
    return a


def test_time_morans_i_exact_on_a_ring():
    n = 40
    i = np.arange(n)
    assert M.time_morans_i(_ring((-1.0) ** i)).value == pytest.approx(-1.0)
    wave = np.cos(2 * np.pi * i / n)
    assert M.time_morans_i(_ring(wave)).value == pytest.approx(np.cos(2 * np.pi / n))
    assert M.time_morans_i(_ring(1 - wave)).value == pytest.approx(
        M.time_morans_i(_ring(wave)).value
    )


def test_time_morans_i_nan_shuffled_and_statuses(rng):
    n = 40
    t = np.cos(2 * np.pi * np.arange(n) / n)
    t[[3, 17]] = np.nan
    assert np.isfinite(M.time_morans_i(_ring(t)).value)

    shuffled = M.time_morans_i(_ring(rng.permutation(400).astype(float)))
    assert shuffled.value == pytest.approx(-1 / 399, abs=0.2)
    assert shuffled.detail == "expected -0.003 without spatial structure"

    a = _ring(np.ones(n))
    assert M.time_morans_i(a).status == "not_applicable"
    assert M.time_morans_i(a, time_key="absent").status == "not_applicable"
    del a.obsm[ve.SPATIAL_KNN_KEY]
    a.obs["latent_time"] = np.arange(n, dtype=float)
    assert M.time_morans_i(a).status == "missing_input"


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------

@pytest.fixture
def cells(rng):
    """200 cells in 2-D with a shared 12-neighbour kNN and a +x flow."""
    X = rng.normal(size=(200, 2))
    from tests.conftest import knn_indices

    return X, knn_indices(X, 13)[:, 1:], np.tile([1.0, 0.0], (200, 1))


def _run(X, knn, V):
    """A run whose velocity graph holds cos(x_j - x_i, v_i) over the kNN."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    n, k = knn.shape
    rows = np.repeat(np.arange(n), k)
    cols = knn.ravel()
    d = X[cols] - X[rows]
    v = V[rows]
    cos = (d * v).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(v, axis=1))
    a = ad.AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]))
    for key, keep in (("velocity_graph", cos > 0), ("velocity_graph_neg", cos < 0)):
        a.uns[key] = csr_matrix((cos[keep], (rows[keep], cols[keep])), shape=(n, n))
    return a


def _noisy_runs(cells, rng, n_runs=4, sd=0.3):
    X, knn, V = cells
    return {f"m{r}": _run(X, knn, V + rng.normal(scale=sd, size=V.shape))
            for r in range(n_runs)}


def test_agreement_identical_runs_agree_fully(cells):
    pytest.importorskip("scvelo")
    run = _run(*cells)
    res = M.agreement({"a": run, "b": run.copy(), "c": run.copy()}, min_methods=3)
    for r in res.values():
        assert r.status == "ok"
        assert r.value == pytest.approx(1.0)
        assert all(v == pytest.approx(1.0) for v in r.per_group.values())
        assert r.detail == "3 methods; median neighbour overlap 1.00"


def test_agreement_singles_out_the_reversed_method(cells, rng):
    pytest.importorskip("scvelo")
    X, knn, V = cells
    runs = _noisy_runs(cells, rng)
    runs["reversed"] = _run(X, knn, -V)
    res = M.agreement(runs)
    others = [res[m].value for m in runs if m != "reversed"]
    assert res["reversed"].value < min(others) - 0.3
    assert set(res["reversed"].per_group) == set(runs) - {"reversed"}

    shuffled = M.agreement(dict(reversed(list(runs.items()))))
    for m in runs:
        np.testing.assert_allclose(shuffled[m].per_cell, res[m].per_cell)


def test_agreement_is_cz_biohubs_dense_computation(cells, rng):
    scv = pytest.importorskip("scvelo")
    runs = _noisy_runs(cells, rng, n_runs=5)
    res = M.agreement(runs)

    dense = np.stack([
        scv.utils.get_transition_matrix(a, vgraph=a.uns["velocity_graph"]).toarray()
        for a in runs.values()
    ])
    median = np.median(dense, axis=0)
    for k, m in enumerate(runs):
        cos = (dense[k] * median).sum(1) / (
            np.linalg.norm(dense[k], axis=1) * np.linalg.norm(median, axis=1)
        )
        np.testing.assert_allclose(res[m].per_cell, cos, atol=1e-10)


def test_agreement_refuses_different_neighbour_graphs(cells, rng):
    pytest.importorskip("scvelo")
    X, knn, V = cells
    runs = _noisy_runs(cells, rng)
    offsets = np.array([rng.choice(np.arange(1, 200), 12, replace=False) for _ in X])
    runs["elsewhere"] = _run(X, (np.arange(200)[:, None] + offsets) % 200, V)
    res = M.agreement(runs)
    assert {r.status for r in res.values()} == {"not_applicable"}
    assert "neighbour overlap" in res["m0"].detail


def test_agreement_statuses(cells, rng):
    pytest.importorskip("scvelo")
    runs = _noisy_runs(cells, rng, n_runs=6)

    few = M.agreement({m: runs[m] for m in ["m0", "m1", "m2"]})
    assert {r.status for r in few.values()} == {"not_applicable"}

    renamed = dict(runs)
    renamed["m5"] = runs["m5"].copy()
    renamed["m5"].obs_names = [f"x{i}" for i in range(200)]
    assert {r.status for r in M.agreement(renamed).values()} == {"not_applicable"}

    broken = dict(runs)
    broken["m5"] = runs["m5"].copy()
    del broken["m5"].uns["velocity_graph"]
    res = M.agreement(broken)
    assert res["m5"].status == "missing_input"
    assert {res[m].status for m in runs if m != "m5"} == {"ok"}
    assert "m5" not in res["m0"].per_group


def test_agreement_without_scvelo_is_missing_input(cells, monkeypatch):
    monkeypatch.setitem(sys.modules, "scvelo", None)
    res = M.agreement({"a": _run(*cells)})
    assert res["a"].status == "missing_input"


# --------------------------------------------------------------------------
# Gene-space direction, field constancy, reference mode
# --------------------------------------------------------------------------

def _curve(rng, n=300, g=40, arc=0.0, noise=0.0):
    """Cells along a curve in gene space, velocity its tangent, three clusters in t.

    ``arc=0`` is a straight line; otherwise the curve turns by *arc* radians
    in a plane spanned by two random gene directions.
    """
    t = np.sort(rng.uniform(0, 1, n))
    d1, d2 = np.linalg.qr(rng.normal(size=(g, 2)))[0].T
    if arc == 0:
        X = np.outer(t, d1)
        V = np.tile(d1, (n, 1))
    else:
        a = arc * t
        X = (np.outer(np.sin(a), d1) + np.outer(1 - np.cos(a), d2)) / arc
        V = np.outer(np.cos(a), d1) + np.outer(np.sin(a), d2)
    X = X + 3.0 + noise * rng.normal(size=X.shape)
    labels = np.array(["A", "B", "C"])[np.minimum((t * 3).astype(int), 2)]
    return t, X.astype(np.float32), V, labels


def _gene_run(X, V, labels, n_neighbors=15):
    """A method with a gene-space velocity and scVelo's own graph on its ``Ms``."""
    import anndata as ad
    import scanpy as sc
    import scvelo as scv

    n, g = X.shape
    a = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame({"clusters": pd.Categorical(labels)},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(g)]),
    )
    a.layers["Ms"] = X.copy()
    a.layers["velocity"] = np.asarray(V, dtype=np.float64)
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X")
    scv.tl.velocity_graph(a, sqrt_transform=False, n_jobs=1)
    ve.build_neighbor_indices(a)
    return a


EDGES3 = [["A", "B"], ["B", "C"]]


def test_cbdir_gene_space_straight_line_is_one(rng):
    pytest.importorskip("scvelo")
    _, X, V, lab = _curve(rng)
    res = M.cbdir(_gene_run(X, V, lab), label_key="clusters",
                  cluster_edges=EDGES3, basis=None)
    assert res.status == "ok"
    assert res.value == pytest.approx(1.0, abs=1e-5)
    assert res.detail.startswith("gene space; 0 of ")


def test_cbdir_gene_space_arc_reverse_and_scale(rng):
    pytest.importorskip("scvelo")
    _, X, V, lab = _curve(rng, arc=np.pi / 2)
    arc = M.cbdir(_gene_run(X, V, lab), label_key="clusters",
                  cluster_edges=EDGES3, basis=None).value
    assert arc > 0.99

    _, X, V, lab = _curve(np.random.default_rng(1), arc=np.pi / 2, noise=0.002)
    kw = dict(label_key="clusters", cluster_edges=EDGES3, basis=None)
    noisy = M.cbdir(_gene_run(X, V, lab), **kw).value
    assert 0.2 < noisy < 0.95
    rev = M.cbdir(_gene_run(X, -V, lab), **kw).value
    assert rev == pytest.approx(-noisy, abs=1e-6)
    big = M.cbdir(_gene_run(X, 1000 * V, lab), **kw).value
    assert big == pytest.approx(noisy, abs=1e-5)

    rand = M.cbdir(_gene_run(X, rng.normal(size=V.shape), lab), **kw).value
    assert abs(rand) < 0.1


def test_cbdir_gene_space_statuses(rng):
    pytest.importorskip("scvelo")
    _, X, V, lab = _curve(rng)
    a = _gene_run(X, V, lab)
    kw = dict(label_key="clusters", cluster_edges=EDGES3, basis=None)

    from tests.conftest import knn_indices

    other = a.copy()
    elsewhere = np.random.default_rng(5).normal(size=(300, 2))
    other.obsm[ve.KNN_KEY] = knn_indices(elsewhere, 15)
    res = M.cbdir(other, **kw)
    assert res.status == "not_applicable" and "reference" in res.detail

    latent = a.copy()
    ve.set_velocity_space(latent, "latent")
    assert M.cbdir(latent, **kw).status == "not_applicable"


def _constancy(V):
    import anndata as ad

    a = ad.AnnData(X=np.zeros(np.shape(V), np.float32))
    a.layers["velocity"] = np.asarray(V, dtype=np.float64)
    return M.field_constancy(a)


def test_field_constancy(rng):
    n, g = 800, 40
    v = rng.normal(size=g)
    assert np.isclose(_constancy(np.tile(v, (n, 1))).value, 1.0)
    assert _constancy(rng.normal(size=(n, g))).value < 3 / np.sqrt(n)

    _, _, V, _ = _curve(rng, n=n, g=g, arc=np.pi / 2)
    arc = _constancy(V)
    assert 0.85 < arc.value < 0.95
    for W in (-V, 7 * V):
        assert _constancy(W).value == pytest.approx(arc.value, abs=1e-12)
    assert np.nanmax(arc.per_cell) <= 1 + 1e-12

    half = np.vstack([np.tile(v, (n // 2, 1)), np.tile(-v, (n // 2, 1))])
    assert _constancy(half).value == pytest.approx(0.0, abs=1e-12)

    still = np.zeros((n, g))
    still[0] = v
    assert _constancy(still).status == "not_applicable"


@pytest.fixture
def reference(rng):
    """A shared reference: PCA, UMAP, scanpy neighbours and Ms for 300 cells."""
    pytest.importorskip("scvelo")
    import anndata as ad
    import scanpy as sc

    _, X, V, lab = _curve(rng, arc=np.pi / 2, noise=0.01)
    ref = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame({"clusters": pd.Categorical(lab)},
                         index=[f"c{i}" for i in range(len(X))]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]),
    )
    ref.layers["Ms"] = X.copy()
    Xc = X - X.mean(0)
    pcs = np.linalg.svd(Xc, full_matrices=False)[2][:10]
    ref.obsm["X_pca"] = (Xc @ pcs.T).astype(np.float32)
    ref.obsm["X_umap"] = ref.obsm["X_pca"][:, :2].astype(np.float64)
    sc.pp.neighbors(ref, n_neighbors=15, use_rep="X_pca")
    return ref, X, V


def _method(ref, X, V, seed, n_neighbors):
    """Same velocity, but the method's own neighbours, UMAP, Ms and velocity graph."""
    r = np.random.default_rng(seed)
    a = _gene_run(X + 0.05 * r.normal(size=X.shape).astype(np.float32), V,
                  ref.obs["clusters"].to_numpy(), n_neighbors=n_neighbors)
    a.obsm["X_umap"] = r.normal(size=(a.n_obs, 2))
    a.obsm["velocity_umap"] = r.normal(size=(a.n_obs, 2))
    return a


def _graph(a, neg=False):
    from veloeval.access import get_velocity_graph

    return get_velocity_graph(a, negative=neg).toarray()


def test_reference_mode_puts_methods_on_one_footing(reference):
    ref, X, V = reference
    m1, m2 = _method(ref, X, V, 1, 8), _method(ref, X, V, 2, 25)
    assert not np.array_equal(_graph(m1), _graph(m2))
    for m in (m1, m2):
        ve.prepare(m, reference=ref)
    np.testing.assert_array_equal(m1.obsm[ve.KNN_KEY], m2.obsm[ve.KNN_KEY])
    np.testing.assert_array_equal(_graph(m1), _graph(m2))
    np.testing.assert_array_equal(_graph(m1, True), _graph(m2, True))
    np.testing.assert_array_equal(m1.obsm["velocity_umap"], m2.obsm["velocity_umap"])
    np.testing.assert_array_equal(m1.obsm["X_umap"], ref.obsm["X_umap"])

    rec = m1.uns["veloeval"]["prepared"]
    assert rec["neighbors"]["source"] == "reference"
    assert {k: v for k, v in rec["reference"].items() if k != "removed"} == {
        "n_obs": 300, "dropped": 0, "graph": "copied",
    }
    assert rec["velocity_graph"]["n_genes"] == 40
    assert "veloeval_Ms" not in m1.layers


def test_reference_mode_is_order_free(reference):
    ref, X, V = reference
    m = _method(ref, X, V, 1, 8)
    order = np.random.default_rng(3).permutation(m.n_obs)
    shuffled = m[order].copy()
    ve.prepare(shuffled, reference=ref)
    base = ref.copy()
    ve.build_neighbor_indices(base)
    names = lambda a: a.obs_names.to_numpy()[a.obsm[ve.KNN_KEY]]  # noqa: E731
    got = pd.DataFrame(names(shuffled), index=shuffled.obs_names).loc[ref.obs_names]
    np.testing.assert_array_equal(got.to_numpy(), names(base))


def test_reference_mode_rebuilds_the_graph_when_cells_are_dropped(reference):
    import scanpy as sc

    ref, X, V = reference
    keep = np.sort(np.random.default_rng(4).choice(300, 270, replace=False))
    m = _method(ref, X, V, 1, 8)[keep].copy()
    ve.prepare(m, reference=ref)
    knn = m.obsm[ve.KNN_KEY]
    assert knn.shape == (270, 14) and knn.max() < 270
    assert m.uns["veloeval"]["prepared"]["reference"]["dropped"] == 30

    sub = ref[keep].copy()
    p = sub.uns["neighbors"]["params"]
    sc.pp.neighbors(
        sub, n_neighbors=p["n_neighbors"], use_rep="X_pca", method=p["method"],
        metric=p["metric"], random_state=p["random_state"],
    )
    np.testing.assert_array_equal(knn, ve.build_neighbor_indices(sub))


def test_reference_mode_genes_and_reversal(reference):
    import anndata as ad
    import scvelo as scv

    ref, X, V = reference
    m = _method(ref, X, V, 1, 8)
    extra = ad.AnnData(
        X=np.ones((300, 5), np.float32), obs=m.obs[[]],
        var=pd.DataFrame(index=[f"x{j}" for j in range(5)]),
    )
    extra.layers["Ms"] = extra.X.copy()
    extra.layers["velocity"] = np.ones((300, 5))
    wide = ad.concat([m[:, 10:], extra], axis=1, merge="first")
    wide.obs = m.obs.copy()
    wide.obsp = m.obsp
    wide.uns = dict(m.uns)
    wide.obsm = m.obsm.copy()
    wide.layers["velocity"][:, 0] = np.nan
    ve.prepare(wide, reference=ref)
    assert wide.uns["veloeval"]["prepared"]["velocity_graph"]["n_genes"] == 29

    genes = [f"g{j}" for j in range(11, 40)]
    direct = ref[:, genes].copy()
    direct.layers["velocity"] = V[:, 11:]
    scv.tl.velocity_graph(direct, xkey="Ms", sqrt_transform=False, n_jobs=1)
    np.testing.assert_allclose(_graph(wide), _graph(direct), atol=1e-6)

    rev = _method(ref, X, -V, 1, 8)
    fwd = _method(ref, X, V, 1, 8)
    for m_ in (rev, fwd):
        ve.prepare(m_, reference=ref)
    np.testing.assert_array_equal(_graph(rev), -_graph(fwd, True))
    np.testing.assert_array_equal(_graph(rev, True), -_graph(fwd))


def test_reference_mode_errors(reference):
    ref, X, V = reference
    m = _method(ref, X, V, 1, 8)
    stranger = m.copy()
    stranger.obs_names = ["zz"] + list(stranger.obs_names[1:])
    with pytest.raises(ValueError, match="not in the reference"):
        ve.prepare(stranger, reference=ref)

    no_umap = ref.copy()
    del no_umap.obsm["X_umap"]
    with pytest.raises(ValueError, match="X_umap"):
        ve.prepare(m.copy(), reference=no_umap)

    with pytest.raises(ValueError, match="overwrite_neighbors"):
        ve.prepare(m.copy(), reference=ref, overwrite_neighbors=True)


def test_reference_mode_clears_the_method_s_own_derivations(reference):
    ref, X, V = reference
    ref = ref.copy()
    ref.obsm["X_tsne"] = np.random.default_rng(5).normal(size=(300, 2))
    m = _method(ref, X, V, 1, 8)
    m.obs["velocity_pseudotime"] = 0.5
    m.obs["latent_time"] = 0.25
    m.obsp["T_fwd"] = np.eye(300)
    m.obsm["X_tsne"] = np.zeros((300, 2))
    m.obsm["X_own"] = np.zeros((300, 3))
    m.obsm["velocity_tsne"] = np.ones((300, 2))
    ve.prepare(m, reference=ref, pseudotime=True, transition=True)

    assert m.obs["velocity_pseudotime"].nunique() > 1
    assert (m.obs["latent_time"] == 0.25).all()
    assert not np.allclose(m.obsp["T_fwd"].toarray(), np.eye(300))
    np.testing.assert_array_equal(m.obsm["X_tsne"], ref.obsm["X_tsne"])
    assert "X_own" not in m.obsm and "velocity_tsne" not in m.obsm
    removed = m.uns["veloeval"]["prepared"]["reference"]["removed"]
    for key in ("obs['velocity_pseudotime']", "obsp['T_fwd']", "obsm['X_tsne']",
                "obsm['velocity_tsne']", "uns['velocity_graph']", "uns['neighbors']"):
        assert key in removed
    assert "obs['latent_time']" not in removed


def test_reference_mode_records_a_failed_velocity_graph(reference, monkeypatch):
    import scvelo as scv

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    ref, X, V = reference
    m = _method(ref, X, V, 1, 8)
    monkeypatch.setattr(scv.tl, "velocity_graph", boom)
    ve.prepare(m, reference=ref, transition=True, pseudotime=True)
    rec = m.uns["veloeval"]["prepared"]
    assert rec["velocity_graph"] == {"error": "RuntimeError: boom"}
    for key in ("velocity_umap", "T_fwd", "velocity_pseudotime"):
        assert "failed" in rec[key]["skipped"]
    for basis in ("umap", None):
        res = M.cbdir(m, label_key="clusters", cluster_edges=EDGES3, basis=basis)
        assert res.status == "missing_input"


@pytest.mark.parametrize("space", ["embedding", "scalar"])
def test_prepare_builds_no_velocity_graph_outside_gene_and_latent(reference, space):
    ref, X, V = reference
    m = ref.copy()
    m.layers["velocity"] = V.astype(np.float64)
    m.obsm["velocity_umap"] = np.ones((300, 2))
    ve.prepare(m, space=space, transition=True, pseudotime=True)
    rec = m.uns["veloeval"]["prepared"]
    for key in ("velocity_umap", "T_fwd", "velocity_pseudotime"):
        assert rec[key] == {"skipped": f"no velocity graph in {space!r} space"}
    assert "velocity_graph" not in m.uns and "T_fwd" not in m.obsp
    np.testing.assert_array_equal(m.obsm["velocity_umap"], 1.0)


def test_reference_mode_rebuilds_on_the_reference_s_use_rep(reference):
    import scanpy as sc

    ref, X, V = reference
    ref = ref.copy()
    ref.obsm["X_other"] = np.random.default_rng(6).normal(size=(300, 5))
    del ref.obsm["X_pca"]
    sc.pp.neighbors(ref, n_neighbors=12, use_rep="X_other")
    keep = np.sort(np.random.default_rng(4).choice(300, 270, replace=False))
    m = _method(ref, X, V, 1, 8)[keep].copy()
    ve.prepare(m, reference=ref)
    assert m.uns["veloeval"]["prepared"]["reference"]["graph"] == (
        "rebuilt on reference X_other"
    )
    sub = ref[keep].copy()
    p = sub.uns["neighbors"]["params"]
    sc.pp.neighbors(
        sub, n_neighbors=p["n_neighbors"], use_rep="X_other", method=p["method"],
        metric=p["metric"], random_state=p["random_state"],
    )
    np.testing.assert_array_equal(m.obsm[ve.KNN_KEY], ve.build_neighbor_indices(sub))

    del ref.obsm["X_other"]
    with pytest.raises(ValueError, match="X_other"):
        ve.prepare(_method(ref, X, V, 1, 8), reference=ref)


def test_reference_mode_reads_one_gene_set(reference, monkeypatch):
    import scvelo as scv

    ref, X, V = reference
    widths = []
    graph = scv.tl.velocity_graph

    def spy(adata, *args, **kwargs):
        widths.append(adata.n_vars)
        return graph(adata, *args, **kwargs)

    def scores(m):
        kw = dict(label_key="clusters", cluster_edges=EDGES3, basis=None)
        return [
            M.field_constancy(m).value,
            M.icvcoh(m, label_key="clusters", basis=None).value,
            M.velocity_consistency(m).value,
            M.cbdir(m, **kw).value,
        ]

    runs = []
    for wild in (False, True):
        m = _method(ref, X, V, 1, 8)
        m.var["velocity_genes"] = m.var_names != "g0"
        if wild:
            m.layers["velocity"][:, 0] = 1e3
        monkeypatch.setattr(scv.tl, "velocity_graph", spy)
        ve.prepare(m, reference=ref)
        monkeypatch.undo()
        assert m.var["veloeval_genes"].sum() == 39 and not m.var["veloeval_genes"]["g0"]
        assert "veloeval_Ms" not in m.layers
        runs.append(scores(m))
    assert widths == [39, 39]
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-6)
