"""Sign, scale and status behaviour of every metric on known-answer inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import veloeval as ve
from veloeval import metrics as M

from .conftest import make_adata


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
    V = rng.normal(size=(noisy.n_obs, 2))
    noisy.layers["velocity"] = V
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
    near_origin = (cycle.obs["fucci_phase"].to_numpy() < 0.2) | (
        cycle.obs["fucci_phase"].to_numpy() > 2 * np.pi - 0.2
    )
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
    assert M.tsc(linear, time_key="latent_time", true_time_key="nope").status == "not_applicable"


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


def test_mag_ratio(linear):
    pos = linear.copy()
    neg = linear.copy()
    neg.layers["velocity"] = neg.layers["velocity"] * 0.01
    assert M.mag_ratio(neg, pos).value == pytest.approx(0.01, abs=1e-9)


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------

def test_rho_rank():
    df = pd.DataFrame(
        {
            "method": list("abcdef"),
            "cbdir": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "phase_dir": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    res = M.rho_rank(df, conventional="cbdir", ground_truth="phase_dir")
    assert res.value == pytest.approx(-1.0, abs=1e-9)

    small = df.head(3)
    assert M.rho_rank(small, conventional="cbdir", ground_truth="phase_dir").status == (
        "not_applicable"
    )


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
    corrupt.uns["neighbors"] = {"indices": "not an array"}
    assert M.cbdir(corrupt, label_key="clusters", cluster_edges=edges).status == "failed"

    # ok
    assert M.cbdir(linear, label_key="clusters", cluster_edges=edges).status == "ok"


def test_not_applicable_without_labels(linear):
    assert M.icvcoh(linear, label_key=None).status == "not_applicable"


def test_compute_all_and_to_row(linear, edges):
    linear.obs["latent_time"] = np.linspace(0, 1, linear.n_obs)
    results = ve.compute_all(linear, label_key="clusters", cluster_edges=edges)
    row = ve.to_row(results, method="demo", dataset="linear", seed=42)

    assert row["method"] == "demo"
    assert row["veloeval_version"] == ve.__version__
    assert row["cbdir"] == pytest.approx(1.0, abs=1e-6)
    assert row["cbdir_status"] == "ok"
    # no FUCCI, no transition matrix, no labelling reference on this fixture
    assert row["phase_dir_status"] == "not_applicable"
    assert row["sts_status"] == "missing_input"
    assert row["truth_cos_status"] == "not_applicable"


def test_no_metric_ever_raises(linear):
    """An empty AnnData must produce statuses, not a traceback."""
    empty = make_adata(np.zeros((3, 2)), np.zeros((3, 2)))
    del empty.layers["velocity"]
    del empty.obsm["velocity_umap"]
    results = ve.compute_all(empty)
    assert all(r.status != "ok" for r in results.values())
