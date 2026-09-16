"""Run every applicable metric on one velocity result and flatten it to a row."""

from __future__ import annotations

from typing import Any

from . import metrics as M
from .access import gene_coverage
from .result import MetricResult

__all__ = ["compute_all", "to_row", "coverage_summary"]


def compute_all(
    adata,
    *,
    label_key: str | None = None,
    cluster_edges: list | None = None,
    basis: str = "umap",
    vkey: str = "velocity",
    time_key: str = "latent_time",
    true_time_key: str | None = None,
    phase_key: str | None = None,
    tkey: str = "T_fwd",
    reference=None,
) -> dict[str, MetricResult]:
    """Compute all single-run metrics.

    Nothing raises: a metric that cannot run returns a ``MetricResult`` whose
    ``status`` says why.  Pass only what the dataset actually has -- leaving
    ``phase_key=None`` on a dataset without FUCCI gives ``not_applicable``,
    which is the correct entry for that cell of the results table.

    ``reference`` is a second AnnData carrying labelling-derived velocity, for
    :func:`~veloeval.metrics.groundtruth.truth_cos` and
    :func:`~veloeval.metrics.groundtruth.gamma_corr`.

    :func:`~veloeval.metrics.negative.mag_ratio` is not included: it needs a
    second run of the same method on a paired control dataset, which only the
    pipeline knows how to find.  Call it separately.
    """
    out: dict[str, MetricResult] = {}

    out["icvcoh"] = M.icvcoh(adata, label_key=label_key, vkey=vkey)
    out["velocity_consistency"] = M.velocity_consistency(adata, vkey=vkey)

    out["cbdir"] = M.cbdir(
        adata, label_key=label_key, cluster_edges=cluster_edges, basis=basis, vkey=vkey
    )
    out["cbvcoh"] = M.cbvcoh(
        adata, label_key=label_key, cluster_edges=cluster_edges, basis=basis, vkey=vkey
    )
    out["cto"] = M.cto(
        adata, label_key=label_key, cluster_edges=cluster_edges, time_key=time_key
    )

    out["tsc"] = M.tsc(
        adata, time_key=time_key, true_time_key=true_time_key or "__absent__"
    )

    out["sts"] = M.sts(adata, tkey=tkey)
    out["ees"] = M.ees(adata, tkey=tkey)

    out["phase_dir"] = M.phase_dir(
        adata, phase_key=phase_key or "__absent__", basis=basis, vkey=vkey
    )

    if reference is not None:
        out["truth_cos"] = M.truth_cos(adata, reference, vkey=vkey)
        out["gamma_corr"] = M.gamma_corr(adata, reference)
    else:
        from .result import not_applicable

        out["truth_cos"] = not_applicable(
            "truth_cos", "no metabolic-labelling reference for this dataset"
        )
        out["gamma_corr"] = not_applicable(
            "gamma_corr", "no metabolic-labelling reference for this dataset"
        )

    return out


def coverage_summary(adata, vkey: str = "velocity") -> dict[str, Any]:
    """Velocity gene coverage -- context for every other number in the row.

    A method that kept 200 of 2000 genes is not directly comparable to one that
    kept all of them, however good its scores look.
    """
    try:
        valid, total = gene_coverage(adata, vkey)
    except Exception:
        return {
            "velocity_features_valid": None,
            "velocity_features_total": None,
            "velocity_feature_coverage": None,
        }
    return {
        "velocity_features_valid": valid,
        "velocity_features_total": total,
        "velocity_feature_coverage": valid / total if total else None,
    }


def to_row(
    results: dict[str, MetricResult],
    **context: Any,
) -> dict[str, Any]:
    """Flatten results to one CSV row: ``<name>`` and ``<name>_status`` columns.

    The ``veloeval_version`` column is written automatically.  Do not drop it:
    it is what lets you tell results computed before a metric was fixed from
    results computed after, which is otherwise invisible in the table.
    """
    from . import __version__

    row: dict[str, Any] = dict(context)
    row["veloeval_version"] = __version__
    for name, res in results.items():
        row[name] = res.value
        row[f"{name}_status"] = res.status
        if res.status != "ok" and res.detail:
            row[f"{name}_detail"] = res.detail
    return row
