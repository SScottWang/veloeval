"""Meta-metrics: scored on a results table, not on a single run.

These take the assembled benchmark output rather than an AnnData, which is why
they live apart from everything else.  :func:`rho_rank` is the one that asks the
question the benchmark exists to ask.
"""

from __future__ import annotations

import numpy as np

from .._math import spearman
from ..result import NotApplicable, metric

__all__ = ["rho_rank"]


@metric
def rho_rank(
    results,
    *,
    conventional: str,
    ground_truth: str,
    method_key: str = "method",
    conventional_higher_better: bool = True,
    ground_truth_higher_better: bool = True,
    min_methods: int = 5,
):
    """Rank concordance between a conventional metric and a physical truth.

    Rank the same set of methods twice on the same data -- once by a
    conventional metric (CBDir, ICVCoh, ...), once by a metric scored against
    an independent physical truth (PhaseDir, TruthCos) -- and report the
    Spearman correlation between the two rankings.

    A high value means the conventional metrics are a usable proxy.  A low one
    means the field's standard ranking does not track the physical truth, which
    is a claim about the metrics rather than about any single method.

    Parameters
    ----------
    results
        A ``pandas.DataFrame`` with one row per method and both metric columns.
    conventional, ground_truth
        Column names.
    conventional_higher_better, ground_truth_higher_better
        Set ``False`` for a metric where lower is better, so both sides are
        ranked in the same direction before correlating.

    Notes
    -----
    Report this per dataset.  Pooling methods across datasets mixes together
    runs with different applicable-metric sets, and the correlation stops
    meaning anything.
    """
    for col in (conventional, ground_truth, method_key):
        if col not in results.columns:
            raise NotApplicable(f"results table has no column '{col}'")

    df = results[[method_key, conventional, ground_truth]].dropna()
    if len(df) < min_methods:
        raise NotApplicable(
            f"only {len(df)} methods have both metrics; need >= {min_methods} "
            "for a meaningful rank correlation"
        )

    a = df[conventional].to_numpy(dtype=np.float64)
    b = df[ground_truth].to_numpy(dtype=np.float64)
    if not conventional_higher_better:
        a = -a
    if not ground_truth_higher_better:
        b = -b

    return spearman(a, b)
