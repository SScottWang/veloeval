"""Agreement between inferred time and an experimentally measured time axis."""

from __future__ import annotations

import numpy as np

from .._math import spearman
from ..result import NotApplicable, metric

__all__ = ["tsc"]


@metric
def tsc(adata, *, time_key: str, true_time_key: str):
    """Temporal Spearman correlation.  Higher is better; range [-1, 1].

    Rank correlation between the method's inferred per-cell time
    (``obs[time_key]``) and a real measured time axis (``obs[true_time_key]``)
    -- collection stage, metabolic labelling duration, FUCCI-derived phase.

    Only meaningful where ``true_time_key`` is an *experimental* observable.
    Pointing it at a pseudotime computed from the same velocity field makes
    this circular.
    """
    if time_key not in adata.obs:
        raise NotApplicable(f"method infers no obs['{time_key}']")
    if true_time_key not in adata.obs:
        raise NotApplicable(f"dataset has no measured time axis obs['{true_time_key}']")

    inferred = np.asarray(adata.obs[time_key].values, dtype=np.float64)
    truth = adata.obs[true_time_key].values

    if truth.dtype == object or str(truth.dtype).startswith("category"):
        # Ordered categorical stages ("E7.0" < "E7.25" < ...): rank by sort order.
        codes = adata.obs[true_time_key].astype("category")
        if not codes.cat.ordered:
            codes = codes.cat.as_ordered()
        truth = codes.cat.codes.to_numpy().astype(np.float64)
        truth[truth < 0] = np.nan
    else:
        truth = np.asarray(truth, dtype=np.float64)

    return spearman(inferred, truth)
