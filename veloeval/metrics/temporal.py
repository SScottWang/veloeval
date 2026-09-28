"""Agreement between inferred time and an experimentally measured time axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .._math import spearman
from ..access import get_stages
from ..result import NotApplicable, metric

__all__ = ["tsc"]


@metric
def tsc(adata, *, time_key: str, true_time_key: str):
    """Temporal Spearman correlation.

    Rank correlation between the method's inferred per-cell time and a real
    measured time axis -- collection stage, metabolic labelling duration,
    FUCCI-derived phase.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry both columns in ``obs``.
    time_key : str
        Column holding the method's inferred time.  Absent -> ``not_applicable``.
    true_time_key : str
        Column holding the *measured* time axis.  Numeric values are ranked
        directly; string stages (``"E7.0" < "E7.25" < ...``) must be an ordered
        ``Categorical`` and are ranked by category order -- anything else fails,
        since string sort order is not time order.  Absent -> ``not_applicable``.

    Returns
    -------
    MetricResult
        The Spearman rho, or ``not_applicable`` when either column is missing.

    Warnings
    --------
    Only meaningful when *true_time_key* is an **experimental** observable.
    Pointing it at a pseudotime computed from the same velocity field makes
    the metric circular and it will score near 1 for any method.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from veloeval import metrics as M

        res = M.tsc(adata, time_key="latent_time", true_time_key="day")  # numeric

        # string stages need a declared order
        adata.obs["stage"] = pd.Categorical(
            adata.obs["stage"], categories=["E12.5", "E13.5", "E14.5"], ordered=True
        )
        M.tsc(adata, time_key="latent_time", true_time_key="stage")
    """
    if time_key not in adata.obs:
        raise NotApplicable(f"method infers no obs['{time_key}']")
    if true_time_key not in adata.obs:
        raise NotApplicable(f"dataset has no measured time axis obs['{true_time_key}']")

    inferred = np.asarray(adata.obs[time_key].values, dtype=np.float64)
    col = adata.obs[true_time_key]
    if is_numeric_dtype(col) and not isinstance(col.dtype, pd.CategoricalDtype):
        truth = col.to_numpy(dtype=np.float64)
    else:
        rank = {s: r for r, s in enumerate(get_stages(adata, true_time_key))}
        truth = np.array([rank.get(v, np.nan) for v in col.values], dtype=np.float64)

    return spearman(inferred, truth)
