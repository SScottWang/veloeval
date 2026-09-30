"""Agreement between inferred time and an experimentally measured time axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .._math import fisher_lee, spearman, uniform_scores
from ..access import get_stages
from ..result import MetricResult, NotApplicable, metric

__all__ = ["tsc", "phase_corr"]


@metric
def tsc(adata, *, time_key: str, true_time_key: str):
    """Temporal Spearman correlation.

    Rank correlation between the method's inferred per-cell time and a real
    measured time axis -- collection stage, metabolic labelling duration.

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

    Notes
    -----
    Detects a reversed field: reversing the inferred time flips the sign.

    Assumes a **linear** time axis.  On a cyclic one (a FUCCI position) the
    Spearman rho depends on where the cycle was cut: with evenly spread phases a
    perfectly correct ordering cut at mid-cycle scores about -0.5, and a
    reversed one scores above 0.  Use :func:`phase_corr` there.

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


@metric
def phase_corr(
    adata, *, time_key: str = "latent_time", phase_key: str = "fucci_phase"
):
    r"""Circular correlation between inferred time and a measured cell-cycle phase.

    The cyclic counterpart of :func:`tsc`.  Both sides are replaced by uniform
    scores, :math:`\alpha_i = 2\pi (r_i - 1)/n` with :math:`r_i` the rank of
    cell :math:`i`, and correlated with Fisher & Lee's circular correlation
    (Fisher & Lee, *Biometrika* 1983; the rank form, *Biometrika* 1982):

    .. math::

        \rho = \frac{\sum_{i<j} \sin(\alpha_i-\alpha_j)\sin(\beta_i-\beta_j)}
                    {\sqrt{\sum_{i<j}\sin^2(\alpha_i-\alpha_j)
                            \sum_{i<j}\sin^2(\beta_i-\beta_j)}}

    where :math:`\alpha` comes from the inferred time and :math:`\beta` from
    the phase.  Only the cyclic order of each side matters, as only the linear
    order does for Spearman.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry both columns in ``obs``.
    time_key : str, default: "latent_time"
        Column holding the method's inferred time -- a linear time or a cyclic
        phase (VeloCycle's), since only ranks are used.
    phase_key : str, default: "fucci_phase"
        Column holding the measured position within **one** cycle (not an
        unwrapped cumulative value), increasing as the cycle advances.  Origin
        and unit are free.

    Returns
    -------
    MetricResult
        :math:`\rho` over the cells with both values; ``detail`` gives their
        number.  ``not_applicable`` when either column is missing, fewer than
        10 cells remain, or either side takes fewer than three distinct values.

    Notes
    -----
    Detects a reversed field: reversing either side flips the sign.

    Unchanged by rotating either side, so where a linear time cut the cycle
    does not matter: a method that cuts it somewhere and unrolls it in the
    right order -- the best a linear time can do on a cycle -- scores 1.

    For the same disorder the value runs well below a Spearman rho: with rank
    errors of a tenth of a cycle, :math:`\rho \approx 0.73` where :func:`tsc`
    gives 0.95.  Compare methods with each other on this metric, not its value
    with a ``tsc`` value.

    :func:`~veloeval.metrics.phase_dir` asks whether the *velocity* follows the
    phase gradient; this asks whether the inferred *time* orders the cycle.

    Examples
    --------
    .. code-block:: python

        from veloeval import metrics as M

        # relative FUCCI position on [0, 1), e.g. dynamo's scEU-seq RPE1
        adata.obs["fucci_phase"] = adata.obs["Cell_cycle_relativePos"]
        res = M.phase_corr(adata, time_key="latent_time")
        res.value
        res.detail  # number of cells scored
    """
    if time_key not in adata.obs:
        raise NotApplicable(f"method infers no obs['{time_key}']")
    if phase_key not in adata.obs:
        raise NotApplicable(f"dataset has no cell-cycle phase obs['{phase_key}']")

    t = adata.obs[time_key].to_numpy(dtype=np.float64)
    phase = adata.obs[phase_key].to_numpy(dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(phase)
    t, phase = t[keep], phase[keep]
    n = int(keep.sum())
    if n < 10:
        raise NotApplicable(f"only {n} cells carry both values")
    if np.unique(phase).size < 3:
        raise NotApplicable(f"obs['{phase_key}'] carries no cyclic ordering")
    if np.unique(t).size < 3:
        raise NotApplicable("inferred time carries no cyclic ordering")

    rho = fisher_lee(uniform_scores(t), uniform_scores(phase))
    if np.isnan(rho):
        raise NotApplicable("inferred time carries no cyclic ordering")
    return MetricResult(name="phase_corr", value=rho, detail=f"{n} cells")
