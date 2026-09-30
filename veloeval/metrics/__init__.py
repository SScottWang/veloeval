"""Metric functions, grouped the way the benchmark groups them.

Each takes an ``AnnData`` and returns a :class:`~veloeval.MetricResult`.  The
two that need a labelling reference (:func:`truth_cos`, :func:`gamma_corr`)
take a second ``AnnData``; :func:`agreement` takes every run on a dataset and
returns one result per run; everything else is one run in, one result out.
"""

from .agreement import agreement
from .coherence import (
    icvcoh,
    spatial_consistency,
    time_morans_i,
    velocity_consistency,
)
from .direction import cbdir, cbvcoh, cto
from .groundtruth import gamma_corr, lineage_fate, phase_dir, rate_err, truth_cos
from .negative import ees, sts, sts_abs
from .temporal import phase_corr, tsc

#: Metric name -> ``"higher"`` / ``"lower"`` / ``"zero"``.  What "better" means
#: for each metric, for ranking and for colour scales in a plotting layer.
DIRECTION = {
    "cbdir": "higher",
    "cbvcoh": "higher",
    "cto": "higher",
    "icvcoh": "higher",
    "velocity_consistency": "higher",
    "spatial_consistency": "higher",
    "time_morans_i": "higher",
    "tsc": "higher",
    "phase_corr": "higher",
    "sts": "higher",
    "sts_abs": "higher",
    "ees": "higher",
    "phase_dir": "higher",
    "truth_cos": "higher",
    "gamma_corr": "higher",
    "lineage_fate": "higher",
    "rate_err": "lower",
    "agreement": "higher",
}

__all__ = [
    "cbdir",
    "cbvcoh",
    "cto",
    "icvcoh",
    "velocity_consistency",
    "spatial_consistency",
    "time_morans_i",
    "tsc",
    "phase_corr",
    "sts",
    "sts_abs",
    "ees",
    "phase_dir",
    "truth_cos",
    "gamma_corr",
    "lineage_fate",
    "rate_err",
    "agreement",
    "DIRECTION",
]
