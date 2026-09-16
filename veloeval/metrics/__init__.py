"""Metric functions, grouped the way the benchmark groups them.

Single-run metrics take an ``AnnData`` and return a
:class:`~veloeval.result.MetricResult`.  Two metrics do not fit that shape and
say so in their signatures:

* :func:`~veloeval.metrics.negative.mag_ratio` takes two runs of one method
  (negative control, positive control);
* :func:`~veloeval.metrics.meta.rho_rank` takes an assembled results table.
"""

from .coherence import icvcoh, velocity_consistency
from .direction import cbdir, cbvcoh, cto
from .groundtruth import gamma_corr, phase_dir, truth_cos
from .meta import rho_rank
from .negative import ees, mag_ratio, sts
from .temporal import tsc

#: Metric name -> ``"higher"`` / ``"lower"`` / ``"zero"``, for ranking and
#: for colour scales in the plotting layer.  Keep in sync with the framework's
#: metrics table.
DIRECTION = {
    "cbdir": "higher",
    "cbvcoh": "higher",
    "cto": "higher",
    "icvcoh": "higher",
    "velocity_consistency": "higher",
    "tsc": "higher",
    "sts": "higher",
    "ees": "higher",
    "mag_ratio": "zero",
    "phase_dir": "higher",
    "truth_cos": "higher",
    "gamma_corr": "higher",
    "rho_rank": "higher",
}

__all__ = [
    "cbdir",
    "cbvcoh",
    "cto",
    "icvcoh",
    "velocity_consistency",
    "tsc",
    "sts",
    "ees",
    "mag_ratio",
    "phase_dir",
    "truth_cos",
    "gamma_corr",
    "rho_rank",
    "DIRECTION",
]
