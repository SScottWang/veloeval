"""Metric functions, grouped the way the benchmark groups them.

Each takes an ``AnnData`` and returns a :class:`~veloeval.MetricResult`.  The
two that need a labelling reference (:func:`truth_cos`, :func:`gamma_corr`)
take a second ``AnnData``; everything else is one run in, one result out.
"""

from .coherence import icvcoh, velocity_consistency
from .direction import cbdir, cbvcoh, cto
from .groundtruth import gamma_corr, phase_dir, truth_cos
from .negative import ees, sts
from .temporal import tsc

#: Metric name -> ``"higher"`` / ``"lower"`` / ``"zero"``.  What "better" means
#: for each metric, for ranking and for colour scales in a plotting layer.
DIRECTION = {
    "cbdir": "higher",
    "cbvcoh": "higher",
    "cto": "higher",
    "icvcoh": "higher",
    "velocity_consistency": "higher",
    "tsc": "higher",
    "sts": "higher",
    "ees": "higher",
    "phase_dir": "higher",
    "truth_cos": "higher",
    "gamma_corr": "higher",
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
    "phase_dir",
    "truth_cos",
    "gamma_corr",
    "DIRECTION",
]
