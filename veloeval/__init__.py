"""veloeval -- metrics for benchmarking RNA velocity methods.

The library evaluates; it does not run velocity methods and does not own a
workflow.  That split is deliberate: the pipeline changes often, the metric
definitions should not, and only one of the two belongs in a paper's methods
section.

Typical use, per (method, dataset, seed)::

    import anndata as ad
    import veloeval as ve

    adata = ad.read_h5ad("velocity.h5ad")
    results = ve.compute_all(
        adata,
        label_key="clusters",
        cluster_edges=[["Ductal", "Ngn3 low EP"], ...],
    )
    row = ve.to_row(results, method="scvelo_dynamical", dataset="pancreas", seed=42)

Every metric returns a :class:`~veloeval.result.MetricResult` carrying a status,
so "not applicable to this dataset", "upstream field missing" and "crashed"
stay distinguishable all the way into the results table.
"""

from .access import set_velocity_space, velocity_space
from .metrics import DIRECTION
from .prepare import prepare
from .result import MetricResult, Status
from .runner import compute_all, coverage_summary, to_row

__version__ = "0.1.0"

__all__ = [
    "compute_all",
    "to_row",
    "coverage_summary",
    "prepare",
    "MetricResult",
    "Status",
    "DIRECTION",
    "set_velocity_space",
    "velocity_space",
    "__version__",
]
