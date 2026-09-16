"""veloeval -- metrics for benchmarking RNA velocity methods.

The library implements metrics and nothing else.  It does not run velocity
methods, does not own a workflow, and does not orchestrate a run: assembling
rows, aggregating seeds and recording runtimes belong to the pipeline, which
changes far more often than a metric definition should.

Each metric takes an ``AnnData`` and returns a :class:`~veloeval.MetricResult`
carrying a status, so "not applicable to this dataset", "upstream field
missing" and "crashed" stay distinguishable all the way into the results
table::

    import anndata as ad
    from veloeval.metrics import cbdir, icvcoh, velocity_consistency

    adata = ad.read_h5ad("2.velocity/pancreas/scvelo_dynamical/seed_42/velocity.h5ad")

    r = cbdir(
        adata,
        label_key="clusters",
        cluster_edges=[["Ductal", "Ngn3 low EP"], ["Ngn3 low EP", "Ngn3 high EP"]],
    )
    r.status    # "ok" | "not_applicable" | "missing_input" | "failed"
    r.value     # 0.43
    r.per_cell  # array, nan where the cell could not be scored

Record :data:`__version__` alongside the numbers.  Without it a table computed
before a metric was fixed is indistinguishable from one computed after.
"""

from .access import set_velocity_space, velocity_space
from .metrics import DIRECTION
from .prepare import prepare
from .result import MetricResult, MissingInput, NotApplicable, Status

__version__ = "0.0.1"

__all__ = [
    "MetricResult",
    "Status",
    "MissingInput",
    "NotApplicable",
    "DIRECTION",
    "prepare",
    "set_velocity_space",
    "velocity_space",
    "__version__",
]
