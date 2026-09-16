# API reference

Everything below is generated from the docstrings in the source. Change a
docstring, push, and this page changes with it.

## Running the metrics

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.compute_all
   veloeval.to_row
   veloeval.coverage_summary
   veloeval.prepare
```

## Direction correctness

Needs curated `cluster_edges`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.metrics.cbdir
   veloeval.metrics.cbvcoh
   veloeval.metrics.cto
```

## Coherence

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.metrics.icvcoh
   veloeval.metrics.velocity_consistency
```

## Temporal

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.metrics.tsc
```

## Negative controls

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.metrics.sts
   veloeval.metrics.ees
   veloeval.metrics.mag_ratio
```

## Independent ground truth

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.metrics.phase_dir
   veloeval.metrics.truth_cos
   veloeval.metrics.gamma_corr
```

## Meta

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.metrics.rho_rank
```

## Result type

```{eval-rst}
.. autoclass:: veloeval.MetricResult
   :members:
   :undoc-members:
```

```{eval-rst}
.. automodule:: veloeval.result
   :members: metric, MissingInput, NotApplicable
```

## Accessors

Read-only helpers the metrics use. Relevant when writing a new metric.

```{eval-rst}
.. automodule:: veloeval.access
   :members: velocity_space, set_velocity_space, get_velocity, get_velocity_embedding, get_embedding, get_neighbor_indices, get_transition_matrix, get_labels, gene_coverage
```
