# API reference

Everything below is generated from the docstrings in the source. Change a
docstring, push, and this page changes with it.

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

## Preparing an h5ad

The metrics never derive a velocity graph, a kNN or a transition matrix on the
fly: doing so would give the methods whose wrappers already built one their own
parameters and the rest defaults, and nothing in the results would record
which. Call this once at the end of each method wrapper instead.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   veloeval.prepare
```

```{eval-rst}
.. automodule:: veloeval.prepare
   :members: project_velocity, build_neighbor_indices
```

## Accessors

Read-only helpers the metrics use. Relevant when writing a new metric.

```{eval-rst}
.. automodule:: veloeval.access
   :members: velocity_space, set_velocity_space, get_velocity, get_velocity_embedding, get_embedding, get_neighbor_indices, get_transition_matrix, get_labels, gene_coverage
```
