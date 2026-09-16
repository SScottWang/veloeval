# Result Type and Helpers

## The result type

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
fly. Doing so would give the methods whose wrappers already built one *their*
parameters and everyone else defaults — so two methods would no longer be on the
same footing, and nothing in the results would record which was which.

Call {func}`~veloeval.prepare` once at the end of each method wrapper instead.
It writes what it did, including the scVelo version, into
`adata.uns["veloeval"]["prepared"]`, so the provenance travels with the file.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.prepare
   veloeval.set_velocity_space
```

```{eval-rst}
.. automodule:: veloeval.prepare
   :members: project_velocity, build_neighbor_indices
```

## Accessors

Read-only helpers the metrics use to reach into an `AnnData`. Relevant when
writing a new metric: they raise `MissingInput` or `NotApplicable`, which the
`@metric` decorator turns into the matching status, so a metric body never needs
a `try`.

```{eval-rst}
.. automodule:: veloeval.access
   :members: velocity_space, set_velocity_space, get_velocity, get_velocity_embedding, get_embedding, get_neighbor_indices, get_transition_matrix, get_labels, gene_coverage
```
