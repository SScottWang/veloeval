# Direction Correctness Metrics

Direction metrics score the velocity field against known differentiation steps:
curated `cluster_edges`, a list of `[source, target]` cell-type pairs. They are
the closest thing the field has to an accuracy measure, and they inherit that
list's limits — a dataset with no curated edges returns `not_applicable`, which
is a fact about the dataset rather than a failure of the method.

All three are bounded, higher is better, and are computed in a shared
low-dimensional embedding so that methods whose velocity lives in different
native spaces stay comparable. The basis must be identical across every method
in a comparison.

{func}`~veloeval.metrics.cbdir` also runs without an embedding: `basis=None`
reads the gene-space cosines from scVelo's velocity graph. Its values run far
lower than in UMAP, so record the basis with the result and never pool the two;
build the graphs with `prepare(..., reference=ref)` so every method shares one
neighbourhood.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.cbdir
   veloeval.metrics.cbvcoh
   veloeval.metrics.cto
```
