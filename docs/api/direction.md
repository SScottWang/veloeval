# Direction Correctness Metrics

Direction metrics score the velocity field against known differentiation steps:
curated `cluster_edges`, a list of `[source, target]` cell-type pairs. They are
the closest thing the field has to an accuracy measure, and they inherit that
list's limits — a dataset with no curated edges returns `not_applicable`, which
is a fact about the dataset rather than a failure of the method.

CBDir, CBVCoh and CTO are bounded, higher is better, and are computed in a
shared low-dimensional embedding so that methods whose velocity lives in
different native spaces stay comparable. The basis must be identical across
every method in a comparison.

{func}`~veloeval.metrics.cbdir` also runs without an embedding: `basis=None`
reads the gene-space cosines from scVelo's velocity graph. Its values run far
lower than in UMAP, so record the basis with the result and never pool the two;
build the graphs with `prepare(..., reference=ref)` so every method shares one
neighbourhood.

A field can score well on these from one global direction alone: a constant
field such as {func}`~veloeval.constvelo` does, with no local information.
{func}`~veloeval.metrics.cbturn` keeps only what changes from edge to edge — the
mean boundary velocity of each edge, minus the mean over edges, against the
edge's own direction minus theirs — so every constant field scores 0 and a
reversed one the negative. It reads gene or latent space, never an embedding,
whose projection turns the velocity along the data by itself, and it takes the
reference so that every method is scored against the same edge directions:

```python
ve.prepare(adata, reference=ref)
M.cbturn(adata, ref, label_key="clusters", cluster_edges=edges)
```

On pancreas (7 edges, scvelo 0.3.2):

| velocity | CBDir (gene) | CBTurn | shuffled q97.5 |
| --- | --- | --- | --- |
| scVelo dynamical | 0.173 | 0.207 | 0.023 |
| scVelo deterministic | 0.093 | 0.060 | 0.006 |
| scVelo stochastic | 0.078 | 0.045 | 0.007 |
| ConstVelo, Ductal to Beta | 0.139 | 0.000 | 0.000 |
| reversed dynamical | −0.173 | −0.207 | 0.023 |

CBTurn is a slope, not bounded by 1; compare methods within a dataset.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.cbdir
   veloeval.metrics.cbvcoh
   veloeval.metrics.cbturn
   veloeval.metrics.cto
```
