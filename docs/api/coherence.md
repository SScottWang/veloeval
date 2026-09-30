# Coherence Metrics

Coherence metrics measure how smoothly the velocity field varies between
neighbouring cells. They need no ground truth of any kind, which is what makes
them available everywhere — and also what limits them.

**Coherence is not correctness.** A field that is confidently, smoothly wrong
scores as high as one that is right. Read these next to the direction metrics,
never instead of them: a method that ranks first on coherence and last on
{func}`~veloeval.metrics.cbdir` has produced a convincing artefact. None of
them can tell a field from its reverse.

The two spatial metrics measure the same kind of smoothness over *physical*
neighbours, read from `obsm['veloeval_spatial_knn']` (build it with
{func}`veloeval.build_spatial_neighbors`, or `prepare(..., spatial_key="spatial")`).
Both come from TopoVelo, and methods that smooth over space in their model —
TopoVelo, spVelo — are favoured by construction. Score correctness on spatial
data with a known lineage with {func}`~veloeval.metrics.cbdir` and
{func}`~veloeval.metrics.cto`.

{func}`~veloeval.metrics.field_constancy` is a diagnostic rather than a score:
how close the field is to one constant direction. A method far more constant
than the others on the same dataset, with high coherence, has probably
collapsed to a constant vector, and its coherence is then no evidence of
quality. `DIRECTION` maps it to `None`.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.icvcoh
   veloeval.metrics.velocity_consistency
   veloeval.metrics.spatial_consistency
   veloeval.metrics.time_morans_i
   veloeval.metrics.field_constancy
```
