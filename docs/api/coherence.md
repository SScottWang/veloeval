# Coherence Metrics

Coherence metrics measure how smoothly the velocity field varies between
neighbouring cells. They need no ground truth of any kind, which is what makes
them available everywhere — and also what limits them.

**Coherence is not correctness.** A field that is confidently, smoothly wrong
scores as high as one that is right. Read these next to the direction metrics,
never instead of them: a method that ranks first on coherence and last on
{func}`~veloeval.metrics.cbdir` has produced a convincing artefact.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.icvcoh
   veloeval.metrics.velocity_consistency
```
