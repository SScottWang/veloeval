# Temporal Metrics

Temporal metrics compare the per-cell time a method infers against a time axis
that was actually measured — collection stage, metabolic labelling duration, a
FUCCI-derived phase.

A linear time axis and a cyclic one need different statistics. {func}`~veloeval.metrics.tsc`
is a Spearman correlation and assumes the axis has two ends; on a cell cycle it
depends on where the method happened to cut the cycle. {func}`~veloeval.metrics.phase_corr`
is a circular correlation and does not.

The measured part is the whole point. Scoring inferred time against a
pseudotime computed from the same velocity field is circular and will look
excellent for every method, including a random one.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.tsc
   veloeval.metrics.phase_corr
```
