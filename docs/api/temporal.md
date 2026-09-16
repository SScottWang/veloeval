# Temporal Metrics

Temporal metrics compare the per-cell time a method infers against a time axis
that was actually measured — collection stage, metabolic labelling duration, a
FUCCI-derived phase.

The measured part is the whole point. Scoring inferred time against a
pseudotime computed from the same velocity field is circular and will look
excellent for every method, including a random one.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.tsc
```
