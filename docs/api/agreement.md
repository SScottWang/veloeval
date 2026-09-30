# Agreement Metrics

Agreement asks whether the methods on one dataset send each cell to the same
places. It is the one metric that takes several runs at once, because a
consensus is only defined over a set of methods, and it returns one result per
run.

**Agreement is consensus, not correctness.** A single reversed method scores
low; a reversed majority scores high. Every score also depends on which methods
took part, so compare only within one dataset and one method set.

All runs must share one neighbour graph. The metric checks this and returns
`not_applicable` when they do not, since it would otherwise compare the graphs
rather than the velocities.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.agreement
```
