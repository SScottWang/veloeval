# Negative Control Metrics

Every other metric asks "is the arrow pointing the right way", which can only be
asked on data where there *is* a right way. Negative controls ask the opposite
question: on a population with no ongoing differentiation, does the method have
the good sense to stay quiet?

A trustworthy method gives a diffuse, low-confidence field there. A confident
one has hallucinated a trajectory, and no accuracy metric can see it. Both
metrics on this page therefore only mean something when read on a negative
control — a terminal-state subset, a population of differentiated cells —
ideally against the same method's score on a paired positive control.

Both read a row-stochastic transition matrix from `adata.obsp`. It is not
computed for you; see {func}`~veloeval.prepare`.

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.sts
   veloeval.metrics.ees
```
