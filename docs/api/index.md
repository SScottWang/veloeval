# API

Generated from the docstrings in the source. Change a docstring, push, and
these pages change with it.

```{toctree}
:maxdepth: 2

direction
coherence
temporal
negative
groundtruth
helpers
```

Every metric takes an `AnnData` and returns a
{class}`~veloeval.MetricResult` — a value plus a status saying why the value is
(or is not) there. Two of them, {func}`~veloeval.metrics.truth_cos` and
{func}`~veloeval.metrics.gamma_corr`, take a second `AnnData` carrying the
labelling-derived reference.

| status | meaning |
| --- | --- |
| `ok` | a real measurement |
| `not_applicable` | undefined for this method or dataset — information, not a gap |
| `missing_input` | a required field is absent from the h5ad; `detail` names it |
| `failed` | the computation raised; `detail` carries the exception |
