# The Result Type

Every metric returns a `MetricResult`, never a bare float, because an absent
value has to say why it is absent. Collapsing `not_applicable`, `missing_input`
and `failed` into `np.nan` makes a results table unreadable and aggregation
wrong — averaging three seeds when one crashed silently reports a mean over two
while the row still claims three.

Writing a new metric: raise `MissingInput` or `NotApplicable` from anywhere,
including deep inside an accessor. The `@metric` decorator turns them into the
matching status, so a metric body never needs a `try`.

```{eval-rst}
.. automodule:: veloeval.result
   :members: MetricResult, metric, MissingInput, NotApplicable
   :undoc-members:
```
