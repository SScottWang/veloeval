# Plots

Pictures of what a metric measured, for reading one result rather than a
table of them. They need matplotlib: `pip install "veloeval[plot]"`.

## Field constancy

{func}`~veloeval.metrics.field_constancy` gives one number, $R$, the length of
the mean unit velocity. The plot shows where it comes from: every cell's
angle to the mean direction, 0° pointing with it and 180° against it, as
wedges whose area counts cells, stacked by cell type. The arrow is $R$ on the
scale of the dashed half-circle — and also the mean cosine of every cell with
the mean direction. The grey band at 90° is where random directions fall in as
many genes: in gene space, a cell there points nowhere in particular.

```python
import veloeval as ve

ve.pl.field_constancy(adata, label_key="clusters", title="scVelo dynamical")
```

On pancreas — 2000 HVGs, 30 PCs and 30 neighbours:

![Field constancy on pancreas](../_static/field_constancy_pancreas.png)

ConstVelo puts every cell at 0°. Both scVelo models keep their cells well off
the random band, at different angles for different cell types. The figure
comes from `docs/_scripts/field_constancy_pancreas.py`.

```{eval-rst}
.. autofunction:: veloeval.pl.field_constancy
```
