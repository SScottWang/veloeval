# ConstVelo

A probe of the evaluation, not a velocity method: every cell gets the same
unit gene-space velocity, from one cluster's mean `Ms` to another's. Holding
the velocity fixed, a score that moves across neighbour graphs or layouts is
moving because of the evaluation alone. It does not enter the ranking.

Compute the direction once, on one reference, and give the same matrix to
every configuration. `Ms` is used as it is when present; otherwise it is built
with 30 PCs and 30 neighbours, both changeable.

```python
import anndata as ad
import numpy as np
import veloeval as ve

c = ve.constvelo(ref30.copy(), start="Ductal", end="Beta")
for k, ref in references.items():                      # same c for every k
    m = ad.AnnData(obs=ref.obs[["clusters"]], var=ref.var[[]])
    m.layers["velocity"] = np.tile(c, (m.n_obs, 1)).astype(np.float32)
    m.var["velocity_genes"] = True
    ve.prepare(m, reference=ref)
```

On scVelo's pancreas data — 2000 HVGs, Ductal to Beta, the dataset's own UMAP
held fixed, kNN rebuilt with $k = 15, 30, 50$ on 30 PCs — the projected PCA
arrow is one vector everywhere, while the arrows scVelo projects into the UMAP,
and the scores, change with the graph alone. Pancreas branches into four
lineages, so one direction is an idealised example, right only roughly.

![ConstVelo on pancreas](../_static/constvelo_pancreas.png)

| k | CBDir (UMAP) | CBDir (gene) | CBVCoh (UMAP) | ICVCoh (UMAP) |
| --- | --- | --- | --- | --- |
| 15 | 0.473 | 0.100 | 0.870 | 0.809 |
| 30 | 0.476 | 0.136 | 0.882 | 0.846 |
| 50 | 0.466 | 0.152 | 0.873 | 0.835 |

Rescaling the velocity leaves the four scores unchanged: bit for bit at
$\times 0.125$ and $\times 8$, to within $2 \times 10^{-7}$ at $\times 0.1$ and
$\times 10$, where the float32 velocity itself is rounded. The figure and the
table come from `docs/_scripts/constvelo_pancreas.py`.

```{eval-rst}
.. autofunction:: veloeval.constvelo
```
