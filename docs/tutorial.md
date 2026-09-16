# Tutorial

End to end, the way the pipeline uses it.

## In the method wrapper

At the end of Module 2, once the method has written its velocity, call
{func}`~veloeval.prepare` and nothing else. It gives every method the same kNN,
the same embedding projection and the same transition matrix, and records the
parameters it used.

```python
import veloeval as ve

# ... method has written adata.layers["velocity"] ...

ve.prepare(
    adata,
    space="gene",       # "latent" for VeloAE / VeloVAE, "scalar" for VeloCycle
    basis="umap",
    n_neighbors=30,
)
adata.write_h5ad(out_path)
```

`prepare` never raises on a failed projection: it records the error in
`uns["veloeval"]["prepared"]` and lets the metrics report `missing_input`
downstream, so one method failing to project does not take the run down.

## In the metrics rule

```python
import anndata as ad
import pandas as pd
import veloeval as ve

adata = ad.read_h5ad(velocity_path)

results = ve.compute_all(
    adata,
    label_key=ds.get("label_key"),           # None on FUCCI datasets
    cluster_edges=ds.get("cluster_edges"),   # None where none are curated
    true_time_key=ds.get("time_key"),        # None without a measured axis
    phase_key=ds.get("phase_key"),           # None without FUCCI
    basis="umap",
)

row = ve.to_row(
    results,
    method=method,
    dataset=dataset,
    seed=seed,
    **ve.coverage_summary(adata),
)
pd.DataFrame([row]).to_csv(output_path, index=False)
```

Pass `None` freely. A dataset without curated edges gives
`cbdir_status = "not_applicable"`, which is the correct entry for that cell of
the results table — not an error to work around.

## Reading one result

```python
r = results["cbdir"]
r.status    # "ok"
r.value     # 0.43
r.per_cell  # array, nan where the cell had no cross-boundary neighbour
float(r)    # 0.43, or nan when status != "ok"
bool(r)     # True only when status == "ok"
```

`per_cell` is what to plot when a method scores badly and you want to know
which cells are responsible.

## Aggregating across seeds

```python
frames = [pd.read_csv(f) for f in seed_files]
df = pd.concat(frames, ignore_index=True)

rows = []
for metric in ve.DIRECTION:                      # the metric names
    status_col = f"{metric}_status"
    statuses = set(df[status_col].dropna())
    if statuses != {"ok"}:
        # mixed or non-ok: record why, do not average
        rows.append({"metric": metric, "value": None,
                     "status": "|".join(sorted(statuses)), "n_seeds": 0})
        continue
    rows.append({"metric": metric,
                 "value": df[metric].mean(),
                 "std": df[metric].std(),
                 "status": "ok",
                 "n_seeds": len(df)})
```

The point of the `n_seeds: 0` branch: a metric that succeeded on one seed and
crashed on two must not appear as a confident mean.

## The two metrics that are not single-run

`mag_ratio` needs a paired negative and positive control of the same method:

```python
from veloeval.metrics import mag_ratio

neg = ad.read_h5ad("…/pancreas_terminal/scvelo_dynamical/seed_42/velocity.h5ad")
pos = ad.read_h5ad("…/pancreas/scvelo_dynamical/seed_42/velocity.h5ad")

mag_ratio(neg, pos).value   # near 0 for a method that detects steady state
```

`rho_rank` needs the assembled table — this is the benchmark's headline number:

```python
from veloeval.metrics import rho_rank

table = pd.read_csv("3.metrics/rpe1_fucci/all_methods.csv")
rho_rank(table, conventional="cbdir", ground_truth="phase_dir").value
```

Report it per dataset. Pooling methods across datasets mixes runs with
different applicable-metric sets, and the correlation stops meaning anything.

## Adding a metric

Write the function, decorate it, export it, add a synthetic test.

```python
from ..result import metric, NotApplicable
from ..access import get_velocity

@metric
def my_metric(adata, *, some_key: str, vkey: str = "velocity"):
    """One-line summary.

    Higher is better; range ``[-1, 1]``.

    Parameters
    ----------
    adata : anndata.AnnData
        ...

    Returns
    -------
    MetricResult
        ...
    """
    if some_key not in adata.obs:
        raise NotApplicable(f"dataset has no obs['{some_key}']")
    V = get_velocity(adata, vkey)      # raises MissingInput if absent
    ...
    return value, per_cell             # or just value
```

The `@metric` decorator converts `MissingInput` / `NotApplicable` / any other
exception into the matching status, so the body never needs a `try`.

The docstring **is** the documentation — this site is generated from it, so
there is nothing else to update. Add the metric to `DIRECTION` in
`veloeval/metrics/__init__.py` and to `docs/api.md`.

Then the test, on an input whose answer you can write down:

```python
def test_my_metric(linear):
    assert my_metric(linear, some_key="x").value == pytest.approx(1.0, abs=1e-6)
```

Perfect input scores 1, reversed −1, unstructured ≈0, and rescaling the
velocity must not change the answer. That is the class of bug that silently
flips a benchmark's conclusions.
