# Tutorial

End to end, the way the pipeline uses it.

## In the method wrapper

At the end of Module 2, once the method has written its velocity, call
{func}`~veloeval.prepare`. It gives every method the same kNN, the same
embedding projection and the same transition matrix, and records the parameters
it used in `adata.uns["veloeval"]["prepared"]`.

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

The `space` argument matters. A latent-space velocity and a gene-space velocity
are both an `(n_cells, n_features)` array in `layers["velocity"]`, and a metric
that assumes the wrong one returns a number that looks fine and is not
comparable. Declaring it makes {func}`~veloeval.metrics.truth_cos` and
{func}`~veloeval.metrics.gamma_corr` report `not_applicable` for those methods
instead of a meaningless cosine.

`prepare` never raises on a failed projection: it records the error and lets the
metrics report `missing_input` downstream, so one method failing to project does
not take the run down.

## In the metrics rule

The library gives you metrics; assembling the row is yours. Roughly:

```python
import anndata as ad
import pandas as pd
import veloeval as ve
from veloeval import metrics as M

adata = ad.read_h5ad(velocity_path)
label_key = ds.get("label_key")          # None on single cell-line datasets
edges = ds.get("cluster_edges")          # None where none are curated

results = {
    "cbdir": M.cbdir(adata, label_key=label_key, cluster_edges=edges),
    "cbvcoh": M.cbvcoh(adata, label_key=label_key, cluster_edges=edges),
    "cto": M.cto(adata, label_key=label_key, cluster_edges=edges),
    "icvcoh": M.icvcoh(adata, label_key=label_key),
    "velocity_consistency": M.velocity_consistency(adata),
    "tsc": M.tsc(adata, time_key="latent_time",
                 true_time_key=ds.get("time_key", "__absent__")),
    "sts": M.sts(adata),
    "ees": M.ees(adata),
    "phase_dir": M.phase_dir(adata, phase_key=ds.get("phase_key", "__absent__")),
}

row = {"method": method, "dataset": dataset, "seed": seed,
       "veloeval_version": ve.__version__}
for name, r in results.items():
    row[name] = r.value
    row[f"{name}_status"] = r.status
    if r.status != "ok":
        row[f"{name}_detail"] = r.detail

pd.DataFrame([row]).to_csv(output_path, index=False)
```

Pass `None` freely. A dataset without curated edges gives
`cbdir_status = "not_applicable"`, which is the correct entry for that cell of
the results table — not an error to work around.

The two metrics needing a labelling reference take a second AnnData:

```python
ref = ad.read_h5ad(labelling_reference_path)
results["truth_cos"] = M.truth_cos(adata, ref)
results["gamma_corr"] = M.gamma_corr(adata, ref)
```

## Reading one result

```python
r = results["cbdir"]
r.status    # "ok" | "not_applicable" | "missing_input" | "failed"
r.value     # 0.43, or None when status != "ok"
r.detail    # why, when status != "ok"
r.per_cell  # array, nan where the cell could not be scored
float(r)    # 0.43, or nan
bool(r)     # True only when status == "ok"
```

`per_cell` is what to plot when a method scores badly and you want to know which
cells are responsible.

## Aggregating across seeds

```python
frames = [pd.read_csv(f) for f in seed_files]
df = pd.concat(frames, ignore_index=True)

rows = []
for metric in ve.DIRECTION:
    statuses = set(df[f"{metric}_status"].dropna())
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
crashed on two must not appear as a confident mean, and `mean()` skipping NaN
silently makes it look like one.

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
