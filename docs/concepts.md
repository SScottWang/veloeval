# Three things that are deliberate

If you read nothing else, read this page. Each of these is a decision that
looks like extra friction and exists to stop a specific way benchmarks go
wrong.

## 1. A missing value says why it is missing

Every metric returns a {class}`~veloeval.MetricResult`, never a bare float:

```{list-table}
:header-rows: 1
:widths: 18 42 40

* - status
  - meaning
  - what to do
* - `ok`
  - a real measurement
  - use it
* - `not_applicable`
  - the metric is undefined for this method or dataset
  - leave the cell empty in the paper — this is information, not a gap
* - `missing_input`
  - a required field is absent from the h5ad
  - fix the pipeline; `detail` names the field
* - `failed`
  - the computation raised
  - fix the bug; `detail` carries the exception
```

Collapsing these into `np.nan` is the default thing to do and it is wrong.
`cbdir` alone can be empty for four reasons: the dataset has no curated edges,
the projection was never written upstream, the kNN is corrupt, or no cell
happened to sit on a boundary. Those need four different responses from a
human, and in a table of NaN they are indistinguishable.

It also breaks aggregation. Averaging three seeds when one crashed silently
reports a mean over two while the row still says `n_seeds = 3`:

```python
# wrong
df.groupby(["method", "dataset"])[metric].mean()

# right — a metric is only aggregated over seeds that agree on status
ok = df[df[f"{metric}_status"] == "ok"]
agg = ok.groupby(["method", "dataset"])[metric].agg(["mean", "std", "count"])
```

**Group by status before aggregating, and refuse to average a mixed group.**

## 2. Nothing is derived on the fly

If `velocity_graph`, the kNN or a transition matrix is missing, that is
`missing_input`. The library never builds them.

It is tempting to build them — one call to `scv.tl.velocity_graph` and the
metric runs. But a velocity graph is not a lookup, it is a modelling step with
parameters (`n_neighbors`, `n_recurse_neighbors`, `mode_neighbors`). Derive it
here and the methods whose wrappers already built one get scored on *their*
parameters while the rest get scored on defaults — so two methods are no longer
on the same footing, and nothing in the results records which happened.

Derive once, upstream, identically for everyone, at the end of each wrapper:

```python
import veloeval as ve

ve.prepare(adata, space="gene", basis="umap", n_neighbors=30)
adata.write_h5ad(out)
```

{func}`~veloeval.prepare` writes what it did — including the scVelo version —
into `adata.uns["veloeval"]["prepared"]`, so the provenance travels with the
file rather than living in someone's memory.

## 3. Velocity space is declared, not guessed

A latent-space velocity and a gene-space velocity are both an
`(n_cells, n_features)` array in `layers["velocity"]`. A metric that assumes
the wrong one returns a number that looks fine and is not comparable.

**A wrong number is worse than a missing one.** A missing one gets
investigated; a wrong one gets into a figure.

So the wrapper declares the space, and metrics needing gene-wise
correspondence return `not_applicable` for the rest:

```{list-table}
:header-rows: 1
:widths: 18 40 42

* - space
  - methods
  - what still applies
* - `gene`
  - scVelo, velocyto, veloVI, cellDancer, …
  - everything
* - `latent`
  - VeloAE, VeloVAE, …
  - embedding-space metrics; not `truth_cos` / `gamma_corr`
* - `embedding`
  - projected-only outputs
  - embedding-space metrics
* - `scalar`
  - VeloCycle (angular velocity)
  - none of the vector metrics
```

```python
ve.prepare(adata, space="latent", basis="umap")
# or, if you are not using prepare:
ve.set_velocity_space(adata, "latent")
```

The default is `"gene"`, which is right for most methods and wrong loudly
rather than quietly for the rest.
