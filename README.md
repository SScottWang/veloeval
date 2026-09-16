# veloeval

Metrics for benchmarking RNA velocity methods.

**Docs: <https://veloeval.readthedocs.io>** — generated from the docstrings, so
it is never out of date with the code.

`veloeval` evaluates. It does not run velocity methods and it does not own a
workflow — those live in the VeloBench pipeline. The split is the same one
`scib` and `scib-pipeline` use, and for the same reason: the pipeline changes
weekly, the metric definitions should not, and only one of the two belongs in a
paper's methods section.

```bash
pip install git+https://github.com/<org>/veloeval.git@v0.1.0
```

## Use

```python
import anndata as ad
import veloeval as ve

adata = ad.read_h5ad("2.velocity/pancreas/scvelo_dynamical/seed_42/velocity.h5ad")

results = ve.compute_all(
    adata,
    label_key="clusters",
    cluster_edges=[["Ductal", "Ngn3 low EP"], ["Ngn3 low EP", "Ngn3 high EP"]],
    true_time_key=None,      # this dataset has no measured time axis
    phase_key=None,          # ...and no FUCCI phase
)

row = ve.to_row(results, method="scvelo_dynamical", dataset="pancreas", seed=42)
```

`row` carries one `<metric>` column, one `<metric>_status` column, and
`veloeval_version`.

## Three things that are deliberate

### 1. A missing value says why it is missing

Every metric returns a `MetricResult`, never a bare float:

| status | meaning | what to do |
| --- | --- | --- |
| `ok` | a real measurement | use it |
| `not_applicable` | the metric is undefined for this method/dataset | leave the cell empty in the paper; this is information, not a gap |
| `missing_input` | a required field is absent from the h5ad | fix the pipeline — `detail` names the field |
| `failed` | the computation raised | fix the bug — `detail` carries the exception |

Collapsing these into `np.nan` makes the results table unreadable and
aggregation wrong: averaging three seeds when one crashed silently reports a
mean over two while the row still says `n_seeds = 3`. **Group by status before
aggregating across seeds, and refuse to average a mixed group.**

### 2. Nothing is derived on the fly

If `velocity_graph`, the kNN, or a transition matrix is missing, that is
`missing_input`. The library never builds them.

Building them here would apply *default* parameters to the methods whose
wrappers did not build them and the wrapper's own parameters to those that did
— so two methods would no longer be on the same footing, and nothing in the
results would record which happened. Derive once, upstream, identically for
everyone, at the end of each method wrapper:

```python
import veloeval as ve
ve.prepare(adata, space="gene", basis="umap", n_neighbors=30)
adata.write_h5ad(out)
```

`prepare` writes what it did into `adata.uns["veloeval"]["prepared"]`, including
the scVelo version, so the provenance travels with the file.

### 3. Velocity space is declared, not guessed

A latent-space velocity and a gene-space velocity are both an
`(n_cells, n_features)` array in `layers["velocity"]`. A metric that assumes the
wrong one returns a number that looks fine and is not comparable — and **a wrong
number is worse than a missing one**.

So the wrapper declares it (`ve.prepare(..., space=...)`, or
`ve.set_velocity_space(adata, "latent")`), and metrics that need gene-wise
correspondence return `not_applicable` for the rest:

| space | methods | what still applies |
| --- | --- | --- |
| `gene` | scVelo, velocyto, veloVI, cellDancer, … | everything |
| `latent` | VeloAE, VeloVAE, … | embedding-space metrics; not `truth_cos` / `gamma_corr` |
| `embedding` | projected-only outputs | embedding-space metrics |
| `scalar` | VeloCycle (angular velocity) | none of the vector metrics |

## Metrics

Single-run — `compute_all` runs all of these:

| name | group | needs | direction |
| --- | --- | --- | --- |
| `cbdir` | direction | `cluster_edges` | higher |
| `cbvcoh` | direction | `cluster_edges` | higher |
| `cto` | direction | `cluster_edges`, inferred time | higher |
| `icvcoh` | coherence | cell-type labels | higher |
| `velocity_consistency` | coherence | — | higher |
| `tsc` | temporal | measured time axis | higher |
| `sts` | negative control | transition matrix | higher |
| `ees` | negative control | transition matrix | higher |
| `phase_dir` | ground truth | FUCCI phase | higher |
| `truth_cos` | ground truth | labelling reference, gene space | higher |
| `gamma_corr` | ground truth | labelling reference, per-gene rate | higher |

Two do not fit "one run in, one number out", and their signatures say so:

- **`mag_ratio(adata_negative, adata_positive)`** — one method on a paired
  negative and positive control. Closer to 0 is better.
- **`rho_rank(results_df, conventional=..., ground_truth=...)`** — a results
  table, not an AnnData. Rank the same methods by a conventional metric and by
  a physical truth, and correlate. Report it per dataset; pooling across
  datasets mixes runs with different applicable-metric sets.

Coherence metrics say nothing about whether the field points the *right* way —
a confidently wrong field scores high. Read them next to the direction metrics,
never instead of them.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The site is generated from the docstrings — adding a metric means writing its
docstring and listing it in `docs/api.md`, not writing a second copy of the
explanation. readthedocs rebuilds on every push.

Every metric is checked on a field whose answer is known analytically: a perfect
field must score 1, its reverse −1, an unstructured one ≈0, and the score must
not change when the velocity is rescaled. That is the class of bug that
silently flips a benchmark's conclusions.

## Open

- **Reference parity.** `cbdir` / `cbvcoh` / `icvcoh` follow VeloAE (Qiao &
  Huang, PNAS 2021); `tsc` / `sts` / `ees` follow the 2026 *Genome Biology*
  benchmark. They have not yet been checked against those implementations on
  the same data. Until they are, do not present these as reproducing published
  numbers.
- **`truth_cos` for latent-space methods.** Needs a space-free measure
  (distance correlation between the two neighbourhood geometries). Not
  implemented — those methods return `not_applicable` rather than a number that
  cannot be compared.
- **`rate_err`.** Absolute rate error against live-imaging cycle duration,
  pending the RPE1 GSE250148 data.

## Versioning

Pin a version in the pipeline config and let `veloeval_version` ride along in
every results row. Tracking `main` across a multi-month project means results
computed in March and in June may come from different implementations, and
nothing in the table would show it. Freeze a tag at submission.
