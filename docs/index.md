# veloeval

Metrics for benchmarking RNA velocity methods.

`veloeval` evaluates. It does not run velocity methods and it does not own a
workflow — those live in the VeloBench pipeline. The split is the one `scib`
and `scib-pipeline` use, and for the same reason: the pipeline changes weekly,
the metric definitions should not, and only one of the two belongs in a paper's
methods section.

```{toctree}
:maxdepth: 2
:hidden:

concepts
tutorial
api
```

## Install

```bash
pip install git+https://github.com/<org>/veloeval.git@v0.1.0
```

Pin a tag rather than tracking `main`. Every results row carries
`veloeval_version`, so a table computed in March and one computed in June stay
distinguishable — but only if the version is a thing you chose.

## Thirty seconds

```python
import anndata as ad
import veloeval as ve

adata = ad.read_h5ad("2.velocity/pancreas/scvelo_dynamical/seed_42/velocity.h5ad")

results = ve.compute_all(
    adata,
    label_key="clusters",
    cluster_edges=[["Ductal", "Ngn3 low EP"], ["Ngn3 low EP", "Ngn3 high EP"]],
)
row = ve.to_row(results, method="scvelo_dynamical", dataset="pancreas", seed=42)
```

`row` carries one `<metric>` column, one `<metric>_status` column, and
`veloeval_version`.

## The metrics

Eleven run from a single h5ad; two need more than one input and say so in their
signatures.

| | metric | needs | direction |
| --- | --- | --- | --- |
| **direction** | {func}`~veloeval.metrics.cbdir` | curated edges | higher |
| | {func}`~veloeval.metrics.cbvcoh` | curated edges | higher |
| | {func}`~veloeval.metrics.cto` | curated edges, inferred time | higher |
| **coherence** | {func}`~veloeval.metrics.icvcoh` | cell-type labels | higher |
| | {func}`~veloeval.metrics.velocity_consistency` | — | higher |
| **temporal** | {func}`~veloeval.metrics.tsc` | measured time axis | higher |
| **negative control** | {func}`~veloeval.metrics.sts` | transition matrix | higher |
| | {func}`~veloeval.metrics.ees` | transition matrix | higher |
| | {func}`~veloeval.metrics.mag_ratio` | **two runs** | toward 0 |
| **ground truth** | {func}`~veloeval.metrics.phase_dir` | FUCCI phase | higher |
| | {func}`~veloeval.metrics.truth_cos` | labelling reference | higher |
| | {func}`~veloeval.metrics.gamma_corr` | labelling reference, per-gene rate | higher |
| **meta** | {func}`~veloeval.metrics.rho_rank` | **a results table** | higher |

Coherence says nothing about whether the field points the *right* way — a
confidently wrong field scores high. Read it next to the direction metrics,
never instead of them.

## Not implemented

- **`rate_err`** — absolute rate error against live-imaging cycle duration,
  pending the RPE1 GSE250148 data.
- **`lineage_fate`** — CellRank fate probabilities against LARRY clonal
  barcodes. Needs a CellRank dependency and a dataset still marked conditional.
- **`truth_cos` for latent-space methods** — needs a space-free measure
  (distance correlation between the two neighbourhood geometries). Those
  methods currently return `not_applicable` rather than an incomparable number.

Robustness (HVG sensitivity, sequencing-depth stability) and efficiency
(runtime, peak memory, completion rate) are properties of *repeated runs*, not
of one h5ad. They belong to the pipeline and are deliberately not here.

## Reference parity

`cbdir`, `cbvcoh` and `icvcoh` follow VeloAE (Qiao & Huang, *PNAS* 2021);
`tsc`, `sts` and `ees` follow the 2026 *Genome Biology* benchmark. They have not
yet been checked against those implementations on the same data. **Until they
are, do not present these numbers as reproducing published ones.**
