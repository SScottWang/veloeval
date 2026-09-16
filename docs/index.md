# veloeval

Metrics for benchmarking RNA velocity methods.

`veloeval` implements metrics and nothing else. It does not run velocity
methods, does not own a workflow, and does not assemble result tables — those
belong to the VeloBench pipeline, which changes far more often than a metric
definition should.

```{toctree}
:maxdepth: 2
:hidden:

tutorial
api/index
```

## Install

```bash
pip install git+https://github.com/SScottWang/veloeval.git@v0.0.1
```

Pin a tag rather than tracking `main`, and record `veloeval.__version__`
alongside the numbers. Without it a table computed before a metric was fixed is
indistinguishable from one computed after.

## Thirty seconds

```python
import anndata as ad
from veloeval.metrics import cbdir, icvcoh

adata = ad.read_h5ad("2.velocity/pancreas/scvelo_dynamical/seed_42/velocity.h5ad")

r = cbdir(
    adata,
    label_key="clusters",
    cluster_edges=[["Ductal", "Ngn3 low EP"], ["Ngn3 low EP", "Ngn3 high EP"]],
)
r.status    # "ok"
r.value     # 0.43
r.per_cell  # array, nan where the cell could not be scored
```

## The metrics

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
| **ground truth** | {func}`~veloeval.metrics.phase_dir` | FUCCI phase | higher |
| | {func}`~veloeval.metrics.truth_cos` | labelling reference | higher |
| | {func}`~veloeval.metrics.gamma_corr` | labelling reference, per-gene rate | higher |

Coherence says nothing about whether the field points the *right* way — a
confidently wrong field scores high. Read it next to the direction metrics,
never instead of them.

## Every result carries a status

A metric returns a {class}`~veloeval.result.MetricResult`, never a bare float, because
an absent value has to say why it is absent:

| status | meaning |
| --- | --- |
| `ok` | a real measurement |
| `not_applicable` | undefined for this method or dataset — information, not a gap |
| `missing_input` | a required field is absent from the h5ad; `detail` names it |
| `failed` | the computation raised; `detail` carries the exception |

Group by status before aggregating across seeds, and do not average a mixed
group: three seeds where one crashed must not be reported as a confident mean.

## Deliberately not here

- **Orchestration** — assembling rows, aggregating seeds, recording runtime and
  peak memory. Properties of a *run*, not of an h5ad.
- **Robustness sweeps** — HVG sensitivity, sequencing-depth stability. These
  are repeated runs; the pipeline owns them.
- **Arithmetic over results** — magnitude ratios between paired control runs,
  rank concordance between two metric columns. A few lines of pandas at
  analysis time, with no definition worth freezing in a library.
- **`rate_err`** — absolute rate error against live-imaging cycle duration,
  pending the RPE1 GSE250148 data.
- **`truth_cos` for latent-space methods** — needs a space-free measure
  (distance correlation between the two neighbourhood geometries). Those
  methods currently return `not_applicable` rather than an incomparable number.

## Reference parity

`cbdir`, `cbvcoh` and `icvcoh` follow VeloAE (Qiao & Huang, *PNAS* 2021);
`tsc`, `sts` and `ees` follow the 2026 *Genome Biology* benchmark. They have not
yet been checked against those implementations on the same data. **Until they
are, do not present these numbers as reproducing published ones.**
