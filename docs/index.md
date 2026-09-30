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
pip install git+https://github.com/SScottWang/veloeval.git@v0.0.2
```

`prepare()` and `velocity_consistency` (which calls scVelo's own
`velocity_confidence`) also need scvelo and scanpy: install with the `prepare`
extra, `pip install "veloeval[prepare] @ git+https://github.com/SScottWang/veloeval.git@v0.0.2"`.

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
| | {func}`~veloeval.metrics.cto` | inferred time; curated edges or measured stages | higher |
| **coherence** | {func}`~veloeval.metrics.icvcoh` | cell-type labels | higher |
| | {func}`~veloeval.metrics.velocity_consistency` | — | higher |
| | {func}`~veloeval.metrics.spatial_consistency` | spatial coordinates | higher |
| | {func}`~veloeval.metrics.time_morans_i` | inferred time; spatial coordinates | higher |
| **temporal** | {func}`~veloeval.metrics.tsc` | measured linear time axis | higher |
| | {func}`~veloeval.metrics.phase_corr` | inferred time; FUCCI phase | higher |
| **negative control** | {func}`~veloeval.metrics.sts` | velocity graph | higher |
| | {func}`~veloeval.metrics.sts_abs` | velocity graph | higher |
| | {func}`~veloeval.metrics.ees` | velocity graph | higher |
| **ground truth** | {func}`~veloeval.metrics.phase_dir` | FUCCI phase | higher |
| | {func}`~veloeval.metrics.truth_cos` | labelling reference | higher |
| | {func}`~veloeval.metrics.gamma_corr` | labelling reference, per-gene rate | higher |
| | {func}`~veloeval.metrics.lineage_fate` | clonal barcodes; velocity graph | higher |
| | {func}`~veloeval.metrics.rate_err` | inferred period in hours; measured period | lower |
| **agreement** | {func}`~veloeval.metrics.agreement` | every run on the dataset, one neighbour graph | higher |

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
| `missing_input` | a required field is absent from the h5ad, or scvelo is not installed; `detail` names it |
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
- **CellRank's terminal-state identification (TSI)** — on a synthetic tree
  the walk without any velocity direction scored 1.0, above the true field
  (0.73) and barely above the reversed one (0.67), and it needs CellRank with
  PETSc/SLEPc.
- **`truth_cos` for latent-space methods** — needs a space-free measure
  (distance correlation between the two neighbourhood geometries). Those
  methods currently return `not_applicable` rather than an incomparable number.

## Reference parity

`cbdir` follows VeloAE (Qiao & Huang, *PNAS* 2021) and the 2026 *Genome
Biology* benchmark; `cbvcoh` follows VeloAE; `cto` is the CTO of the
*Genome Biology* benchmark, a refinement of VeloVAE's Time Accuracy Score (Gu et al.);
`sts` is scVelo's self-transition probability (Bergen et al., 2020) as used by the
*Genome Biology* benchmark, and `sts_abs` its variant without the 98th-percentile
reference; `icvcoh`, `tsc` and `ees` follow the *Genome Biology* benchmark.
`cbdir`, `cbvcoh` and `icvcoh` all read the UMAP-projected velocity by default
(VeloAE computed the two coherence metrics on `layers`; pass `basis=None` to
`icvcoh` for that). `phase_corr` is Fisher & Lee's circular correlation
(*Biometrika* 1983) on uniform scores; `spatial_consistency` and `time_morans_i`
follow TopoVelo (Gu et al., *Nat Biotechnol* 2025), as used by Huang et al.
(bioRxiv 2026); `agreement` is the A2 of CZ Biohub (bioRxiv 2024), as used by the
CRM 2026 benchmark; `lineage_fate` is the clonal fate benchmark of LARRY (Weinreb et
al., *Science* 2020), and `rate_err` VeloCycle's comparison with live imaging
(Lederer et al., *Nat Methods* 2024).

Checked against scVelo's code, cell by cell, in the tests: `sts` against
`scvelo.tl.velocity_graph`'s self-transition probability, `velocity_consistency`
and `spatial_consistency` against `scvelo.tl.velocity_confidence`, and the
transition matrix `ees` reads against `scvelo.tl.transition_matrix`. `agreement`
is checked cell by cell against CZ Biohub's dense computation, run as their
script runs it. `phase_corr` is checked against the pairwise definition,
`time_morans_i` against analytic values on a ring, and the absorption
probabilities `lineage_fate` reads against a dense solve.

Not yet checked against the original code on the same data: `cbdir`, `cbvcoh`,
`icvcoh`, `cto`, `tsc` and the EES formula itself (VeloAE, *Genome Biology*),
`spatial_consistency` and `time_morans_i` (Huang et al.), `agreement`
(CRM 2026), and `lineage_fate`, which scores every barcoded progenitor rather than
LARRY's curated subset. **Until they
are, do not present these numbers as reproducing published ones.**
