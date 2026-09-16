# veloeval

Metrics for benchmarking RNA velocity methods.

**Docs: <https://veloeval.readthedocs.io>** — generated from the docstrings, so
it is never out of date with the code.

`veloeval` implements metrics and nothing else. It does not run velocity
methods, does not own a workflow, and does not assemble result tables — those
belong to the VeloBench pipeline, which changes far more often than a metric
definition should.

```bash
pip install git+https://github.com/SScottWang/veloeval.git@v0.0.1
```

## Use

```python
import anndata as ad
from veloeval.metrics import cbdir

adata = ad.read_h5ad("2.velocity/pancreas/scvelo_dynamical/seed_42/velocity.h5ad")

r = cbdir(
    adata,
    label_key="clusters",
    cluster_edges=[["Ductal", "Ngn3 low EP"], ["Ngn3 low EP", "Ngn3 high EP"]],
)
r.status, r.value      # ("ok", 0.43)
```

## Metrics

| group | metric | needs | direction |
| --- | --- | --- | --- |
| direction | `cbdir`, `cbvcoh`, `cto` | curated `cluster_edges` | higher |
| coherence | `icvcoh` | cell-type labels | higher |
| | `velocity_consistency` | — | higher |
| temporal | `tsc` | measured time axis | higher |
| negative control | `sts`, `ees` | transition matrix | higher |
| ground truth | `phase_dir` | FUCCI phase | higher |
| | `truth_cos`, `gamma_corr` | labelling reference | higher |

Coherence says nothing about whether the field points the *right* way — a
confidently wrong field scores high. Read it next to the direction metrics,
never instead of them.

## Three things that are deliberate

**A missing value says why it is missing.** Every metric returns a
`MetricResult`, never a bare float: `ok`, `not_applicable` (undefined for this
method/dataset), `missing_input` (a required field is absent upstream —
`detail` names it), `failed` (it raised — `detail` carries the exception).
Collapsing these into `np.nan` makes the table unreadable and aggregation
wrong. Group by status before averaging seeds, and refuse to average a mixed
group.

**Nothing is derived on the fly.** A missing velocity graph, kNN or transition
matrix is `missing_input`. Building one here would give the methods whose
wrappers already built one *their* parameters and the rest defaults, so two
methods would no longer be on the same footing and nothing would record which.
Derive once, upstream, identically for everyone, with `ve.prepare(...)` at the
end of each wrapper; it writes what it did into
`adata.uns["veloeval"]["prepared"]`.

**Velocity space is declared, not guessed.** `gene` / `latent` / `embedding` /
`scalar`. A metric that assumes the wrong one returns a number that looks fine
and is not comparable — a wrong number is worse than a missing one. Declare it
via `ve.prepare(..., space=...)` or `ve.set_velocity_space(adata, "latent")`.

## Deliberately not here

Orchestration (assembling rows, aggregating seeds, runtime and memory),
robustness sweeps (HVG sensitivity, depth stability), and arithmetic over
results (magnitude ratios between paired control runs, rank concordance between
two metric columns) all belong to the pipeline. So do `rate_err`, pending the
RPE1 GSE250148 data, and a space-free `truth_cos` variant for latent-space
methods.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Every metric is checked on a field whose answer is known analytically: a perfect
field scores 1, its reverse −1, an unstructured one ≈0, and the score must not
change when the velocity is rescaled. That is the class of bug that silently
flips a benchmark's conclusions.

## Docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The site is generated from the docstrings — adding a metric means writing its
docstring and listing it in `docs/api.md`, not writing a second copy of the
explanation. readthedocs rebuilds on every push.

## Reference parity

`cbdir`, `cbvcoh` and `icvcoh` follow VeloAE (Qiao & Huang, *PNAS* 2021);
`tsc`, `sts` and `ees` follow the 2026 *Genome Biology* benchmark. They have not
yet been checked against those implementations on the same data. Until they are,
do not present these numbers as reproducing published ones.
