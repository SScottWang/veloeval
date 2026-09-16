# Independent Ground-Truth Metrics

The direction, coherence and temporal metrics are all scored against structure
that was itself read off the transcriptome. A systematic error shared across the
field is invisible to them by construction.

The metrics here use a signal the model never saw:

- {func}`~veloeval.metrics.phase_dir` — FUCCI fluorescence, a protein-level,
  live-imageable readout of cell-cycle position
- {func}`~veloeval.metrics.truth_cos` — metabolic labelling, a physical clock
  with real units
- {func}`~veloeval.metrics.gamma_corr` — labelling-derived degradation rates,
  compared gene by gene

They are the point of the benchmark, and they have the narrowest applicability
of anything here: they need FUCCI reporters or a labelling channel, and
`truth_cos` and `gamma_corr` need gene-wise correspondence, so methods whose
velocity lives in a learned latent space return `not_applicable` rather than an
incomparable number. **Read the `not_applicable` details, not just the values.**

```{eval-rst}
.. autosummary::
   :toctree: generated

   veloeval.metrics.phase_dir
   veloeval.metrics.truth_cos
   veloeval.metrics.gamma_corr
```
