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

```{rubric} The labelling reference
```

veloeval compares against a reference; it does not build one. Derive it from
the labelling channel with a tool such as dynamo, on the same cells, and hide
that channel from the method being scored. How depends on the design:

- **one-shot** — a single labelling pulse of length $t$. Only under steady state
  does the new/total fraction give the rate, $k = 1 - e^{-\gamma t}$, so fit $k$
  on unperturbed cells; stimulated or differentiating cells read a rise in
  transcription as fast decay. The reference velocity is then
  $\gamma\,(N/k - R)$ for new RNA $N$ and total RNA $R$.
- **kinetics (pulse)** — several labelling lengths; the accumulation of new RNA
  fits $\alpha$ and $\gamma$ without assuming steady state.
- **pulse-chase (degradation)** — labelled RNA decays as $e^{-\gamma t}$ once
  labelling stops, giving $\gamma$ directly.

`gamma_corr` needs a trustworthy $\gamma$; `truth_cos` also needs cells that are
changing, since at steady state the reference velocity is close to 0
everywhere and the cosine is noise.

| Design | `gamma_corr` | `truth_cos` |
| --- | --- | --- |
| one-shot, every cell at steady state | yes | no — nothing is changing |
| one-shot, cells differentiating, no unperturbed group | with caution — $\gamma$ biased | with caution |
| one-shot + perturbation time course with an unperturbed group | yes — fit on that group | yes |
| kinetics (pulse) | yes | yes |
| pulse-chase, chase only | yes | no — transcription unseen |
| pulse and chase together | yes | yes |
| no labelling | no | no |

Labelled and spliced reads must come from the same cells, and incomplete 4sU
conversion lowers the new/total fraction and so $\gamma$ unless corrected.

Values are low on real data and need controls beside them: a cell-shuffled
velocity keeps each gene's average direction and scores above 0. When
comparing methods, pass them all the same `genes`: a method that leaves
poorly fitted genes `nan` is otherwise scored only on the genes it fits well.
For `gamma_corr`, splicing models fix rates only up to each gene's time scale,
so pass a ratio such as scVelo's `fit_gamma / fit_beta`.
