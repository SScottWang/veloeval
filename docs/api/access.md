# Accessors

Read-only helpers the metrics use to reach into an `AnnData`. Relevant when
writing a new metric: they raise `MissingInput` or `NotApplicable`, which the
`@metric` decorator converts into the matching status.

Two rules they enforce, both deliberate. **Nothing here computes** — see
[Preparing an h5ad](prepare.md). And **velocity space is declared, not
guessed**: a latent-space velocity and a gene-space velocity are both an
`(n_cells, n_features)` array in `layers["velocity"]`, and a metric that assumes
the wrong one returns a number that looks fine and is not comparable. A wrong
number is worse than a missing one, so `get_velocity` refuses rather than
guesses.

| space | methods | what still applies |
| --- | --- | --- |
| `gene` | scVelo, velocyto, veloVI, cellDancer, … | everything |
| `latent` | VeloAE, VeloVAE, … | embedding-space metrics; not `truth_cos` / `gamma_corr` |
| `embedding` | projected-only outputs | embedding-space metrics |
| `scalar` | VeloCycle (angular velocity) | none of the vector metrics |

```{eval-rst}
.. automodule:: veloeval.access
   :members: velocity_space, set_velocity_space, get_velocity, get_velocity_embedding, get_embedding, get_neighbor_indices, get_transition_matrix, get_labels, gene_coverage
```
