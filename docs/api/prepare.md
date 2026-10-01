# Preparing an h5ad

The metrics never derive a velocity graph, a kNN or a transition matrix on the
fly. Doing so would give the methods whose wrappers already built one *their*
parameters and everyone else defaults — so two methods would no longer be on the
same footing, and nothing in the results would record which was which. A missing
field is `missing_input`, not something to quietly invent.

Call `prepare` once at the end of each method wrapper instead. It gives every
method the same neighbourhood and the same embedding projection, and writes what it did — including the scVelo version — into
`adata.uns["veloeval"]["prepared"]`, so the provenance travels with the file.

```python
import veloeval as ve

ve.prepare(adata, space="gene", basis="umap", n_neighbors=30)
adata.write_h5ad(out_path)
```

## Reference mode

Reusing each method's own neighbour graph still leaves every method on its own
neighbourhood and its own UMAP, and scVelo builds each velocity graph on them.
With `reference=`, `prepare` runs in a separate evaluation step instead, and
every method gets the same PCA, embedding, neighbour graph and `Ms` from one
shared reference; the velocity graph and its projection are then recomputed on
them. The method's environment needs no veloeval.

The reference is built once per dataset, upstream: all cells, `obsm['X_pca']`,
`obsm['X_umap']`, the output of `scanpy.pp.neighbors` and `layers['Ms']` from
`scvelo.pp.moments` on that graph.

```python
ref = ad.read_h5ad(reference_path)
adata = ad.read_h5ad(velocity_path)            # one method's output
ve.prepare(adata, space="gene", basis="umap", reference=ref)
```

Before anything is copied, whatever the method derived from its own graph or
embedding is dropped — its embeddings, projections, transition matrix,
`velocity_*` columns such as its pseudotime, and its velocity graph — so no
metric can read a stale field. Its own time (`latent_time`, `fit_t`) stays.
Then every `obsm['X_*']` of the reference is copied. A method that dropped
cells gets its graph rebuilt on the reference representation of the cells it
kept (`X_pca`, or the reference's `use_rep`), with the reference's parameters.

In gene space the velocity graph is built on one gene set per method — the
genes it scored (`var['velocity_genes']` if present), with finite velocity,
that the reference has — and that set is written to `var['veloeval_genes']`.
Every gene-space metric then reads only those genes, so a method's velocity
graph, `field_constancy`, `icvcoh(basis=None)` and `velocity_consistency` are
on the same genes. The set is not unified across methods: which genes a method
can fit is part of what is being compared.

Only gene and latent velocities are projected; an embedding-space velocity
lives in the method's own embedding and has no counterpart in the reference
one, so its embedding metrics become `missing_input`. A failed velocity graph
is recorded, not raised, and leaves the projection, transition matrix and
pseudotime skipped. What was removed, copied, rebuilt and recomputed is
recorded under `reference`, `neighbors` and `velocity_graph` in
`adata.uns["veloeval"]["prepared"]`.

```{eval-rst}
.. automodule:: veloeval.prepare
   :members: prepare, project_velocity, build_neighbor_indices, build_spatial_neighbors
```
