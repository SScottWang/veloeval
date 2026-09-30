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

A method that dropped cells gets its graph rebuilt on the reference PCA of the
cells it kept, with the reference's parameters. What was copied, rebuilt and
recomputed is recorded under `reference`, `neighbors` and `velocity_graph` in
`adata.uns["veloeval"]["prepared"]`.

```{eval-rst}
.. automodule:: veloeval.prepare
   :members: prepare, project_velocity, build_neighbor_indices, build_spatial_neighbors
```
