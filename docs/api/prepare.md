# Preparing an h5ad

The metrics never derive a velocity graph, a kNN or a transition matrix on the
fly. Doing so would give the methods whose wrappers already built one *their*
parameters and everyone else defaults — so two methods would no longer be on the
same footing, and nothing in the results would record which was which. A missing
field is `missing_input`, not something to quietly invent.

Call `prepare` once at the end of each method wrapper instead. It gives every
method the same neighbourhood, the same embedding projection and the same
transition matrix, and writes what it did — including the scVelo version — into
`adata.uns["veloeval"]["prepared"]`, so the provenance travels with the file.

```python
import veloeval as ve

ve.prepare(adata, space="gene", basis="umap", n_neighbors=30)
adata.write_h5ad(out_path)
```

```{eval-rst}
.. automodule:: veloeval.prepare
   :members: prepare, project_velocity, build_neighbor_indices
```
