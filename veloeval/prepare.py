"""Explicit, recorded derivation of the fields metrics read.

The metrics layer never computes these.  If it did, a method whose wrapper
already built a velocity graph would be scored on its own graph while a method
that did not would be scored on a default one -- and the results table would
not say which.  So derivation happens here, once, at the end of the velocity
module, with identical parameters for every method, and what was done is
written into ``adata.uns["veloeval"]["prepared"]``.

Call :func:`prepare` at the end of each method wrapper, just before writing
``velocity.h5ad`` -- or in a separate evaluation step with ``reference=``, so
the method's environment needs no veloeval and every method is scored on one
shared neighbourhood, embedding and ``Ms``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .access import KNN_KEY, SPATIAL_KNN_KEY, set_velocity_space

__all__ = [
    "prepare",
    "project_velocity",
    "build_neighbor_indices",
    "build_spatial_neighbors",
]


def build_neighbor_indices(
    adata,
    n_neighbors: int = 30,
    n_pcs: int = 30,
    use_rep: str | None = None,
    overwrite: bool = False,
):
    """Write ``obsm['veloeval_knn']`` -- the kNN every local metric uses.

    By default the graph the velocity method's own pipeline left in
    ``obsp['distances']`` is reused: that is the neighbourhood the method was
    fitted on.  With ``overwrite=True``, or when there is no graph yet, one is
    built with :func:`scanpy.pp.neighbors` into a separate slot, so the method's
    own graph -- which ``scv.tl.velocity_graph`` reads -- is left untouched.

    Which of the two happened is recorded in
    ``uns['veloeval']['prepared']['neighbors']``: two methods that each reuse
    their own graph are not necessarily scored on the same neighbourhood.

    Parameters
    ----------
    adata : anndata.AnnData
    n_neighbors, n_pcs, use_rep
        Passed to :func:`scanpy.pp.neighbors` when a graph is built, and mean
        exactly what they mean there: *n_neighbors* counts the cell itself, so
        30 gives 29 neighbours.  The defaults are scVelo's (``scv.pp.moments``
        uses 30 and 30), not scanpy's (15 and 50).  Ignored when an existing
        graph is reused.
    overwrite : bool, default: False
        Build a new graph even if the method already has one.

    Returns
    -------
    numpy.ndarray
        ``(n_obs, k)`` array of neighbour indices, excluding the cell itself.
    """
    import scanpy as sc
    from scipy.sparse import csr_matrix

    if overwrite or "distances" not in adata.obsp:
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
            key_added="veloeval_neighbors",
        )
        graph = adata.uns["veloeval_neighbors"]
        source = "built"
    else:
        graph = adata.uns.get("neighbors", {})
        source = "reused"
    distances = csr_matrix(adata.obsp[graph.get("distances_key", "distances")])
    params = dict(graph.get("params", {}))

    # Sort each row by distance and drop the cell itself.  Not optional: below
    # 8192 cells scanpy stores the other n_neighbors - 1 cells, above it
    # switches to an approximate search that also stores the cell itself and
    # one extra neighbour.
    rows = []
    for i in range(adata.n_obs):
        start, end = distances.indptr[i], distances.indptr[i + 1]
        cols = distances.indices[start:end]
        dists = distances.data[start:end]
        cols, dists = cols[cols != i], dists[cols != i]
        rows.append(cols[np.argsort(dists, kind="stable")])

    if "n_neighbors" in params:
        k = int(params["n_neighbors"]) - 1
    else:
        k = min(len(r) for r in rows)
    short = [i for i, r in enumerate(rows) if len(r) < k]
    if short:
        raise ValueError(
            f"{len(short)} cells have fewer than {k} neighbours in the graph, "
            f"e.g. cell {short[0]}"
        )
    indices = np.array([r[:k] for r in rows], dtype=np.int64)

    adata.obsm[KNN_KEY] = indices
    _record(adata, "neighbors", {"source": source, "k": k, **params})
    return indices


def build_spatial_neighbors(
    adata, *, spatial_key: str = "spatial", n_neighbors: int = 6
) -> np.ndarray:
    """Write ``obsm['veloeval_spatial_knn']`` -- k nearest neighbours in physical space.

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry spot or cell coordinates in ``obsm[spatial_key]``; every
        column is used.
    spatial_key : str, default: "spatial"
        ``obsm`` key of the coordinates.
    n_neighbors : int, default: 6
        Neighbours per cell, excluding the cell itself.  6 is the ring of
        directly adjacent spots on a Visium hexagonal grid.

    Returns
    -------
    numpy.ndarray
        ``(n_obs, n_neighbors)`` array of neighbour indices, excluding the cell
        itself.
    """
    from sklearn.neighbors import NearestNeighbors

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}']")
    xy = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    found = (
        NearestNeighbors(n_neighbors=n_neighbors + 1)
        .fit(xy)
        .kneighbors(xy, return_distance=False)
    )
    # A cell with a coincident twin may be listed after the twin, or not at
    # all; drop it where it is, else drop the farthest neighbour.
    rows = []
    for i, r in enumerate(found):
        rows.append(r[r != i] if (r == i).any() else r[:-1])
    indices = np.array(rows, dtype=np.int64)

    adata.obsm[SPATIAL_KNN_KEY] = indices
    _record(
        adata,
        "spatial_neighbors",
        {"spatial_key": spatial_key, "n_neighbors": n_neighbors},
    )
    return indices


def project_velocity(adata, basis: str = "umap", vkey: str = "velocity", **kwargs):
    """Write ``obsm['{vkey}_{basis}']`` via scVelo's embedding projection.

    Records the scVelo version and the graph parameters, because the projection
    depends on both.
    """
    import scvelo as scv

    graph = f"{vkey}_graph"
    if graph not in adata.uns and graph not in adata.obsp:
        scv.tl.velocity_graph(adata, vkey=vkey, **kwargs)
    scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey)

    _record(adata, f"{vkey}_{basis}", {"scvelo": scv.__version__, **kwargs})
    return adata.obsm[f"{vkey}_{basis}"]


def _use_reference(adata, reference, basis: str) -> int:
    """Give *adata* the reference's PCA, embedding and neighbour graph on its cells.

    Returns how many reference cells the method dropped.  If it dropped some,
    the graph is rebuilt on the reference PCA of the remaining cells with the
    reference's parameters: slicing the reference graph would leave cells short
    of neighbours.
    """
    missing = adata.obs_names.difference(reference.obs_names)
    if len(missing):
        raise ValueError(
            f"{len(missing)} cells are not in the reference, e.g. {missing[0]!r}"
        )
    if "params" not in reference.uns.get("neighbors", {}):
        raise ValueError("reference has no uns['neighbors'] from scanpy.pp.neighbors")
    ref = reference[adata.obs_names]
    for key in ("X_pca", f"X_{basis}"):
        if key not in ref.obsm:
            raise ValueError(f"reference has no obsm['{key}']")
    for key in ("X_pca", f"X_{basis}"):
        adata.obsm[key] = np.asarray(ref.obsm[key])

    params = dict(reference.uns["neighbors"]["params"])
    dropped = reference.n_obs - adata.n_obs
    if dropped == 0:
        for key in ("distances", "connectivities"):
            adata.obsp[key] = ref.obsp[key].copy()
        adata.uns["neighbors"] = {
            "distances_key": "distances",
            "connectivities_key": "connectivities",
            "params": params,
        }
    else:
        import scanpy as sc

        sc.pp.neighbors(
            adata,
            n_neighbors=int(params["n_neighbors"]),
            use_rep="X_pca",
            n_pcs=params.get("n_pcs"),
            method=params.get("method", "umap"),
            metric=params.get("metric", "euclidean"),
            random_state=params.get("random_state", 0),
        )
    return dropped


def _reference_velocity_graph(adata, reference, vkey: str, space: str) -> dict:
    """Recompute scVelo's velocity graph on the reference neighbours.

    Gene space: displacements from the reference ``Ms``, on the genes the
    method scored and the reference has.  Latent space: the method's own
    coordinates in ``layers['Ms']``.  ``sqrt_transform=False`` for everyone:
    scVelo otherwise decides it from the method's own ``uns['{vkey}_params']``.
    Returns what to record.
    """
    import scvelo as scv

    for key in (f"{vkey}_graph", f"{vkey}_graph_neg"):
        adata.uns.pop(key, None)
        if key in adata.obsp:
            del adata.obsp[key]

    if space == "gene":
        if vkey not in adata.layers:
            raise ValueError(f"no layers['{vkey}']")
        if "Ms" not in reference.layers:
            raise ValueError("reference has no layers['Ms']")
        V = adata.layers[vkey]
        V = V.toarray() if hasattr(V, "toarray") else np.asarray(V)
        keep = adata.var_names.isin(reference.var_names) & ~np.isnan(V).any(axis=0)
        if f"{vkey}_genes" in adata.var:
            keep &= adata.var[f"{vkey}_genes"].to_numpy(dtype=bool)
        genes = adata.var_names[keep]
        if len(genes) == 0:
            raise ValueError("no gene is both scored by the method and in the reference")
        ref_ms = reference[adata.obs_names, genes].layers["Ms"]
        Ms = np.zeros(adata.shape, dtype=np.float32)
        Ms[:, keep] = ref_ms.toarray() if hasattr(ref_ms, "toarray") else ref_ms
        # scVelo silently falls back to 'spliced' when xkey is not a layer.
        adata.layers["veloeval_Ms"] = Ms
        try:
            scv.tl.velocity_graph(
                adata, vkey=vkey, xkey="veloeval_Ms", gene_subset=genes,
                sqrt_transform=False,
            )
        finally:
            del adata.layers["veloeval_Ms"]
        return {"xkey": "reference Ms", "n_genes": int(keep.sum()),
                "sqrt_transform": False, "scvelo": scv.__version__}
    if space == "latent":
        if "Ms" not in adata.layers:
            raise ValueError("latent space: put the latent coordinates in layers['Ms']")
        scv.tl.velocity_graph(adata, vkey=vkey, sqrt_transform=False)
        return {"xkey": "method's own layers['Ms']", "sqrt_transform": False,
                "scvelo": scv.__version__}
    return {"skipped": f"no velocity graph in {space!r} space"}


def prepare(
    adata,
    *,
    space: str = "gene",
    basis: str = "umap",
    vkey: str = "velocity",
    reference=None,
    n_neighbors: int = 30,
    n_pcs: int = 30,
    use_rep: str | None = None,
    overwrite_neighbors: bool = False,
    transition: bool = False,
    pseudotime: bool = False,
    spatial_key: str | None = None,
) -> None:
    """One call at the end of a method wrapper.

    Declares the velocity space, picks up (or builds) the kNN, projects velocity
    into *basis*, (with *transition*) writes scVelo's default transition matrix
    to ``obsp['T_fwd']`` -- no metric reads it; the negative-control metrics
    build theirs from the velocity graph -- and (with *pseudotime*) writes
    ``obs['{vkey}_pseudotime']`` with ``scvelo.tl.velocity_pseudotime`` when the
    method left none -- needed only to reproduce the Genome Biology benchmark's
    CTO, see :func:`~veloeval.metrics.cto`.  With *spatial_key* it also writes
    the physical-space kNN, see :func:`build_spatial_neighbors`; it is never
    built just because ``obsm['spatial']`` exists.  Everything it does is
    recorded in ``uns['veloeval']['prepared']``.

    See :func:`build_neighbor_indices` for *n_neighbors*, *n_pcs*, *use_rep* and
    *overwrite_neighbors*.

    Parameters
    ----------
    reference : anndata.AnnData, optional
        A shared reference for every method on the dataset: all cells,
        ``obsm['X_pca']`` and ``obsm['X_{basis}']``, the output of
        :func:`scanpy.pp.neighbors` (``obsp['distances']``,
        ``obsp['connectivities']``, ``uns['neighbors']``) and ``layers['Ms']``
        computed on that graph.  The method's own neighbour graph, PCA,
        embedding, velocity graph and projection are then replaced by ones
        derived from the reference: its cells are looked up by name; if it
        dropped some, the graph is rebuilt on their reference PCA with the
        reference's parameters.  In gene space the velocity graph is rebuilt
        with the reference ``Ms`` on the genes both have; in latent space with
        the method's own ``layers['Ms']`` on the reference neighbours.  Only
        *adata* changes, not the file it was read from.  Cannot be combined
        with *overwrite_neighbors*.
    """
    if reference is not None and overwrite_neighbors:
        raise ValueError(
            "reference= already fixes the neighbours; drop overwrite_neighbors"
        )
    set_velocity_space(adata, space)
    if reference is not None:
        dropped = _use_reference(adata, reference, basis)
        adata.obsm.pop(f"{vkey}_{basis}", None)
    build_neighbor_indices(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        overwrite=overwrite_neighbors,
    )
    if reference is not None:
        adata.uns["veloeval"]["prepared"]["neighbors"]["source"] = "reference"
        _record(
            adata,
            "reference",
            {
                "n_obs": int(reference.n_obs),
                "dropped": int(dropped),
                "graph": "rebuilt on reference PCA" if dropped else "copied",
            },
        )
        graph = _reference_velocity_graph(adata, reference, vkey, space)
        _record(adata, f"{vkey}_graph", graph)

    if spatial_key is not None:
        build_spatial_neighbors(adata, spatial_key=spatial_key)

    try:
        project_velocity(adata, basis=basis, vkey=vkey)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        _record(adata, f"{vkey}_{basis}", {"error": f"{type(exc).__name__}: {exc}"})

    if transition:
        try:
            import scvelo as scv

            adata.obsp["T_fwd"] = scv.utils.get_transition_matrix(adata, vkey=vkey)
            _record(adata, "T_fwd", {"scvelo": scv.__version__})
        except Exception as exc:  # noqa: BLE001
            _record(adata, "T_fwd", {"error": f"{type(exc).__name__}: {exc}"})

    ptime = f"{vkey}_pseudotime"
    if pseudotime and ptime not in adata.obs:
        try:
            import scvelo as scv

            scv.tl.velocity_pseudotime(adata, vkey=vkey)
            _record(adata, ptime, {"scvelo": scv.__version__})
        except Exception as exc:  # noqa: BLE001
            _record(adata, ptime, {"error": f"{type(exc).__name__}: {exc}"})


def _record(adata, field: str, params: dict[str, Any]) -> None:
    store = adata.uns.setdefault("veloeval", {}).setdefault("prepared", {})
    store[field] = {k: ("" if v is None else v) for k, v in params.items()}
