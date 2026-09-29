"""Explicit, recorded derivation of the fields metrics read.

The metrics layer never computes these.  If it did, a method whose wrapper
already built a velocity graph would be scored on its own graph while a method
that did not would be scored on a default one -- and the results table would
not say which.  So derivation happens here, once, at the end of the velocity
module, with identical parameters for every method, and what was done is
written into ``adata.uns["veloeval"]["prepared"]``.

Call :func:`prepare` at the end of each method wrapper, just before writing
``velocity.h5ad``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .access import KNN_KEY, set_velocity_space

__all__ = ["prepare", "project_velocity", "build_neighbor_indices"]


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


def prepare(
    adata,
    *,
    space: str = "gene",
    basis: str = "umap",
    vkey: str = "velocity",
    n_neighbors: int = 30,
    n_pcs: int = 30,
    use_rep: str | None = None,
    overwrite_neighbors: bool = False,
    transition: bool = False,
    pseudotime: bool = False,
) -> None:
    """One call at the end of a method wrapper.

    Declares the velocity space, picks up (or builds) the kNN, projects velocity
    into *basis*, (with *transition*) writes scVelo's default transition matrix
    to ``obsp['T_fwd']`` -- no metric reads it; the negative-control metrics
    build theirs from the velocity graph -- and (with *pseudotime*) writes
    ``obs['{vkey}_pseudotime']`` with ``scvelo.tl.velocity_pseudotime`` when the
    method left none -- needed only to reproduce the Genome Biology benchmark's
    CTO, see :func:`~veloeval.metrics.cto`.  Everything it does is recorded in
    ``uns['veloeval']['prepared']``.

    See :func:`build_neighbor_indices` for *n_neighbors*, *n_pcs*, *use_rep* and
    *overwrite_neighbors*.
    """
    set_velocity_space(adata, space)
    build_neighbor_indices(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        overwrite=overwrite_neighbors,
    )

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
