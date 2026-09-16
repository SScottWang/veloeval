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

from .access import set_velocity_space

__all__ = ["prepare", "project_velocity", "build_neighbor_indices"]


def build_neighbor_indices(adata, n_neighbors: int = 30, use_rep: str | None = None):
    """Write ``uns['neighbors']['indices']`` -- the kNN every local metric uses.

    Uses the same representation for every method so that "neighbourhood"
    means the same thing across the comparison.
    """
    from sklearn.neighbors import NearestNeighbors

    if use_rep and use_rep in adata.obsm:
        X = np.asarray(adata.obsm[use_rep])
    elif "X_pca" in adata.obsm:
        X = np.asarray(adata.obsm["X_pca"])
    else:
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    k = min(n_neighbors + 1, adata.n_obs)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    indices = nn.kneighbors(X, return_distance=False)

    adata.uns.setdefault("neighbors", {})["indices"] = indices
    return indices


def project_velocity(adata, basis: str = "umap", vkey: str = "velocity", **kwargs):
    """Write ``obsm['{vkey}_{basis}']`` via scVelo's embedding projection.

    Records the scVelo version and the graph parameters, because the projection
    depends on both.
    """
    import scvelo as scv

    if "velocity_graph" not in adata.uns:
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
    use_rep: str | None = None,
    transition: bool = True,
) -> None:
    """One call at the end of a method wrapper.

    Declares the velocity space, builds the shared kNN, projects velocity into
    *basis*, and (optionally) writes a transition matrix to ``obsp['T_fwd']``.
    Everything it does is recorded in ``uns['veloeval']['prepared']``.
    """
    set_velocity_space(adata, space)
    build_neighbor_indices(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    _record(adata, "neighbors", {"n_neighbors": n_neighbors, "use_rep": use_rep})

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


def _record(adata, field: str, params: dict[str, Any]) -> None:
    store = adata.uns.setdefault("veloeval", {}).setdefault("prepared", {})
    store[field] = {k: ("" if v is None else v) for k, v in params.items()}
