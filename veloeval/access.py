"""Read-only accessors for the fields metrics need.

Two rules, both deliberate:

1. **Nothing here computes.**  If ``velocity_graph`` or ``Ms``/``Mu`` or a
   transition matrix is missing, that is a :class:`MissingInput`, not something
   to derive on the fly.  Deriving it here would silently apply *default*
   parameters to some methods and the wrapper's parameters to others, so two
   methods would no longer be on the same footing -- and nothing in the results
   would record which happened.  Derive once, explicitly, upstream:
   see :mod:`veloeval.prepare`.

2. **Velocity space is declared, not guessed.**  A latent-space velocity and a
   gene-space velocity both live in ``adata.layers["velocity"]``-shaped arrays,
   and a metric that assumes the wrong one returns a number that looks fine and
   is not comparable.  A wrong number is worse than a missing one.
"""

from __future__ import annotations

import numpy as np

from .result import MissingInput, NotApplicable

__all__ = [
    "VeloSpace",
    "velocity_space",
    "set_velocity_space",
    "get_velocity",
    "get_velocity_embedding",
    "get_embedding",
    "get_neighbor_indices",
    "get_transition_matrix",
    "get_labels",
    "gene_coverage",
]

#: Where a method's velocity vectors live.
#:
#: ``gene``      one component per gene (scVelo, velocyto, veloVI, ...)
#: ``latent``    a learned low-dimensional space (VeloAE, VeloVAE, ...)
#: ``embedding`` directly in a 2-D visualisation basis
#: ``scalar``    a single signed rate per cell (VeloCycle angular velocity)
VeloSpace = str

_UNS = "veloeval"
_SPACE = "velocity_space"


def velocity_space(adata, default: VeloSpace = "gene") -> VeloSpace:
    """Return the declared velocity space, falling back to *default*."""
    return adata.uns.get(_UNS, {}).get(_SPACE, default)


def set_velocity_space(adata, space: VeloSpace) -> None:
    """Declare the velocity space.  Call this in the method wrapper."""
    if space not in ("gene", "latent", "embedding", "scalar"):
        raise ValueError(f"unknown velocity space: {space!r}")
    adata.uns.setdefault(_UNS, {})[_SPACE] = space


def _dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


def get_velocity(
    adata,
    vkey: str = "velocity",
    *,
    allowed_spaces: tuple[str, ...] = ("gene",),
    drop_nan_genes: bool = True,
) -> np.ndarray:
    """Velocity matrix ``(n_cells, n_features)`` in its native space.

    Genes carrying *any* NaN are dropped entirely rather than zero-filled:
    zero-filling injects false "no change" information and biases every
    distance-based metric.  Gene loss is a real cost of the method and is
    reported separately by :func:`gene_coverage`.
    """
    space = velocity_space(adata)
    if space not in allowed_spaces:
        raise NotApplicable(
            f"velocity lives in {space!r} space; this metric needs one of "
            f"{allowed_spaces}"
        )

    for key in (vkey, f"{vkey}_S"):
        if key in adata.layers:
            V = _dense(adata.layers[key])
            if drop_nan_genes:
                V = V[:, ~np.isnan(V).any(axis=0)]
            if V.shape[1] == 0:
                raise NotApplicable("every feature had NaN velocity")
            return V
    raise MissingInput(f"layers['{vkey}']")


def gene_coverage(adata, vkey: str = "velocity") -> tuple[int, int]:
    """``(n_valid, n_total)`` velocity features after dropping NaN columns."""
    for key in (vkey, f"{vkey}_S"):
        if key in adata.layers:
            V = _dense(adata.layers[key])
            return int((~np.isnan(V).any(axis=0)).sum()), int(V.shape[1])
    raise MissingInput(f"layers['{vkey}']")


def get_velocity_embedding(adata, basis: str = "umap", vkey: str = "velocity") -> np.ndarray:
    """Velocity projected into ``obsm['{vkey}_{basis}']``.

    Not computed here -- see the module docstring.  Produce it in the pipeline
    with :func:`veloeval.prepare.project_velocity`.
    """
    key = f"{vkey}_{basis}"
    if key not in adata.obsm:
        raise MissingInput(f"obsm['{key}']")
    return np.asarray(adata.obsm[key], dtype=np.float64)


def get_embedding(adata, basis: str = "umap") -> np.ndarray:
    key = f"X_{basis}"
    if key not in adata.obsm:
        raise MissingInput(f"obsm['{key}']")
    return np.asarray(adata.obsm[key], dtype=np.float64)


def get_neighbor_indices(adata) -> np.ndarray:
    """kNN index array ``(n_cells, k)`` from ``uns['neighbors']['indices']``."""
    nn = adata.uns.get("neighbors", {})
    if "indices" not in nn:
        raise MissingInput("uns['neighbors']['indices']")
    return np.asarray(nn["indices"])


def get_transition_matrix(adata, key: str = "T_fwd"):
    """Row-stochastic cell-cell transition matrix from ``obsp[key]``."""
    if key not in adata.obsp:
        raise MissingInput(f"obsp['{key}']")
    return adata.obsp[key]


def get_labels(adata, label_key: str | None) -> np.ndarray:
    if not label_key:
        raise NotApplicable("no cell-type labels for this dataset")
    if label_key not in adata.obs:
        raise MissingInput(f"obs['{label_key}']")
    return np.asarray(adata.obs[label_key].values)
