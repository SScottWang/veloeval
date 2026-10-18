"""ConstVelo: one gene-space velocity, the same for every cell.

A probe of the evaluation, not a method: with the velocity held fixed, any
change in a score across neighbour graphs or layouts comes from the evaluation
alone.  It is not ranked.
"""

from __future__ import annotations

import numpy as np

from .access import set_velocity_space


def constvelo(
    adata,
    *,
    start: str,
    end: str,
    label_key: str = "clusters",
    xkey: str = "Ms",
    n_pcs: int = 30,
    n_neighbors: int = 30,
    vkey: str = "velocity",
) -> np.ndarray:
    r"""Give every cell the same unit velocity, from cluster *start* to *end*.

    .. math::

        c = \overline{M_s}[\text{end}] - \overline{M_s}[\text{start}],
        \qquad V_{i,:} = c / \lVert c \rVert \quad \text{for every cell } i,

    the means taken over all cells of each cluster, on every gene of *adata*
    -- subset to the highly variable genes first.  Writes ``layers[vkey]``
    (float32), marks every gene in ``var['{vkey}_genes']`` so scVelo's
    velocity graph uses them all, declares gene space and records the inputs
    in ``uns['veloeval']['constvelo']``.

    Compute *c* once, on the reference, and give the same matrix to every
    configuration being compared.  Recomputing it from each configuration's
    own ``Ms`` would let the velocity follow the reference, which is what the
    probe holds fixed.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells with ``obs[label_key]``, and ``layers[xkey]`` or (to build it)
        ``layers['spliced']`` and ``layers['unspliced']``.
    start, end : str
        The two clusters of ``obs[label_key]`` the direction runs between.
    label_key : str, default: ``"clusters"``
    xkey : str, default: ``"Ms"``
        The layer the cluster means are taken on; the space scVelo's velocity
        graph compares velocity with.  Used as it is if present.
    n_pcs, n_neighbors : int, default: 30
        Only when ``layers['Ms']`` is absent: it is then built with
        ``scanpy.pp.pca(n_comps=n_pcs)``, ``scanpy.pp.neighbors(n_neighbors,
        n_pcs)`` and ``scvelo.pp.moments``, replacing ``obsm['X_pca']`` and the
        neighbour graph.
    vkey : str, default: ``"velocity"``

    Returns
    -------
    numpy.ndarray
        The unit direction *c*, one entry per gene.

    Notes
    -----
    In a multi-lineage dataset one direction is right only roughly, and only
    for some cells: present it as an idealised example.  Every direction-only
    score is unchanged when *V* is rescaled, so the unit norm is a convention.

    Examples
    --------
    .. code-block:: python

        import veloeval as ve

        c = ve.constvelo(ref, start="Ductal", end="Beta")   # on the reference
        for adata in configurations:                        # same c for all
            adata.layers["velocity"] = np.tile(c, (adata.n_obs, 1))
            adata.var["velocity_genes"] = True
    """
    if label_key not in adata.obs:
        raise KeyError(f"obs['{label_key}']")
    labels = adata.obs[label_key].to_numpy()
    for name in (start, end):
        if not (labels == name).any():
            raise ValueError(f"no cell of obs['{label_key}'] is {name!r}")

    built = xkey not in adata.layers
    if built:
        if xkey != "Ms":
            raise KeyError(f"layers['{xkey}']")
        _build_moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    X = adata.layers[xkey]
    c = _mean(X[labels == end]) - _mean(X[labels == start])
    norm = np.linalg.norm(c)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"mean {xkey} of {start!r} and {end!r} coincide")
    c /= norm

    adata.layers[vkey] = np.tile(c.astype(np.float32), (adata.n_obs, 1))
    adata.var[f"{vkey}_genes"] = True
    set_velocity_space(adata, "gene")
    record = {
        "start": start,
        "end": end,
        "label_key": label_key,
        "xkey": xkey,
        "n_genes": int(adata.n_vars),
        "raw_norm": float(norm),
    }
    if built:
        record.update(n_pcs=n_pcs, n_neighbors=n_neighbors)
    adata.uns.setdefault("veloeval", {})["constvelo"] = record
    return c


def _mean(X) -> np.ndarray:
    return np.asarray(X.mean(axis=0), dtype=np.float64).ravel()


def _build_moments(adata, *, n_pcs: int, n_neighbors: int) -> None:
    import scanpy as sc

    try:
        import scvelo as scv
    except ImportError:
        raise ImportError(
            "building Ms needs scvelo (pip install veloeval[prepare])"
        ) from None

    for layer in ("spliced", "unspliced"):
        if layer not in adata.layers:
            raise KeyError(f"layers['{layer}'] (needed to build Ms)")
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    # n_neighbors=None: smooth over the graph just built, not a pruned one.
    scv.pp.moments(adata, n_neighbors=None, n_pcs=None)
