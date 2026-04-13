"""Core CFA algorithm primitives — pair discovery and synthesis.

Implements Steps 1 and 3 of the algorithm in Temraz & Keane (2022), §3.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from ._utils import match_mask


def find_cf_pairs(
    X_majority: NDArray[np.floating],
    X_minority: NDArray[np.floating],
    tol_per_feature: NDArray[np.floating],
    max_diff_features: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Find "good" native counterfactual pairs (``cf(x, p)``).

    For every minority instance ``p``, scan majority instances in ascending
    order of Euclidean distance. Accept the first unpaired majority whose
    difference-feature count (features outside ``tol_per_feature``) is at
    most ``max_diff_features``. This is greedy one-to-one matching.

    Parameters
    ----------
    X_majority:
        Majority-class instances, shape ``(Nmaj, D)``.
    X_minority:
        Minority-class instances, shape ``(Nmin, D)``.
    tol_per_feature:
        Per-feature tolerance threshold, shape ``(D,)``.
    max_diff_features:
        Maximum allowed difference-features for a "good" counterfactual
        (the paper's default is 2).

    Returns
    -------
    (majority_idx, minority_idx):
        Parallel arrays of accepted pair indices. Length is ``≤ min(Nmaj, Nmin)``.
    """
    if X_majority.shape[0] == 0 or X_minority.shape[0] == 0:
        empty = np.empty((0,), dtype=np.intp)
        return empty, empty

    # Single KNN fit on majority, queried **per minority** rather than in a batch,
    # so the full (Nmin x Nmaj) neighbour matrix is never materialised — each
    # iteration only allocates an (Nmaj,) row. The pre-refactor version re-fit
    # KNN inside the loop, which was O(Nmin * Nmaj^2) and the dominant cost.
    n_maj = X_majority.shape[0]
    nn = NearestNeighbors(n_neighbors=n_maj).fit(X_majority)

    majority_paired: set[int] = set()
    majority_indices: list[int] = []
    minority_indices: list[int] = []

    for min_idx in range(X_minority.shape[0]):
        p = X_minority[min_idx]
        _, neighbour_row = nn.kneighbors(p[np.newaxis, :])  # (1, Nmaj)
        for maj_idx_raw in neighbour_row[0]:
            maj_idx = int(maj_idx_raw)
            if maj_idx in majority_paired:
                continue
            x = X_majority[maj_idx]
            diff_count = int(np.sum(~match_mask(x, p, tol_per_feature)))
            if diff_count <= max_diff_features:
                majority_paired.add(maj_idx)
                majority_indices.append(maj_idx)
                minority_indices.append(min_idx)
                break

    return (
        np.asarray(majority_indices, dtype=np.intp),
        np.asarray(minority_indices, dtype=np.intp),
    )


def generate_synthetic_instance(
    x_unpaired: NDArray[np.floating],
    x_paired: NDArray[np.floating],
    p_paired: NDArray[np.floating],
    tol_per_feature: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Synthesize one counterfactual ``p'`` (paper §3.2, Step 3).

    Transfer rule, verbatim from the paper:

    * For each **difference-feature** between ``x_paired`` and ``p_paired``:
      take the value from ``p_paired`` into ``p'``.
    * For each **match-feature**: take the value from ``x_unpaired`` into ``p'``.

    A feature is a match if ``|x_paired[j] - p_paired[j]| <= tol_per_feature[j]``.

    Notes
    -----
    The pre-refactor implementation used ``x_unpaired + (p_paired - x_paired)``,
    which produces a different result: for match-features it gives
    ``x_unpaired[j] + (p_j - x_j)`` (approximately ``x_unpaired[j]`` only when
    ``x_j == p_j``), and for difference-features it gives
    ``x_unpaired[j] + (p_j - x_j)`` rather than ``p_j``. This function restores
    the paper's definition.
    """
    matches = match_mask(x_paired, p_paired, tol_per_feature)
    return np.where(matches, x_unpaired, p_paired)
