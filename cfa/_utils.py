"""Internal helpers for CFA."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def split_classes(
    X: NDArray[np.floating],
    y: NDArray[Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]:
    """Split a binary-class dataset into (majority, minority) by class count.

    Raises
    ------
    ValueError
        If the dataset has fewer than two classes, or more than two
        (CFA is a strictly binary method in the paper; multi-class users
        should convert via 1v1 / 1vR first).
    """
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        msg = f"CFA requires at least two classes; got {len(unique)}."
        raise ValueError(msg)
    if len(unique) > 2:
        msg = (
            f"CFA is a binary-class method; got {len(unique)} classes: "
            f"{unique.tolist()!r}. Convert to binary via 1v1 or 1vR first."
        )
        raise ValueError(msg)
    # argmax/argmin return the first occurrence on ties → perfectly balanced
    # datasets would pick the same class for both labels, silently dropping
    # one of the classes. CFA's purpose is imbalanced data anyway, so refuse.
    if counts.max() == counts.min():
        msg = (
            f"CFA requires an imbalanced binary dataset; the two classes have "
            f"equal counts ({unique.tolist()!r} -> {counts.max()} each) so there is "
            f"no unambiguous majority/minority."
        )
        raise ValueError(msg)
    majority_label = unique[np.argmax(counts)]
    minority_label = unique[np.argmin(counts)]
    X_majority = X[y == majority_label]
    X_minority = X[y == minority_label]
    return X_majority, X_minority, majority_label, minority_label


def compute_feature_tolerance(
    X: NDArray[np.floating],
    tolerance: float,
) -> NDArray[np.floating]:
    """Per-feature matching tolerance.

    The paper (Temraz & Keane 2022, §3.2) defines a feature as "matching"
    between two instances if the absolute difference is within a fraction
    (``tolerance``, default 0.1 = 10%) of the standard deviation of that
    feature across the full dataset.

    Returns
    -------
    np.ndarray
        Shape ``(D,)`` per-feature tolerance ``tolerance * std(X, axis=0)``.

    Notes
    -----
    The pre-refactor implementation computed ``tolerance * std + mean`` and
    then never used the result, instead comparing against the scalar
    ``self.tol``. Both were incorrect; this function restores the paper's
    definition.
    """
    return np.asarray(tolerance * np.std(X, axis=0), dtype=np.float64)


def match_mask(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    tol: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Elementwise boolean mask: ``True`` where ``|a - b| <= tol``.

    Supports 1-D (single pair) or 2-D (batched) ``a`` and ``b``. ``tol`` is
    broadcast along the leading dimension.
    """
    return np.abs(a - b) <= tol
