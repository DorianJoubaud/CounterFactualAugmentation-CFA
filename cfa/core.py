"""CFA public API.

Entry point for the Counterfactual Data Augmentation method of
Temraz & Keane (2022). See :class:`CFA` for usage.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from ._algorithm import find_cf_pairs, generate_synthetic_instance
from ._utils import compute_feature_tolerance, split_classes
from .types import CFAResult


class CFA:
    """Counterfactual Data Augmentation for binary class imbalance.

    Implements the algorithm of Temraz & Keane (2022), *"Solving the Class
    Imbalance Problem Using a Counterfactual Method for Data Augmentation"*,
    *Machine Learning with Applications* 9, 100375.
    `doi:10.1016/j.mlwa.2022.100375 <https://doi.org/10.1016/j.mlwa.2022.100375>`_.

    The method generates minority-class synthetics by transferring
    feature-values from existing counterfactual pairs onto unpaired
    majority instances, using the actual values in the dataset rather than
    interpolations (unlike SMOTE and its variants).

    Parameters
    ----------
    max_diff_features:
        Maximum number of difference-features allowed in a "good" native
        counterfactual pair. The paper uses ``2``.
    tolerance:
        Per-feature matching tolerance, expressed as a fraction of the
        feature's standard deviation across the dataset. The paper uses
        ``0.1`` (i.e. ±10 % of the std).
    random_state:
        Accepted for scikit-learn-style API compatibility. CFA is
        deterministic given the KNN neighbour order, so this argument is
        currently unused.

    Examples
    --------
    >>> from cfa import CFA
    >>> cfa = CFA(max_diff_features=2, tolerance=0.1)
    >>> X_resampled, y_resampled = cfa.fit_resample(X, y)  # imblearn-style

    To access full metadata (synthetic mask, number of synthetics, minority
    label), use :meth:`run_cfa`, which returns a :class:`CFAResult`.
    """

    def __init__(
        self,
        max_diff_features: int = 2,
        tolerance: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        if max_diff_features < 0:
            msg = f"max_diff_features must be >= 0; got {max_diff_features}"
            raise ValueError(msg)
        if tolerance < 0:
            msg = f"tolerance must be >= 0; got {tolerance}"
            raise ValueError(msg)
        self.max_diff_features = max_diff_features
        self.tolerance = tolerance
        self.random_state = random_state

    def run_cfa(
        self,
        X: NDArray[np.floating] | list,
        y: NDArray[Any] | list,
    ) -> CFAResult:
        """Run CFA and return a :class:`CFAResult` with full metadata."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            msg = f"X must be 2-D (n_samples, n_features); got shape {X_arr.shape}"
            raise ValueError(msg)
        if y_arr.ndim != 1:
            msg = f"y must be 1-D; got shape {y_arr.shape}"
            raise ValueError(msg)
        if X_arr.shape[0] != y_arr.shape[0]:
            msg = (
                f"X and y must have the same number of samples; "
                f"got {X_arr.shape[0]} and {y_arr.shape[0]}"
            )
            raise ValueError(msg)

        X_maj, X_min, _maj_label, min_label = split_classes(X_arr, y_arr)
        tol_per_feature = compute_feature_tolerance(X_arr, self.tolerance)

        # Step 1 — compute the CF-Set.
        maj_idx, min_idx = find_cf_pairs(
            X_maj, X_min, tol_per_feature, self.max_diff_features
        )

        if len(maj_idx) == 0:
            warnings.warn(
                "CFA: no native counterfactual pairs found. "
                "Consider increasing `tolerance` or `max_diff_features`. "
                "Returning the original data unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return self._empty_result(X_arr, y_arr, min_label)

        # Step 2 — identify unpaired majority instances.
        unpaired_mask = np.ones(X_maj.shape[0], dtype=bool)
        unpaired_mask[maj_idx] = False
        unpaired_local_indices = np.nonzero(unpaired_mask)[0]

        if len(unpaired_local_indices) == 0:
            warnings.warn(
                "CFA: every majority instance is paired, no synthetics can "
                "be generated. Returning the original data unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return self._empty_result(X_arr, y_arr, min_label)

        # For each unpaired majority, find the nearest *paired* majority.
        paired_majorities = X_maj[maj_idx]  # (Npaired, D)
        nn_paired = NearestNeighbors(n_neighbors=1).fit(paired_majorities)
        _, nearest_paired_slot = nn_paired.kneighbors(X_maj[unpaired_local_indices])
        nearest_paired_slot = nearest_paired_slot[:, 0]  # slot index into maj_idx

        # Step 3 — generate synthetics.
        synthetics = np.empty(
            (len(unpaired_local_indices), X_arr.shape[1]), dtype=np.float64
        )
        for out_i, (unpaired_local, pair_slot) in enumerate(
            zip(unpaired_local_indices, nearest_paired_slot, strict=True)
        ):
            x_unpaired = X_maj[unpaired_local]
            x_paired = X_maj[maj_idx[pair_slot]]
            p_paired = X_min[min_idx[pair_slot]]
            synthetics[out_i] = generate_synthetic_instance(
                x_unpaired, x_paired, p_paired, tol_per_feature
            )

        n_syn = synthetics.shape[0]
        X_out = np.concatenate([X_arr, synthetics], axis=0)
        y_syn = np.full(n_syn, min_label, dtype=y_arr.dtype)
        y_out = np.concatenate([y_arr, y_syn], axis=0)
        synthetic_mask = np.concatenate(
            [np.zeros(len(X_arr), dtype=bool), np.ones(n_syn, dtype=bool)]
        )

        return CFAResult(
            X_resampled=X_out,
            y_resampled=y_out,
            synthetic_mask=synthetic_mask,
            minority_label=min_label,
            n_synthetic=n_syn,
        )

    def fit_resample(
        self,
        X: NDArray[np.floating] | list,
        y: NDArray[Any] | list,
    ) -> tuple[NDArray[np.floating], NDArray[Any]]:
        """``imbalanced-learn``-compatible resampling interface.

        Returns the resampled ``(X, y)`` directly so CFA drops into imblearn
        :class:`~imblearn.pipeline.Pipeline` objects (the ``fit_resample``
        contract is duck-typed — no imblearn dependency is required).
        """
        result = self.run_cfa(X, y)
        return result.X_resampled, result.y_resampled

    @staticmethod
    def _empty_result(
        X_arr: NDArray[np.floating],
        y_arr: NDArray[Any],
        min_label: Any,
    ) -> CFAResult:
        return CFAResult(
            X_resampled=X_arr.copy(),
            y_resampled=y_arr.copy(),
            synthetic_mask=np.zeros(len(X_arr), dtype=bool),
            minority_label=min_label,
            n_synthetic=0,
        )
