"""Public type definitions for CFA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CFAResult:
    """Full result of a CFA augmentation pass.

    Attributes
    ----------
    X_resampled:
        Features after augmentation, shape ``(N + K, D)``. The first ``N``
        rows are the original data in their original order; the last ``K``
        rows are the synthetic counterfactuals.
    y_resampled:
        Labels after augmentation, shape ``(N + K,)``.
    synthetic_mask:
        Boolean array of length ``N + K`` — ``True`` where the corresponding
        row is a synthetic counterfactual (i.e. only for the last ``K``).
    minority_label:
        The label value that CFA identified as the minority class.
    n_synthetic:
        Number of synthetic rows generated (``K``).
    """

    X_resampled: np.ndarray
    y_resampled: np.ndarray
    synthetic_mask: np.ndarray
    minority_label: Any
    n_synthetic: int
