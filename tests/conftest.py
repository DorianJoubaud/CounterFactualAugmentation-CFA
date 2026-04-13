"""Shared pytest fixtures for CFA tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def binary_imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """A 2-D imbalanced dataset: 20 majority (class 0) + 5 minority (class 1).

    Classes are well-separated so CFA should find native counterfactual
    pairs with modest tolerance.
    """
    rng = np.random.default_rng(42)
    X_maj = rng.normal(loc=[0.0, 0.0], scale=[0.3, 0.3], size=(20, 2))
    X_min = rng.normal(loc=[1.0, 1.0], scale=[0.3, 0.3], size=(5, 2))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(20, dtype=int), np.ones(5, dtype=int)])
    return X, y


@pytest.fixture
def overlapping_imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """An imbalanced dataset whose classes overlap in feature space — lots
    of native counterfactual pairs should exist with modest tolerance."""
    rng = np.random.default_rng(1)
    X_maj = rng.normal(loc=[0.0, 0.0, 0.0], scale=[0.5, 0.5, 0.5], size=(30, 3))
    X_min = rng.normal(loc=[0.1, 0.1, 0.1], scale=[0.5, 0.5, 0.5], size=(6, 3))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(30, dtype=int), np.ones(6, dtype=int)])
    return X, y
