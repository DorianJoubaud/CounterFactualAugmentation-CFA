"""Tests for ``cfa._utils``."""

from __future__ import annotations

import numpy as np
import pytest

from cfa._utils import compute_feature_tolerance, match_mask, split_classes


class TestSplitClasses:
    def test_basic_binary(self) -> None:
        X = np.array([[0.0], [0.0], [0.0], [1.0]])
        y = np.array([0, 0, 0, 1])
        X_maj, X_min, maj_label, min_label = split_classes(X, y)
        assert maj_label == 0
        assert min_label == 1
        assert len(X_maj) == 3
        assert len(X_min) == 1

    def test_raises_on_single_class(self) -> None:
        X = np.array([[0.0], [0.0]])
        y = np.array([0, 0])
        with pytest.raises(ValueError, match="at least two classes"):
            split_classes(X, y)

    def test_raises_on_multiclass(self) -> None:
        X = np.array([[0.0], [0.0], [0.0]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="binary-class"):
            split_classes(X, y)

    def test_string_labels(self) -> None:
        X = np.array([[0.0], [0.0], [1.0]])
        y = np.array(["A", "A", "B"])
        _, _, maj_label, min_label = split_classes(X, y)
        assert maj_label == "A"
        assert min_label == "B"

    def test_raises_on_balanced_classes(self) -> None:
        """Regression guard: argmax/argmin on a tie returns the same index,
        which would silently collapse both labels to the same class."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="imbalanced"):
            split_classes(X, y)


class TestFeatureTolerance:
    def test_formula_matches_paper(self) -> None:
        """Paper: tolerance = fraction * std. The pre-refactor formula also
        added the mean, which made the tolerance location-dependent."""
        X = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        tol = compute_feature_tolerance(X, tolerance=0.1)
        expected = 0.1 * np.std(X, axis=0)
        np.testing.assert_allclose(tol, expected)

    def test_differs_from_pre_refactor_formula(self) -> None:
        """Regression guard: the old ``tol*std + mean`` is NOT the paper's."""
        X = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        tol = compute_feature_tolerance(X, tolerance=0.1)
        buggy = 0.1 * np.std(X, axis=0) + np.mean(X, axis=0)
        assert not np.allclose(tol, buggy)

    def test_zero_tolerance(self) -> None:
        X = np.array([[0.0], [1.0]])
        tol = compute_feature_tolerance(X, tolerance=0.0)
        np.testing.assert_array_equal(tol, [0.0])


class TestMatchMask:
    def test_exact_match(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        tol = np.array([0.1, 0.1, 0.1])
        np.testing.assert_array_equal(match_mask(a, b, tol), [True, True, True])

    def test_within_tolerance(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([1.05, 2.2])
        tol = np.array([0.1, 0.1])
        np.testing.assert_array_equal(match_mask(a, b, tol), [True, False])

    def test_edge_equality(self) -> None:
        """Exactly on the tolerance threshold counts as match."""
        a = np.array([0.0])
        b = np.array([0.1])
        tol = np.array([0.1])
        assert match_mask(a, b, tol)[0]
