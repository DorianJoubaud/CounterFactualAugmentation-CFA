"""Tests for ``cfa._algorithm`` — the algorithmic core."""

from __future__ import annotations

import numpy as np

from cfa._algorithm import find_cf_pairs, generate_synthetic_instance


class TestGenerateSynthetic:
    def test_paper_transfer_rule_exactly(self) -> None:
        """Paper §3.2, Step 3: synthetic ``p'`` takes ``p``'s values on
        difference-features and ``x'``'s values on match-features."""
        tol = np.array([0.1, 0.1, 0.1])
        x_paired = np.array([10.0, 20.0, 30.0])
        # Feature 1 differs (20 vs 99); 0 and 2 match (within tol).
        p_paired = np.array([10.05, 99.0, 29.95])
        x_unpaired = np.array([-1.0, -2.0, -3.0])
        result = generate_synthetic_instance(x_unpaired, x_paired, p_paired, tol)
        # Match-features → x_unpaired; diff-features → p_paired.
        expected = np.array([-1.0, 99.0, -3.0])
        np.testing.assert_allclose(result, expected)

    def test_all_features_match(self) -> None:
        """If every feature matches, synthetic equals ``x_unpaired``."""
        tol = np.array([1.0, 1.0])
        x_paired = np.array([1.0, 2.0])
        p_paired = np.array([1.1, 2.1])
        x_unpaired = np.array([100.0, 200.0])
        result = generate_synthetic_instance(x_unpaired, x_paired, p_paired, tol)
        np.testing.assert_allclose(result, x_unpaired)

    def test_no_features_match(self) -> None:
        """If no feature matches, synthetic equals ``p_paired``."""
        tol = np.array([0.01, 0.01])
        x_paired = np.array([1.0, 2.0])
        p_paired = np.array([5.0, 10.0])
        x_unpaired = np.array([100.0, 200.0])
        result = generate_synthetic_instance(x_unpaired, x_paired, p_paired, tol)
        np.testing.assert_allclose(result, p_paired)

    def test_not_pre_refactor_additive_formula(self) -> None:
        """Regression guard: ``x' + (p - x)`` would give a different answer."""
        tol = np.array([0.1, 0.1])
        x_paired = np.array([10.0, 20.0])
        p_paired = np.array([10.05, 99.0])  # feat 0 match, feat 1 diff
        x_unpaired = np.array([-1.0, -2.0])
        result = generate_synthetic_instance(x_unpaired, x_paired, p_paired, tol)
        additive = x_unpaired + (p_paired - x_paired)  # pre-refactor formula
        # Feature-0 agrees accidentally (both ≈ -1). Feature-1 disagrees.
        assert not np.allclose(result, additive)


class TestFindCFPairs:
    def test_basic_pairing(self) -> None:
        X_maj = np.array([[0.0, 0.0], [10.0, 10.0]])
        X_min = np.array([[0.1, 0.1]])
        tol = np.array([0.5, 0.5])
        maj_idx, min_idx = find_cf_pairs(X_maj, X_min, tol, max_diff_features=2)
        # Minority at [0.1, 0.1] is nearest to majority[0].
        assert len(maj_idx) == 1
        assert maj_idx[0] == 0
        assert min_idx[0] == 0

    def test_one_majority_one_minority_pair(self) -> None:
        """Each majority can be paired with at most one minority."""
        X_maj = np.array([[0.0, 0.0]])  # single majority
        X_min = np.array([[0.1, 0.1], [0.2, 0.2]])
        tol = np.array([1.0, 1.0])
        maj_idx, _ = find_cf_pairs(X_maj, X_min, tol, max_diff_features=2)
        assert len(maj_idx) == 1

    def test_no_pairs_when_too_many_diffs(self) -> None:
        X_maj = np.array([[0.0, 0.0]])
        X_min = np.array([[10.0, 10.0]])  # both features well outside tol
        tol = np.array([0.1, 0.1])
        maj_idx, _ = find_cf_pairs(X_maj, X_min, tol, max_diff_features=1)
        assert len(maj_idx) == 0

    def test_empty_majority(self) -> None:
        X_maj = np.empty((0, 2))
        X_min = np.array([[0.0, 0.0]])
        tol = np.array([0.1, 0.1])
        maj_idx, min_idx = find_cf_pairs(X_maj, X_min, tol, max_diff_features=2)
        assert len(maj_idx) == 0
        assert len(min_idx) == 0

    def test_empty_minority(self) -> None:
        X_maj = np.array([[0.0, 0.0]])
        X_min = np.empty((0, 2))
        tol = np.array([0.1, 0.1])
        maj_idx, min_idx = find_cf_pairs(X_maj, X_min, tol, max_diff_features=2)
        assert len(maj_idx) == 0
        assert len(min_idx) == 0
