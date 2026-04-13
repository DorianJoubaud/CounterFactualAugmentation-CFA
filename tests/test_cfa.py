"""Integration tests for the ``CFA`` class."""

from __future__ import annotations

import numpy as np
import pytest

from cfa import CFA, CFAResult


class TestCFAInit:
    def test_defaults(self) -> None:
        cfa = CFA()
        assert cfa.max_diff_features == 2
        assert cfa.tolerance == 0.1

    def test_negative_max_diff_features_raises(self) -> None:
        with pytest.raises(ValueError, match="max_diff_features"):
            CFA(max_diff_features=-1)

    def test_negative_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            CFA(tolerance=-0.1)


class TestFitResample:
    def test_returns_two_arrays(self, binary_imbalanced_data) -> None:
        X, y = binary_imbalanced_data
        cfa = CFA(max_diff_features=2, tolerance=1.0)  # generous
        X_out, y_out = cfa.fit_resample(X, y)
        assert X_out.ndim == 2
        assert y_out.ndim == 1
        assert X_out.shape[0] == y_out.shape[0]

    def test_originals_preserved_in_prefix(self, binary_imbalanced_data) -> None:
        X, y = binary_imbalanced_data
        cfa = CFA(tolerance=1.0)
        X_out, y_out = cfa.fit_resample(X, y)
        np.testing.assert_array_equal(X_out[: len(X)], X)
        np.testing.assert_array_equal(y_out[: len(y)], y)

    def test_synthetic_labels_are_minority(self, overlapping_imbalanced_data) -> None:
        X, y = overlapping_imbalanced_data
        cfa = CFA(max_diff_features=3, tolerance=1.0)  # permissive, force synthetics
        _X_out, y_out = cfa.fit_resample(X, y)
        n_syn = len(y_out) - len(y)
        assert n_syn > 0, "test assumes synthetics are generated"
        assert np.all(y_out[-n_syn:] == 1)

    def test_synthetic_labels_use_actual_minority_label(self) -> None:
        """Regression guard: the pre-refactor code hardcoded synthetic
        labels to ``1`` in the ``get_synt_labels=False`` branch. Fixed to
        use the detected minority label."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(3, 0.3, (5, 2))])
        y = np.array(["A"] * 20 + ["B"] * 5)  # minority is "B", not 1
        cfa = CFA(max_diff_features=2, tolerance=2.0)  # very permissive
        _X_out, y_out = cfa.fit_resample(X, y)
        n_syn = len(y_out) - len(y)
        assert n_syn > 0
        assert np.all(y_out[-n_syn:] == "B")


class TestRunCFA:
    def test_returns_cfaresult(self, binary_imbalanced_data) -> None:
        X, y = binary_imbalanced_data
        result = CFA(tolerance=1.0).run_cfa(X, y)
        assert isinstance(result, CFAResult)
        assert result.synthetic_mask.dtype == bool
        assert len(result.synthetic_mask) == len(result.X_resampled)

    def test_synthetic_mask_marks_only_appended_rows(
        self, overlapping_imbalanced_data
    ) -> None:
        X, y = overlapping_imbalanced_data
        result = CFA(max_diff_features=3, tolerance=1.0).run_cfa(X, y)
        # First len(X) entries are originals → mask must be all False there.
        assert not result.synthetic_mask[: len(X)].any()
        # Tail corresponds to synthetics → all True.
        if result.n_synthetic > 0:
            assert result.synthetic_mask[len(X) :].all()

    def test_warns_when_no_pairs_found(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.01, (10, 2)), rng.normal(100, 0.01, (3, 2))])
        y = np.concatenate([np.zeros(10, dtype=int), np.ones(3, dtype=int)])
        cfa = CFA(max_diff_features=0, tolerance=1e-9)  # impossible to match
        with pytest.warns(UserWarning, match="no native counterfactual pairs"):
            result = cfa.run_cfa(X, y)
        assert result.n_synthetic == 0
        np.testing.assert_array_equal(result.X_resampled, X)


class TestEdgeCases:
    def test_multiclass_raises(self) -> None:
        X = np.array([[0.0], [0.0], [1.0], [2.0]])
        y = np.array([0, 1, 2, 2])
        with pytest.raises(ValueError, match="binary-class"):
            CFA().run_cfa(X, y)

    def test_single_class_raises(self) -> None:
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 0])
        with pytest.raises(ValueError, match="at least two classes"):
            CFA().run_cfa(X, y)

    def test_invalid_x_shape_raises(self) -> None:
        X = np.array([0.0, 1.0])  # 1-D
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="2-D"):
            CFA().run_cfa(X, y)

    def test_length_mismatch_raises(self) -> None:
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="same number"):
            CFA().run_cfa(X, y)

    def test_accepts_python_lists(self) -> None:
        """Duck-type input: plain Python lists should work."""
        X = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [1.0, 1.0]]
        y = [0, 0, 0, 1]
        X_out, _y_out = CFA(tolerance=2.0).fit_resample(X, y)
        assert X_out.shape[0] >= len(X)
