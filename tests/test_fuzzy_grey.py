"""Tests for the Fuzzy-Grey Comprehensive Evaluation module (src/fuzzy_grey.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.fuzzy_grey import fuzzy_grey_scores, _fuzzy_membership, _grey_relational_coefficients


# ---------------------------------------------------------------------------
# _fuzzy_membership
# ---------------------------------------------------------------------------

class TestFuzzyMembership:
    def test_all_same_returns_ones(self):
        values = np.array([3.0, 3.0, 3.0])
        result = _fuzzy_membership(values)
        np.testing.assert_array_equal(result, np.ones(3))

    def test_linear_mapping(self):
        values = np.array([0.0, 5.0, 10.0])
        result = _fuzzy_membership(values)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_single_element(self):
        values = np.array([42.0])
        result = _fuzzy_membership(values)
        np.testing.assert_array_equal(result, np.ones(1))

    def test_output_range(self):
        rng = np.random.default_rng(0)
        values = rng.uniform(-10, 10, size=20)
        result = _fuzzy_membership(values)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_monotone_increasing(self):
        values = np.arange(10, dtype=float)
        result = _fuzzy_membership(values)
        assert np.all(np.diff(result) >= 0)


# ---------------------------------------------------------------------------
# _grey_relational_coefficients
# ---------------------------------------------------------------------------

class TestGreyRelationalCoefficients:
    def test_ideal_matrix_returns_ones(self):
        """When all indicators equal the reference (1.0) the coefficient is 1."""
        matrix = np.ones((4, 3))
        coeff = _grey_relational_coefficients(matrix, rho=0.5)
        np.testing.assert_allclose(coeff, np.ones((4, 3)))

    def test_coefficient_range(self):
        rng = np.random.default_rng(1)
        matrix = rng.uniform(0, 1, size=(10, 5))
        coeff = _grey_relational_coefficients(matrix, rho=0.5)
        assert coeff.min() > 0.0
        assert coeff.max() <= 1.0

    def test_shape_preserved(self):
        matrix = np.random.rand(6, 4)
        coeff = _grey_relational_coefficients(matrix)
        assert coeff.shape == (6, 4)

    def test_rho_effect(self):
        """Smaller rho increases the spread between high and low coefficients."""
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        coeff_low = _grey_relational_coefficients(matrix, rho=0.1)
        coeff_high = _grey_relational_coefficients(matrix, rho=0.9)
        spread_low = coeff_low.max() - coeff_low.min()
        spread_high = coeff_high.max() - coeff_high.min()
        assert spread_low > spread_high


# ---------------------------------------------------------------------------
# fuzzy_grey_scores
# ---------------------------------------------------------------------------

class TestFuzzyGreyScores:
    def _make_matrix(self):
        """Simple 4-candidate × 3-indicator matrix."""
        return np.array([
            [0.9, 0.8, 10.0],   # strong candidate
            [0.3, 0.2, 1.0],    # weak candidate
            [0.6, 0.5, 5.0],    # mid candidate
            [0.1, 0.1, 0.0],    # very weak
        ], dtype=float)

    def test_output_shape(self):
        matrix = self._make_matrix()
        weights = np.array([0.5, 0.3, 0.2])
        scores = fuzzy_grey_scores(matrix, weights)
        assert scores.shape == (4,)

    def test_strong_candidate_highest(self):
        matrix = self._make_matrix()
        weights = np.ones(3)
        scores = fuzzy_grey_scores(matrix, weights)
        assert scores.argmax() == 0, "Strongest candidate should have the highest score"

    def test_weak_candidate_lowest(self):
        matrix = self._make_matrix()
        weights = np.ones(3)
        scores = fuzzy_grey_scores(matrix, weights)
        assert scores.argmin() == 3, "Weakest candidate should have the lowest score"

    def test_scores_in_01(self):
        matrix = self._make_matrix()
        weights = np.array([1.0, 1.0, 1.0])
        scores = fuzzy_grey_scores(matrix, weights)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_equal_candidates_equal_scores(self):
        matrix = np.ones((5, 3))
        weights = np.array([0.4, 0.3, 0.3])
        scores = fuzzy_grey_scores(matrix, weights)
        assert np.allclose(scores, scores[0]), "Identical candidates must receive the same score"

    def test_fuzzy_weight_zero_is_pure_grey(self):
        """With fuzzy_weight=0 only the grey relational grade matters."""
        matrix = self._make_matrix()
        weights = np.ones(3)
        scores_fg = fuzzy_grey_scores(matrix, weights, fuzzy_weight=0.0)
        # Scores should still order correctly
        assert scores_fg.argmax() == 0

    def test_fuzzy_weight_one_is_pure_fuzzy(self):
        """With fuzzy_weight=1 only the fuzzy weighted average matters."""
        matrix = self._make_matrix()
        weights = np.ones(3)
        scores_ff = fuzzy_grey_scores(matrix, weights, fuzzy_weight=1.0)
        assert scores_ff.argmax() == 0

    def test_unnormalised_weights(self):
        """Weights are normalised internally; results should not depend on scale."""
        matrix = self._make_matrix()
        w1 = np.array([1.0, 2.0, 3.0])
        w2 = w1 * 100
        s1 = fuzzy_grey_scores(matrix, w1)
        s2 = fuzzy_grey_scores(matrix, w2)
        np.testing.assert_allclose(s1, s2, atol=1e-10)

    # --- validation errors ---
    def test_raises_on_negative_weights(self):
        matrix = np.ones((3, 2))
        with pytest.raises(ValueError, match="non-negative"):
            fuzzy_grey_scores(matrix, np.array([-1.0, 1.0]))

    def test_raises_on_fuzzy_weight_out_of_range(self):
        matrix = np.ones((3, 2))
        with pytest.raises(ValueError, match="fuzzy_weight"):
            fuzzy_grey_scores(matrix, np.ones(2), fuzzy_weight=1.5)

    def test_raises_on_negative_fuzzy_weight(self):
        matrix = np.ones((3, 2))
        with pytest.raises(ValueError, match="fuzzy_weight"):
            fuzzy_grey_scores(matrix, np.ones(2), fuzzy_weight=-0.1)

    def test_raises_on_all_zero_weights(self):
        matrix = np.ones((3, 2))
        with pytest.raises(ValueError, match="zero"):
            fuzzy_grey_scores(matrix, np.zeros(2))

    def test_raises_on_weight_length_mismatch(self):
        matrix = np.ones((3, 4))
        with pytest.raises(ValueError):
            fuzzy_grey_scores(matrix, np.ones(3))

    def test_raises_on_1d_matrix(self):
        with pytest.raises(ValueError):
            fuzzy_grey_scores(np.ones(5), np.ones(5))

    def test_single_candidate(self):
        """A single candidate should receive a well-defined score without error."""
        matrix = np.array([[0.8, 0.5, 3.0]])
        weights = np.array([0.5, 0.3, 0.2])
        scores = fuzzy_grey_scores(matrix, weights)
        assert scores.shape == (1,)
        assert 0.0 <= float(scores[0]) <= 1.0
