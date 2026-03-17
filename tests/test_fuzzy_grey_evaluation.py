"""Tests for src/fuzzy_grey_evaluation.py.

Covers:
- Membership functions (_semi_trapezoidal_left, _semi_trapezoidal_right,
  _triangular, _grade_membership)
- FuzzyGreyEvaluator normalisation, weight computation, grey relational
  analysis, and end-to-end evaluation
- Convenience factory build_reservoir_evaluator
- Edge cases: single sample, single grade, uniform data, cost-type indicators
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.fuzzy_grey_evaluation import (
    EvaluationGrade,
    FuzzyGreyEvaluator,
    IndicatorConfig,
    IndicatorType,
    _grade_membership,
    _semi_trapezoidal_left,
    _semi_trapezoidal_right,
    _triangular,
    build_reservoir_evaluator,
)


# ---------------------------------------------------------------------------
# Membership function unit tests
# ---------------------------------------------------------------------------


class TestSemiTrapezoidalLeft:
    def test_below_a_returns_one(self):
        assert _semi_trapezoidal_left(0.0, 5.0, 10.0) == pytest.approx(1.0)

    def test_at_a_returns_one(self):
        assert _semi_trapezoidal_left(5.0, 5.0, 10.0) == pytest.approx(1.0)

    def test_at_b_returns_zero(self):
        assert _semi_trapezoidal_left(10.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_above_b_returns_zero(self):
        assert _semi_trapezoidal_left(15.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_midpoint_returns_half(self):
        assert _semi_trapezoidal_left(7.5, 5.0, 10.0) == pytest.approx(0.5)

    def test_degenerate_a_equals_b(self):
        # When a == b the function should still return a valid value
        result = _semi_trapezoidal_left(5.0, 5.0, 5.0)
        assert 0.0 <= result <= 1.0


class TestSemiTrapezoidalRight:
    def test_below_a_returns_zero(self):
        assert _semi_trapezoidal_right(0.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_at_a_returns_zero(self):
        assert _semi_trapezoidal_right(5.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_at_b_returns_one(self):
        assert _semi_trapezoidal_right(10.0, 5.0, 10.0) == pytest.approx(1.0)

    def test_above_b_returns_one(self):
        assert _semi_trapezoidal_right(15.0, 5.0, 10.0) == pytest.approx(1.0)

    def test_midpoint_returns_half(self):
        assert _semi_trapezoidal_right(7.5, 5.0, 10.0) == pytest.approx(0.5)

    def test_degenerate_a_equals_b(self):
        result = _semi_trapezoidal_right(5.0, 5.0, 5.0)
        assert 0.0 <= result <= 1.0


class TestTriangular:
    def test_at_apex_returns_one(self):
        assert _triangular(5.0, 0.0, 5.0, 10.0) == pytest.approx(1.0)

    def test_at_left_foot_returns_zero(self):
        assert _triangular(0.0, 0.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_at_right_foot_returns_zero(self):
        assert _triangular(10.0, 0.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_outside_left_returns_zero(self):
        assert _triangular(-1.0, 0.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_outside_right_returns_zero(self):
        assert _triangular(11.0, 0.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_left_slope_midpoint(self):
        assert _triangular(2.5, 0.0, 5.0, 10.0) == pytest.approx(0.5)

    def test_right_slope_midpoint(self):
        assert _triangular(7.5, 0.0, 5.0, 10.0) == pytest.approx(0.5)


class TestGradeMembership:
    """Test _grade_membership with a simple 3-grade setup."""

    def setup_method(self):
        # Three grades with centers at 5, 15, 25
        self.centers = np.array([5.0, 15.0, 25.0])

    def test_worst_grade_below_center_is_one(self):
        assert _grade_membership(0.0, 0, self.centers) == pytest.approx(1.0)

    def test_best_grade_above_center_is_one(self):
        assert _grade_membership(30.0, 2, self.centers) == pytest.approx(1.0)

    def test_middle_grade_at_apex(self):
        assert _grade_membership(15.0, 1, self.centers) == pytest.approx(1.0)

    def test_middle_grade_at_adjacent_center_is_zero(self):
        assert _grade_membership(5.0, 1, self.centers) == pytest.approx(0.0)
        assert _grade_membership(25.0, 1, self.centers) == pytest.approx(0.0)

    def test_single_grade(self):
        centers = np.array([10.0])
        assert _grade_membership(0.0, 0, centers) == pytest.approx(1.0)
        assert _grade_membership(99.0, 0, centers) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FuzzyGreyEvaluator – construction validation
# ---------------------------------------------------------------------------


class TestFuzzyGreyEvaluatorValidation:
    def _make_evaluator(self, **kwargs):
        indicators = [IndicatorConfig("A", IndicatorType.BENEFIT)]
        grades = [
            EvaluationGrade("Low", (0.0, 10.0), 25.0),
            EvaluationGrade("High", (10.0, 20.0), 75.0),
        ]
        return FuzzyGreyEvaluator(indicators=indicators, grades=grades, **kwargs)

    def test_empty_indicators_raises(self):
        with pytest.raises(ValueError, match="indicator"):
            FuzzyGreyEvaluator(
                indicators=[],
                grades=[EvaluationGrade("X", (0.0, 1.0), 1.0)],
            )

    def test_empty_grades_raises(self):
        with pytest.raises(ValueError, match="grade"):
            FuzzyGreyEvaluator(
                indicators=[IndicatorConfig("A")],
                grades=[],
            )

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            self._make_evaluator(rho=0.0)
        with pytest.raises(ValueError, match="rho"):
            self._make_evaluator(rho=1.5)

    def test_invalid_fuzzy_weight_raises(self):
        with pytest.raises(ValueError, match="fuzzy_weight"):
            self._make_evaluator(fuzzy_weight=-0.1)
        with pytest.raises(ValueError, match="fuzzy_weight"):
            self._make_evaluator(fuzzy_weight=1.1)

    def test_data_dimension_mismatch_raises(self):
        ev = self._make_evaluator()
        with pytest.raises(ValueError, match="column"):
            ev.evaluate(np.array([[1.0, 2.0]]))  # 2 cols, 1 indicator

    def test_sample_ids_length_mismatch_raises(self):
        ev = self._make_evaluator()
        with pytest.raises(ValueError, match="sample_ids"):
            ev.evaluate(np.array([[5.0]]), sample_ids=["A", "B"])

    def test_1d_data_raises(self):
        ev = self._make_evaluator()
        with pytest.raises(ValueError, match="2-D"):
            ev.evaluate(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# FuzzyGreyEvaluator – normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:
    def _evaluator(self, ind_type=IndicatorType.BENEFIT):
        indicators = [IndicatorConfig("X", ind_type)]
        grades = [
            EvaluationGrade("Low", (0.0, 5.0), 25.0),
            EvaluationGrade("High", (5.0, 10.0), 75.0),
        ]
        return FuzzyGreyEvaluator(indicators=indicators, grades=grades)

    def test_benefit_min_maps_to_zero(self):
        ev = self._evaluator(IndicatorType.BENEFIT)
        normed = ev._normalize(np.array([[0.0], [10.0]]))
        assert normed[0, 0] == pytest.approx(0.0)

    def test_benefit_max_maps_to_one(self):
        ev = self._evaluator(IndicatorType.BENEFIT)
        normed = ev._normalize(np.array([[0.0], [10.0]]))
        assert normed[1, 0] == pytest.approx(1.0)

    def test_cost_min_maps_to_one(self):
        ev = self._evaluator(IndicatorType.COST)
        normed = ev._normalize(np.array([[0.0], [10.0]]))
        assert normed[0, 0] == pytest.approx(1.0)

    def test_cost_max_maps_to_zero(self):
        ev = self._evaluator(IndicatorType.COST)
        normed = ev._normalize(np.array([[0.0], [10.0]]))
        assert normed[1, 0] == pytest.approx(0.0)

    def test_uniform_column_maps_to_half(self):
        ev = self._evaluator()
        normed = ev._normalize(np.array([[5.0], [5.0], [5.0]]))
        assert np.allclose(normed[:, 0], 0.5)


# ---------------------------------------------------------------------------
# FuzzyGreyEvaluator – entropy weights
# ---------------------------------------------------------------------------


class TestEntropyWeights:
    def _evaluator(self):
        indicators = [
            IndicatorConfig("A", IndicatorType.BENEFIT),
            IndicatorConfig("B", IndicatorType.BENEFIT),
        ]
        grades = [
            EvaluationGrade("Low", (0.0, 5.0), 25.0),
            EvaluationGrade("High", (5.0, 10.0), 75.0),
        ]
        return FuzzyGreyEvaluator(indicators=indicators, grades=grades, use_entropy_weights=True)

    def test_weights_sum_to_one(self):
        ev = self._evaluator()
        data = np.array([[1.0, 9.0], [5.0, 5.0], [9.0, 1.0]])
        weights = ev.get_weights(data)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_weights_are_non_negative(self):
        ev = self._evaluator()
        data = np.random.default_rng(42).random((10, 2)) * 10
        weights = ev.get_weights(data)
        for w in weights.values():
            assert w >= 0.0

    def test_more_discriminating_indicator_gets_higher_weight(self):
        """Indicator B has constant values → lower entropy weight than A."""
        ev = self._evaluator()
        # A varies, B is constant
        data = np.column_stack([
            np.arange(1, 11, dtype=float),   # A: values 1 to 10
            np.full(10, 5.0),               # B: constant
        ])
        weights = ev.get_weights(data)
        assert weights["A"] > weights["B"]

    def test_uniform_columns_give_equal_weights(self):
        """When both indicators have the same distribution, weights should be equal."""
        ev = self._evaluator()
        # Identical columns → identical entropy → equal weights
        col = np.arange(1, 11, dtype=float)
        data = np.column_stack([col, col])
        weights = ev.get_weights(data)
        assert weights["A"] == pytest.approx(weights["B"], abs=1e-9)


# ---------------------------------------------------------------------------
# FuzzyGreyEvaluator – grey relational coefficients
# ---------------------------------------------------------------------------


class TestGreyRelational:
    def _evaluator(self):
        indicators = [IndicatorConfig("A", IndicatorType.BENEFIT)]
        grades = [
            EvaluationGrade("Low", (0.0, 5.0), 25.0),
            EvaluationGrade("High", (5.0, 10.0), 75.0),
        ]
        return FuzzyGreyEvaluator(indicators=indicators, grades=grades)

    def test_ideal_sample_has_highest_grey_degree(self):
        ev = self._evaluator()
        # The ideal normalised value is 1.0 (best possible)
        normed = np.array([[1.0], [0.5], [0.0]])
        xi = ev._grey_relational_coefficients(normed)
        # First row (ideal) should have the largest coefficient
        assert xi[0, 0] >= xi[1, 0]
        assert xi[0, 0] >= xi[2, 0]

    def test_coefficients_in_0_1(self):
        ev = self._evaluator()
        rng = np.random.default_rng(0)
        normed = rng.random((20, 1))
        xi = ev._grey_relational_coefficients(normed)
        assert (xi >= 0.0).all()
        assert (xi <= 1.0).all()


# ---------------------------------------------------------------------------
# FuzzyGreyEvaluator – end-to-end evaluate()
# ---------------------------------------------------------------------------


class TestEvaluateEndToEnd:
    """Integration tests using a simple two-indicator, three-grade setup."""

    def _build_evaluator(self):
        indicators = [
            IndicatorConfig("Porosity", IndicatorType.BENEFIT),
            IndicatorConfig("Clay", IndicatorType.COST),
        ]
        grades = [
            EvaluationGrade("Poor", (0.0, 8.0), 25.0),
            EvaluationGrade("Medium", (8.0, 15.0), 50.0),
            EvaluationGrade("Good", (15.0, 100.0), 100.0),
        ]
        return FuzzyGreyEvaluator(indicators=indicators, grades=grades)

    def test_returns_one_result_per_sample(self):
        ev = self._build_evaluator()
        data = np.array([[20.0, 5.0], [5.0, 30.0]])
        results = ev.evaluate(data)
        assert len(results) == 2

    def test_custom_sample_ids_preserved(self):
        ev = self._build_evaluator()
        data = np.array([[20.0, 5.0]])
        results = ev.evaluate(data, sample_ids=["Well-A"])
        assert results[0].sample_id == "Well-A"

    def test_best_sample_scores_highest(self):
        """High porosity + low clay → should beat low porosity + high clay."""
        ev = self._build_evaluator()
        data = np.array([
            [30.0, 5.0],   # Best reservoir
            [3.0, 40.0],   # Worst reservoir
        ])
        results = ev.evaluate(data, sample_ids=["Best", "Worst"])
        scores = {r.sample_id: r.score for r in results}
        assert scores["Best"] > scores["Worst"]

    def test_grade_assignments_logical(self):
        """Very high porosity sample should land in 'Good', very low in 'Poor'."""
        ev = self._build_evaluator()
        data = np.array([
            [50.0, 2.0],   # Clearly good
            [1.0, 50.0],   # Clearly poor
        ])
        results = ev.evaluate(data, sample_ids=["Good", "Poor"])
        grade_map = {r.sample_id: r.grade for r in results}
        assert grade_map["Good"] == "Good"
        assert grade_map["Poor"] == "Poor"

    def test_membership_values_in_0_1(self):
        ev = self._build_evaluator()
        data = np.random.default_rng(7).random((15, 2)) * 30
        for res in ev.evaluate(data):
            for val in res.grade_memberships.values():
                assert 0.0 <= val <= 1.0 + 1e-9

    def test_grey_degree_positive(self):
        ev = self._build_evaluator()
        data = np.random.default_rng(8).random((8, 2)) * 20
        for res in ev.evaluate(data):
            assert res.grey_relational_degree > 0.0

    def test_results_sorted_descending(self):
        ev = self._build_evaluator()
        rng = np.random.default_rng(99)
        data = rng.random((20, 2)) * 30
        results = ev.evaluate(data)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_single_sample(self):
        ev = self._build_evaluator()
        results = ev.evaluate(np.array([[20.0, 10.0]]))
        assert len(results) == 1
        assert isinstance(results[0].grade, str)

    def test_indicator_values_stored_correctly(self):
        ev = self._build_evaluator()
        data = np.array([[17.5, 12.3]])
        results = ev.evaluate(data)
        assert results[0].indicator_values["Porosity"] == pytest.approx(17.5)
        assert results[0].indicator_values["Clay"] == pytest.approx(12.3)


# ---------------------------------------------------------------------------
# FuzzyGreyEvaluator – expert weights interaction
# ---------------------------------------------------------------------------


class TestExpertWeights:
    def _build_evaluator(self, use_entropy: bool):
        indicators = [
            IndicatorConfig("A", IndicatorType.BENEFIT, weight=0.7),
            IndicatorConfig("B", IndicatorType.BENEFIT, weight=0.3),
        ]
        grades = [
            EvaluationGrade("Low", (0.0, 5.0), 25.0),
            EvaluationGrade("High", (5.0, 10.0), 75.0),
        ]
        return FuzzyGreyEvaluator(indicators=indicators, grades=grades, use_entropy_weights=use_entropy)

    def test_expert_only_weights_sum_to_one(self):
        ev = self._build_evaluator(use_entropy=False)
        data = np.array([[1.0, 9.0], [5.0, 5.0], [9.0, 1.0]])
        w = ev.get_weights(data)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_blended_weights_sum_to_one(self):
        ev = self._build_evaluator(use_entropy=True)
        data = np.array([[1.0, 9.0], [5.0, 5.0], [9.0, 1.0]])
        w = ev.get_weights(data)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# FuzzyGreyEvaluator – single-grade edge case
# ---------------------------------------------------------------------------


class TestSingleGrade:
    def test_single_grade_evaluates_without_error(self):
        indicators = [IndicatorConfig("X", IndicatorType.BENEFIT)]
        grades = [EvaluationGrade("Only", (0.0, 10.0), 50.0)]
        ev = FuzzyGreyEvaluator(indicators=indicators, grades=grades)
        results = ev.evaluate(np.array([[3.0], [7.0]]))
        assert len(results) == 2
        for r in results:
            assert r.grade == "Only"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


class TestBuildReservoirEvaluator:
    def test_returns_fuzzy_grey_evaluator(self):
        ev = build_reservoir_evaluator()
        assert isinstance(ev, FuzzyGreyEvaluator)

    def test_four_indicators(self):
        ev = build_reservoir_evaluator()
        assert len(ev.indicators) == 4

    def test_four_grades(self):
        ev = build_reservoir_evaluator()
        assert len(ev.grades) == 4

    def test_grade_names(self):
        ev = build_reservoir_evaluator()
        names = {g.name for g in ev.grades}
        assert names == {"Class I", "Class II", "Class III", "Class IV"}

    def test_class_i_reservoir_identified(self):
        """A sample with excellent petrophysical properties should be graded Class I."""
        ev = build_reservoir_evaluator()
        # Excellent: high porosity, high permeability, high oil saturation, low shale
        data = np.array([[30.0, 500.0, 70.0, 5.0]])
        results = ev.evaluate(data)
        assert results[0].grade == "Class I"

    def test_class_iv_reservoir_identified(self):
        """A sample with poor petrophysical properties should be graded Class IV."""
        ev = build_reservoir_evaluator()
        # Poor: low porosity, low permeability, low oil saturation, high shale
        data = np.array([[2.0, 0.5, 20.0, 55.0]])
        results = ev.evaluate(data)
        assert results[0].grade == "Class IV"

    def test_custom_rho(self):
        ev = build_reservoir_evaluator(rho=0.3)
        assert ev.rho == pytest.approx(0.3)

    def test_evaluate_multiple_samples(self):
        ev = build_reservoir_evaluator()
        # Four samples spanning the quality range
        data = np.array([
            [30.0, 500.0, 70.0, 5.0],   # Class I
            [20.0, 50.0, 55.0, 15.0],   # Class II
            [11.0, 5.0, 40.0, 25.0],    # Class III
            [3.0, 0.5, 20.0, 45.0],     # Class IV
        ])
        results = ev.evaluate(data)
        assert len(results) == 4
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
