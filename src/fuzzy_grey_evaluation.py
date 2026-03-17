"""
改进模糊灰色综合储层评价方法
Improved Fuzzy Grey Comprehensive Reservoir Evaluation Method

This module implements a quantitative multi-indicator evaluation framework
that combines fuzzy mathematics and grey system theory to assess reservoir
quality from geological indicator data.

Algorithm overview
------------------
1. **Fuzzy normalisation** – piecewise-linear membership functions map raw
   indicator values onto evaluation grades (e.g. Class I → IV).
2. **Entropy weight method** – objective indicator weights are derived from
   the information entropy of each indicator's normalised data, optionally
   blended with expert-assigned weights.
3. **Grey relational analysis** – grey relational coefficients measure the
   proximity of each sample to an ideal reference sequence, yielding a grey
   relational degree per sample.
4. **Comprehensive score** – the fuzzy grade-membership vector (dot-producted
   with grade score values) and the grey relational degree are linearly
   combined into a single composite score, which determines the final grade
   assignment.

References
----------
- Deng, J.-L. (1989). Introduction to grey system. *Journal of Grey System*,
  1(1), 1–24.
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System
  Technical Journal*, 27(3), 379–423.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class IndicatorType(str, Enum):
    """Whether a larger indicator value represents better or worse quality."""

    BENEFIT = "benefit"  # Larger is better (e.g. porosity, permeability)
    COST = "cost"  # Smaller is better (e.g. clay content, skin factor)


@dataclass
class EvaluationGrade:
    """A single evaluation grade with a display name, boundary interval, and score.

    Parameters
    ----------
    name:
        Human-readable grade label, e.g. ``"Class I"`` or ``"优"``.
    boundaries:
        ``(lower, upper)`` threshold pair that defines the grade's plateau
        region.  Grades should be supplied in **ascending** order of
        ``lower`` (i.e. from worst to best for benefit-type indicators).
    value:
        Numeric score assigned to this grade (used in the final composite
        score calculation).  Higher values represent better quality.
    """

    name: str
    boundaries: Tuple[float, float]
    value: float


@dataclass
class IndicatorConfig:
    """Configuration for a single evaluation indicator.

    Parameters
    ----------
    name:
        Display name of the indicator (e.g. ``"Porosity (%)"``).
    indicator_type:
        :attr:`IndicatorType.BENEFIT` if a higher value is desirable,
        :attr:`IndicatorType.COST` if a lower value is desirable.
    weight:
        Optional expert-assigned weight in ``[0, 1]``.  When *all*
        indicators carry explicit weights they are normalised to sum to 1.
        When ``use_entropy_weights=True`` (default), expert weights are
        blended 50 / 50 with the entropy-derived weights.
    unit:
        Physical unit string for display purposes only.
    """

    name: str
    indicator_type: IndicatorType = IndicatorType.BENEFIT
    weight: Optional[float] = None
    unit: str = ""


@dataclass
class EvaluationResult:
    """Evaluation result for a single reservoir sample.

    Attributes
    ----------
    sample_id:
        Identifier of the sample.
    grade:
        Name of the assigned evaluation grade.
    score:
        Composite score (fuzzy + grey), higher is better.
    grade_memberships:
        Weighted fuzzy membership degree for each grade.
    grey_relational_degree:
        Weighted grey relational degree (proximity to ideal reference).
    indicator_values:
        Raw indicator values keyed by indicator name.
    """

    sample_id: str
    grade: str
    score: float
    grade_memberships: Dict[str, float]
    grey_relational_degree: float
    indicator_values: Dict[str, float]


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------


def _semi_trapezoidal_left(x: float, a: float, b: float) -> float:
    """Left-open trapezoidal: membership = 1 for x ≤ a, linearly falls to 0 at b.

    .. code-block::

        μ
        1 ──────┐
                ╲
        0        └─── x
                a   b
    """
    if b <= a:
        return 1.0 if x <= a else 0.0
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def _semi_trapezoidal_right(x: float, a: float, b: float) -> float:
    """Right-open trapezoidal: membership rises from 0 at a to 1 at b, stays 1.

    .. code-block::

        μ
        1          ┌──────
                  ╱
        0 ────────  x
                  a  b
    """
    if b <= a:
        return 0.0 if x < a else 1.0
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def _triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership with peak at *b* and zeros at *a* and *c*.

    Parameters
    ----------
    a, b, c:
        Left foot, apex, right foot of the triangle (must satisfy a ≤ b ≤ c).
    """
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def _grade_membership(x: float, grade_idx: int, centers: np.ndarray) -> float:
    """Compute the membership of *x* in grade *grade_idx* given all grade centers.

    Grades are ordered from worst (index 0) to best (index n-1).  The
    membership function is:

    * **Worst grade** – left semi-trapezoidal (full membership below the
      lowest center, falling to 0 at the next center).
    * **Best grade** – right semi-trapezoidal (rising from 0 at the
      second-to-last center to full membership at the last center).
    * **Middle grades** – triangular with apex at the grade center and feet
      at the two adjacent centers.

    Parameters
    ----------
    x:
        (Possibly adjusted) indicator value.
    grade_idx:
        Index of the target grade.
    centers:
        1-D array of grade centers in ascending order.
    """
    n = len(centers)
    c = centers[grade_idx]

    if n == 1:
        return 1.0

    if grade_idx == 0:
        return _semi_trapezoidal_left(x, c, centers[1])
    if grade_idx == n - 1:
        return _semi_trapezoidal_right(x, centers[n - 2], c)
    return _triangular(x, centers[grade_idx - 1], c, centers[grade_idx + 1])


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class FuzzyGreyEvaluator:
    """Improved Fuzzy Grey Comprehensive Reservoir Evaluator.

    The evaluator accepts a matrix of raw indicator values (one row per
    reservoir sample, one column per indicator) and returns a ranked list
    of :class:`EvaluationResult` objects.

    Parameters
    ----------
    indicators:
        Ordered list of :class:`IndicatorConfig` objects—one per column in
        the data matrix.
    grades:
        Ordered list of :class:`EvaluationGrade` objects, sorted from the
        **worst** grade to the **best** grade (ascending ``boundaries``).
    rho:
        Grey relational distinguishing coefficient ρ ∈ (0, 1].  Smaller
        values increase discrimination between samples; 0.5 is the
        conventional default.
    use_entropy_weights:
        When ``True`` (default) the entropy weight method is used to
        derive objective weights.  If expert weights are also supplied they
        are blended with the entropy weights (equal blend).  When ``False``
        only expert weights are used (uniform if none supplied).
    fuzzy_weight:
        Proportion (0–1) of the fuzzy score in the final composite score.
        The grey component receives ``1 - fuzzy_weight``.  Default is 0.6.
    """

    def __init__(
        self,
        indicators: List[IndicatorConfig],
        grades: List[EvaluationGrade],
        rho: float = 0.5,
        use_entropy_weights: bool = True,
        fuzzy_weight: float = 0.6,
    ) -> None:
        if not indicators:
            raise ValueError("At least one indicator must be provided.")
        if not grades:
            raise ValueError("At least one grade must be provided.")
        if not 0 < rho <= 1:
            raise ValueError(f"rho must be in (0, 1]; got {rho!r}.")
        if not 0.0 <= fuzzy_weight <= 1.0:
            raise ValueError(f"fuzzy_weight must be in [0, 1]; got {fuzzy_weight!r}.")

        self.indicators = indicators
        self.grades = grades
        self.rho = rho
        self.use_entropy_weights = use_entropy_weights
        self.fuzzy_weight = fuzzy_weight

        # Pre-compute grade centers (ascending order, worst → best)
        self._grade_centers: np.ndarray = np.array(
            [(g.boundaries[0] + g.boundaries[1]) / 2.0 for g in grades]
        )
        self._grade_values: np.ndarray = np.array([g.value for g in grades])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adjusted_value(self, x: float, ind: IndicatorConfig, col_min: float, col_max: float) -> float:
        """Return *x* mirrored for cost-type indicators so that higher always means better."""
        if ind.indicator_type == IndicatorType.COST:
            return col_max + col_min - x
        return x

    def _membership_matrix(self, col: np.ndarray, ind: IndicatorConfig) -> np.ndarray:
        """Compute fuzzy membership matrix for one indicator column.

        Parameters
        ----------
        col:
            1-D array of raw values for one indicator, shape ``(n_samples,)``.
        ind:
            The indicator configuration.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_grades)``.  Each row sums to ≥ 0; the
            maximum entry per row identifies the most likely grade.
        """
        col_min = float(col.min())
        col_max = float(col.max())
        n_samples = len(col)
        n_grades = len(self.grades)
        R = np.zeros((n_samples, n_grades))

        for s, x in enumerate(col):
            # For cost-type indicators, mirror x so that "better" values map
            # to the higher end of the grade-centre scale, making the
            # benefit-oriented membership function directly applicable.
            x_eff = self._adjusted_value(float(x), ind, col_min, col_max)
            for g in range(n_grades):
                R[s, g] = _grade_membership(x_eff, g, self._grade_centers)
        return R

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalise each column to [0, 1], respecting indicator type.

        For cost-type indicators the direction is flipped so that higher
        normalised values always correspond to better quality.
        """
        normed = np.zeros_like(data, dtype=float)
        for j, ind in enumerate(self.indicators):
            col = data[:, j].astype(float)
            col_min, col_max = col.min(), col.max()
            span = col_max - col_min
            if span == 0.0:
                normed[:, j] = 0.5
                continue
            if ind.indicator_type == IndicatorType.BENEFIT:
                normed[:, j] = (col - col_min) / span
            else:
                normed[:, j] = (col_max - col) / span
        return normed

    def _entropy_weights(self, normed: np.ndarray) -> np.ndarray:
        """Compute indicator weights by the entropy weight method.

        Parameters
        ----------
        normed:
            Normalised data matrix ``(n_samples, n_indicators)`` with values
            in [0, 1].

        Returns
        -------
        np.ndarray
            Weight vector of shape ``(n_indicators,)``, summing to 1.
        """
        n_samples, n_indicators = normed.shape
        eps = 1e-12

        # Proportion matrix
        col_sums = normed.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0.0, eps, col_sums)
        P = normed / col_sums
        P = np.clip(P, eps, 1.0)

        # Entropy per indicator
        k = 1.0 / np.log(max(n_samples, 2))
        H = -k * np.sum(P * np.log(P), axis=0)
        H = np.clip(H, 0.0, 1.0)

        redundancy = 1.0 - H
        total = redundancy.sum()
        if total < eps:
            return np.full(n_indicators, 1.0 / n_indicators)
        return redundancy / total

    def _compute_weights(self, normed: np.ndarray) -> np.ndarray:
        """Return the final indicator weight vector.

        Combines entropy-derived weights and expert weights according to the
        ``use_entropy_weights`` flag and the expert weights embedded in the
        indicator configs.
        """
        n = len(self.indicators)

        # Gather expert weights (NaN where absent)
        expert = np.array(
            [ind.weight if ind.weight is not None else np.nan for ind in self.indicators],
            dtype=float,
        )

        if self.use_entropy_weights:
            ew = self._entropy_weights(normed)
            has_expert = ~np.isnan(expert)
            if has_expert.any():
                # Fill missing expert weights with entropy weights, then blend
                expert_filled = np.where(has_expert, expert, ew)
                expert_total = expert_filled.sum()
                if expert_total > 0:
                    expert_filled /= expert_total
                weights = 0.5 * ew + 0.5 * expert_filled
            else:
                weights = ew
        else:
            has_expert = ~np.isnan(expert)
            if has_expert.any():
                expert_filled = np.where(has_expert, expert, 1.0 / n)
                total = expert_filled.sum()
                weights = expert_filled / total if total > 0 else np.full(n, 1.0 / n)
            else:
                weights = np.full(n, 1.0 / n)

        # Normalise to ensure sum = 1
        total = weights.sum()
        return weights / total if total > 0 else np.full(n, 1.0 / n)

    def _grey_relational_coefficients(self, normed: np.ndarray) -> np.ndarray:
        """Compute grey relational coefficients against the ideal reference.

        The ideal reference sequence is the all-ones vector (maximum
        normalised value for every indicator, i.e. the best possible sample).

        Parameters
        ----------
        normed:
            Normalised data ``(n_samples, n_indicators)``.

        Returns
        -------
        np.ndarray
            Coefficient matrix ``(n_samples, n_indicators)`` with values
            in ``(0, 1]``.
        """
        delta = np.abs(normed - 1.0)  # deviation from ideal
        delta_min = delta.min()
        delta_max = delta.max()

        denom = delta + self.rho * delta_max
        denom = np.where(denom == 0.0, 1e-12, denom)
        xi = (delta_min + self.rho * delta_max) / denom
        return xi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        data: np.ndarray,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate reservoir quality for a batch of samples.

        Parameters
        ----------
        data:
            Raw indicator matrix of shape ``(n_samples, n_indicators)``.
            Column order must match :attr:`indicators`.
        sample_ids:
            Optional identifiers for each sample row.  Defaults to
            ``["S1", "S2", ...]``.

        Returns
        -------
        List[EvaluationResult]
            Results sorted by composite score in descending order (best
            reservoir quality first).

        Raises
        ------
        ValueError
            If the number of columns in *data* does not match the number
            of configured indicators.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"data must be a 2-D matrix; got shape {data.shape}.")
        n_samples, n_cols = data.shape
        if n_cols != len(self.indicators):
            raise ValueError(
                f"data has {n_cols} columns but {len(self.indicators)} indicator(s) are configured."
            )

        if sample_ids is None:
            ids: List[str] = [f"S{i + 1}" for i in range(n_samples)]
        else:
            ids = list(sample_ids)
            if len(ids) != n_samples:
                raise ValueError(
                    f"sample_ids has length {len(ids)} but data has {n_samples} rows."
                )

        # ── Step 1: Normalise ────────────────────────────────────────────────
        normed = self._normalize(data)

        # ── Step 2: Weights ──────────────────────────────────────────────────
        weights = self._compute_weights(normed)
        logger.debug(
            "Indicator weights: %s",
            {ind.name: round(float(w), 4) for ind, w in zip(self.indicators, weights)},
        )

        # ── Step 3: Fuzzy membership (weighted combination over indicators) ──
        R = np.zeros((n_samples, len(self.grades)))
        for j, ind in enumerate(self.indicators):
            R_j = self._membership_matrix(data[:, j], ind)
            R += weights[j] * R_j

        # ── Step 4: Fuzzy score (membership × grade values) ──────────────────
        fuzzy_scores = R.dot(self._grade_values)  # (n_samples,)

        # ── Step 5: Grey relational degree ───────────────────────────────────
        xi = self._grey_relational_coefficients(normed)  # (n_samples, n_indicators)
        grey_degree = xi.dot(weights)  # (n_samples,)

        # Scale grey degree to the grade value range for a fair blend
        v_min = self._grade_values.min()
        v_max = self._grade_values.max()
        g_min = grey_degree.min()
        g_max = grey_degree.max()
        if g_max > g_min:
            grey_scaled = (grey_degree - g_min) / (g_max - g_min) * (v_max - v_min) + v_min
        else:
            grey_scaled = np.full(n_samples, (v_min + v_max) / 2.0)

        # ── Step 6: Composite score ──────────────────────────────────────────
        fw = self.fuzzy_weight
        final_scores = fw * fuzzy_scores + (1.0 - fw) * grey_scaled

        # ── Step 7: Grade assignment (maximum fuzzy membership) ──────────────
        grade_indices = np.argmax(R, axis=1)

        # ── Assemble results ─────────────────────────────────────────────────
        results: List[EvaluationResult] = []
        for i in range(n_samples):
            memberships = {g.name: float(R[i, k]) for k, g in enumerate(self.grades)}
            ind_vals = {ind.name: float(data[i, j]) for j, ind in enumerate(self.indicators)}
            results.append(
                EvaluationResult(
                    sample_id=ids[i],
                    grade=self.grades[int(grade_indices[i])].name,
                    score=float(final_scores[i]),
                    grade_memberships=memberships,
                    grey_relational_degree=float(grey_degree[i]),
                    indicator_values=ind_vals,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def get_weights(self, data: np.ndarray) -> Dict[str, float]:
        """Compute and return indicator weights for *data* without full evaluation.

        Parameters
        ----------
        data:
            Raw indicator matrix ``(n_samples, n_indicators)``.

        Returns
        -------
        Dict[str, float]
            Mapping from indicator name to weight value (sum = 1).
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"data must be a 2-D matrix; got shape {data.shape}.")
        normed = self._normalize(data)
        weights = self._compute_weights(normed)
        return {ind.name: float(w) for ind, w in zip(self.indicators, weights)}


# ---------------------------------------------------------------------------
# Convenience factory – standard four-class reservoir evaluation
# ---------------------------------------------------------------------------


def build_reservoir_evaluator(
    rho: float = 0.5,
    use_entropy_weights: bool = True,
) -> FuzzyGreyEvaluator:
    """Return a pre-configured evaluator for standard four-class reservoir grading.

    The default indicator set covers the four most common petrophysical
    parameters used in Chinese reservoir evaluation standards:

    * **Porosity (%)** – benefit type
    * **Permeability (mD)** – benefit type
    * **Oil saturation (%)** – benefit type
    * **Shale content (%)** – cost type

    Grade scale (Class I = best):

    =========  ============  ========
    Grade      Porosity (%)  Score
    =========  ============  ========
    Class I    ≥ 25          100
    Class II   15 – 25       75
    Class III  8 – 15        50
    Class IV   0 – 8         25
    =========  ============  ========

    Parameters
    ----------
    rho:
        Distinguishing coefficient for grey relational analysis.
    use_entropy_weights:
        Whether to use entropy-derived weights.

    Returns
    -------
    FuzzyGreyEvaluator
        Ready-to-use evaluator instance.
    """
    indicators = [
        IndicatorConfig("Porosity (%)", IndicatorType.BENEFIT),
        IndicatorConfig("Permeability (mD)", IndicatorType.BENEFIT),
        IndicatorConfig("Oil Saturation (%)", IndicatorType.BENEFIT),
        IndicatorConfig("Shale Content (%)", IndicatorType.COST),
    ]

    # Grades ordered worst → best (ascending boundaries)
    grades = [
        EvaluationGrade("Class IV", (0.0, 8.0), 25.0),
        EvaluationGrade("Class III", (8.0, 15.0), 50.0),
        EvaluationGrade("Class II", (15.0, 25.0), 75.0),
        EvaluationGrade("Class I", (25.0, 100.0), 100.0),
    ]

    return FuzzyGreyEvaluator(
        indicators=indicators,
        grades=grades,
        rho=rho,
        use_entropy_weights=use_entropy_weights,
    )


__all__ = [
    "IndicatorType",
    "EvaluationGrade",
    "IndicatorConfig",
    "EvaluationResult",
    "FuzzyGreyEvaluator",
    "build_reservoir_evaluator",
]
