"""Improved Fuzzy-Grey Comprehensive Evaluation (改进模糊灰色综合评价方法).

This module implements a combined Fuzzy Membership + Grey Relational Analysis
scoring approach that replaces the simple linear weighted sum used previously.

Algorithm outline
-----------------
Given an indicator matrix X of shape (n_candidates, n_indicators):

1. **Fuzzy normalisation** – each indicator column is mapped to [0, 1] via a
   linear benefit-type membership function so that cross-candidate variation is
   preserved.

2. **Grey Relational Coefficients** – for each candidate the normalised indicator
   values are compared against the ideal reference sequence (all ones) to produce
   a grey relational coefficient matrix ξ using the standard GRA formula::

       ξ_ij = (Δ_min + ρ·Δ_max) / (Δ_ij + ρ·Δ_max)

   where ρ ∈ (0, 1] is the distinguishing coefficient (default 0.5) and
   Δ_ij = |1 − x̃_ij| is the absolute distance from the ideal.

3. **Grey Relational Grade** – weighted average of coefficients::

       r_i = Σ_j w_j · ξ_ij

4. **Fuzzy Comprehensive Score** – weighted average of normalised membership
   values::

       f_i = Σ_j w_j · x̃_ij

5. **Combined Score** – convex combination controlled by *fuzzy_weight*::

       score_i = α · f_i + (1 − α) · r_i

References
----------
* Deng, J. (1982). Control Problems of Grey Systems.
* Zadeh, L.A. (1965). Fuzzy sets.
"""

from __future__ import annotations

import numpy as np


def _fuzzy_membership(values: np.ndarray) -> np.ndarray:
    """Linear benefit-type fuzzy membership: maps *values* ∈ ℝ^n to [0, 1].

    If all values are identical the membership is 1.0 for every element so the
    indicator is effectively neutral (does not distinguish candidates).

    Parameters
    ----------
    values:
        1-D array of raw indicator scores for a single indicator across all
        candidates.

    Returns
    -------
    np.ndarray
        Membership degrees in [0, 1].
    """
    v_min = float(values.min())
    v_max = float(values.max())
    if v_max == v_min:
        return np.ones_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)


def _grey_relational_coefficients(
    normalised_matrix: np.ndarray,
    rho: float = 0.5,
) -> np.ndarray:
    """Compute grey relational coefficients against the ideal reference (all ones).

    Parameters
    ----------
    normalised_matrix:
        2-D array of shape ``(n_candidates, n_indicators)`` with values in
        [0, 1] produced by :func:`_fuzzy_membership`.
    rho:
        Distinguishing coefficient ρ ∈ (0, 1].  A smaller value increases the
        spread between high- and low-performing candidates (default 0.5).

    Returns
    -------
    np.ndarray
        Coefficient matrix ξ of shape ``(n_candidates, n_indicators)`` with
        values in (0, 1].
    """
    # Ideal reference: the best possible normalised value for every indicator
    reference = np.ones(normalised_matrix.shape[1], dtype=float)
    delta = np.abs(reference - normalised_matrix)  # (n, m)
    delta_min = float(delta.min())
    delta_max = float(delta.max())
    if delta_max == 0.0:
        return np.ones_like(delta, dtype=float)
    return (delta_min + rho * delta_max) / (delta + rho * delta_max)


def fuzzy_grey_scores(
    indicator_matrix: np.ndarray,
    weights: np.ndarray,
    rho: float = 0.5,
    fuzzy_weight: float = 0.4,
) -> np.ndarray:
    """Compute Fuzzy-Grey comprehensive scores for a set of candidates.

    Parameters
    ----------
    indicator_matrix:
        2-D array of shape ``(n_candidates, n_indicators)`` containing **raw**
        (unnormalised) scores.  All values must be non-negative benefit
        indicators (larger is better).
    weights:
        1-D array of shape ``(n_indicators,)`` with non-negative importance
        weights.  They are normalised internally so they need not sum to one.
    rho:
        Grey relational distinguishing coefficient ρ (default 0.5).
    fuzzy_weight:
        Blend coefficient α ∈ [0, 1] that controls the relative contribution of
        the fuzzy comprehensive score vs. the grey relational grade (default
        0.4).  Setting α = 0 gives a pure grey evaluation; α = 1 gives a pure
        fuzzy weighted average.

    Returns
    -------
    np.ndarray
        1-D array of combined scores, one per candidate, in the same order as
        *indicator_matrix*.

    Raises
    ------
    ValueError
        If *weights* do not match the number of indicators, or if any weight is
        negative, or if *fuzzy_weight* is outside [0, 1].
    """
    indicator_matrix = np.asarray(indicator_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if indicator_matrix.ndim != 2:
        raise ValueError(
            f"indicator_matrix must be 2-D, got shape {indicator_matrix.shape}"
        )
    n_candidates, n_indicators = indicator_matrix.shape
    if weights.ndim != 1 or weights.shape[0] != n_indicators:
        raise ValueError(
            f"weights must be 1-D with length {n_indicators}, got shape {weights.shape}"
        )
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative.")
    if weights.sum() == 0:
        raise ValueError("weights must not all be zero.")
    if not 0.0 <= fuzzy_weight <= 1.0:
        raise ValueError(f"fuzzy_weight must be in [0, 1], got {fuzzy_weight}")

    w = weights / weights.sum()

    # --- Step 1: Fuzzy normalisation ---
    fuzzy_matrix = np.apply_along_axis(_fuzzy_membership, 0, indicator_matrix)  # (n, m)

    # --- Step 2: Grey relational coefficients ---
    coeff = _grey_relational_coefficients(fuzzy_matrix, rho=rho)  # (n, m)

    # --- Step 3: Grey relational grade ---
    grey_grade = coeff @ w  # (n,)

    # --- Step 4: Fuzzy comprehensive score ---
    fuzzy_score = fuzzy_matrix @ w  # (n,)

    # --- Step 5: Combined score ---
    return fuzzy_weight * fuzzy_score + (1.0 - fuzzy_weight) * grey_grade


__all__ = ["fuzzy_grey_scores"]
