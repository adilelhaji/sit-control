"""Unit tests for the control-evaluation metrics.

Covers :mod:`sit_control.metrics`: the L^1 and L^2 release costs, the relative
error helper (including its zero-reference guard) and the suppression-time
detector (first crossing, tolerance handling and the never-reached case). The
inputs are chosen so the exact analytic value of each integral is known, so the
assertions pin the numbers rather than merely checking finiteness. Pure Python,
no GEKKO.
"""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.metrics import cost_L1, cost_L2, relative_error, suppression_time

# ---------------------------------------------------------------------------
# cost_L1
# ---------------------------------------------------------------------------


def test_cost_L1_constant_profile() -> None:
    """Integral of a constant u=2 over [0, 10] equals 20 (analytic)."""
    t = np.linspace(0.0, 10.0, 101)
    u = np.full_like(t, 2.0)
    assert cost_L1(t, u) == pytest.approx(20.0)


def test_cost_L1_zero_profile() -> None:
    """A zero release profile costs nothing."""
    t = np.linspace(0.0, 5.0, 11)
    u = np.zeros_like(t)
    assert cost_L1(t, u) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cost_L2
# ---------------------------------------------------------------------------


def test_cost_L2_constant_profile_default_weight() -> None:
    """Integral of (1/2)*u^2 with u=2 over [0, 10] equals 20 (c=1)."""
    t = np.linspace(0.0, 10.0, 101)
    u = np.full_like(t, 2.0)
    # 0.5 * 1 * 2^2 * 10 = 20
    assert cost_L2(t, u) == pytest.approx(20.0)


def test_cost_L2_weight_scales_linearly() -> None:
    """cost_L2 is linear in the quadratic weight c."""
    t = np.linspace(0.0, 10.0, 101)
    u = np.full_like(t, 2.0)
    assert cost_L2(t, u, c=2.0) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# relative_error
# ---------------------------------------------------------------------------


def test_relative_error_basic() -> None:
    """|value - reference| / |reference| for a known pair."""
    assert relative_error(11.0, 10.0) == pytest.approx(0.1)


def test_relative_error_negative_reference_uses_absolute() -> None:
    """The denominator uses |reference|, so sign does not matter."""
    assert relative_error(-8.0, -10.0) == pytest.approx(0.2)


def test_relative_error_zero_reference_raises() -> None:
    """A zero reference has no defined relative error and must raise."""
    with pytest.raises(ValueError, match="non-zero"):
        relative_error(1.0, 0.0)


# ---------------------------------------------------------------------------
# suppression_time
# ---------------------------------------------------------------------------


def test_suppression_time_first_crossing() -> None:
    """Returns the first time at which F drops to/below epsilon."""
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    F = np.array([100.0, 80.0, 40.0, 20.0, 10.0])
    # epsilon=50 first satisfied at index 2 (F=40), t=2.0
    assert suppression_time(t, F, epsilon=50.0) == pytest.approx(2.0)


def test_suppression_time_never_reached_returns_none() -> None:
    """If F never reaches epsilon the result is None."""
    t = np.array([0.0, 1.0, 2.0])
    F = np.array([100.0, 90.0, 80.0])
    assert suppression_time(t, F, epsilon=10.0) is None


def test_suppression_time_tolerance_admits_near_miss() -> None:
    """A positive tolerance lets F_min = epsilon + tol still count as reached."""
    t = np.array([0.0, 1.0, 2.0])
    F = np.array([100.0, 60.0, 50.5])
    # Without tolerance, epsilon=50 is never reached (min is 50.5).
    assert suppression_time(t, F, epsilon=50.0) is None
    # With tol=1.0, threshold becomes 51.0 and F=50.5 at t=2.0 qualifies.
    assert suppression_time(t, F, epsilon=50.0, tol=1.0) == pytest.approx(2.0)


def test_suppression_time_exact_threshold_counts() -> None:
    """F exactly equal to epsilon counts as reached (<= comparison)."""
    t = np.array([0.0, 1.0, 2.0])
    F = np.array([100.0, 50.0, 30.0])
    assert suppression_time(t, F, epsilon=50.0) == pytest.approx(1.0)
