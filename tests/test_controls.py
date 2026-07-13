"""Unit tests for the control-law factories.

Covers :mod:`sit_control.controls`: the constant/zero/impulsive/interpolated
control factories, including the boundary behaviour of the impulsive pulses and
the validation paths (positive duration, matching grid shapes). All tests are
pure Python (no GEKKO, no ODE integration) and therefore fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.controls import (
    constant_control,
    impulsive_control,
    interpolated_control,
    zero_control,
)

# ---------------------------------------------------------------------------
# constant_control / zero_control
# ---------------------------------------------------------------------------


def test_constant_control_returns_value_everywhere() -> None:
    """constant_control(u) must return u for every time queried."""
    law = constant_control(42.0)
    assert law(0.0) == 42.0
    assert law(100.0) == 42.0
    assert law(-5.0) == 42.0


def test_zero_control_is_zero() -> None:
    """zero_control must return 0 at every time."""
    law = zero_control()
    assert law(0.0) == 0.0
    assert law(37.5) == 0.0


# ---------------------------------------------------------------------------
# impulsive_control
# ---------------------------------------------------------------------------


def test_impulsive_control_rate_inside_pulse() -> None:
    """Inside a pulse window the rate is amount/duration (analytic value)."""
    law = impulsive_control(np.array([10.0]), amount=100.0, duration=0.5)
    # rate = 100 / 0.5 = 200
    assert law(10.0) == pytest.approx(200.0)
    assert law(10.25) == pytest.approx(200.0)


def test_impulsive_control_zero_outside_pulse() -> None:
    """Outside every pulse window the control is exactly zero."""
    law = impulsive_control(np.array([10.0, 20.0]), amount=100.0, duration=0.5)
    assert law(5.0) == 0.0
    assert law(15.0) == 0.0
    assert law(30.0) == 0.0


def test_impulsive_control_boundaries_half_open() -> None:
    """The pulse window is half-open [start, start+duration)."""
    law = impulsive_control(np.array([10.0]), amount=100.0, duration=0.5)
    # start included ...
    assert law(10.0) == pytest.approx(200.0)
    # ... end excluded.
    assert law(10.5) == 0.0


def test_impulsive_control_pulse_integrates_to_amount() -> None:
    """Rate * duration must equal the requested amount per pulse."""
    amount, duration = 250.0, 0.5
    law = impulsive_control(np.array([0.0]), amount=amount, duration=duration)
    assert law(0.0) * duration == pytest.approx(amount)


def test_impulsive_control_nonpositive_duration_raises() -> None:
    """duration <= 0 must raise ValueError (guarded path)."""
    with pytest.raises(ValueError, match="duration must be positive"):
        impulsive_control(np.array([0.0]), amount=100.0, duration=0.0)
    with pytest.raises(ValueError, match="duration must be positive"):
        impulsive_control(np.array([0.0]), amount=100.0, duration=-1.0)


# ---------------------------------------------------------------------------
# interpolated_control
# ---------------------------------------------------------------------------


def test_interpolated_control_at_grid_points() -> None:
    """At grid nodes the interpolant reproduces the tabulated values."""
    t_grid = np.array([0.0, 1.0, 2.0])
    u_grid = np.array([0.0, 10.0, 20.0])
    law = interpolated_control(t_grid, u_grid)
    assert law(0.0) == pytest.approx(0.0)
    assert law(1.0) == pytest.approx(10.0)
    assert law(2.0) == pytest.approx(20.0)


def test_interpolated_control_linear_midpoint() -> None:
    """Between two nodes the value is the linear interpolation."""
    t_grid = np.array([0.0, 2.0])
    u_grid = np.array([0.0, 20.0])
    law = interpolated_control(t_grid, u_grid)
    assert law(1.0) == pytest.approx(10.0)


def test_interpolated_control_zero_outside_grid() -> None:
    """Outside the grid the control is clamped to 0 (left=right=0)."""
    t_grid = np.array([0.0, 1.0])
    u_grid = np.array([5.0, 15.0])
    law = interpolated_control(t_grid, u_grid)
    assert law(-1.0) == 0.0
    assert law(10.0) == 0.0


def test_interpolated_control_shape_mismatch_raises() -> None:
    """Mismatched grid shapes must raise ValueError."""
    with pytest.raises(ValueError, match="same shape"):
        interpolated_control(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
