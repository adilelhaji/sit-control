"""Control law factories for SIT simulations.

Each factory returns a callable t -> u(t) suitable for use with the
Simulator class.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

ControlLaw = Callable[[float], float]


def constant_control(u_value: float) -> ControlLaw:
    """Constant release rate.

    Args:
        u_value: Release rate to apply at every time.

    Returns:
        Callable t -> u_value.
    """

    def _u(t: float) -> float:
        return u_value

    return _u


def zero_control() -> ControlLaw:
    """Zero control (no releases)."""
    return constant_control(0.0)


def impulsive_control(
    times: NDArray[np.float64],
    amount: float,
    duration: float = 0.5,
) -> ControlLaw:
    """Periodic impulsive release.

    Approximates Dirac releases by rectangular pulses of finite width
    `duration`, so that the total amount per pulse is `amount`.

    Pulses are assumed disjoint: `duration` must not exceed the smallest
    gap between consecutive `times`. If two windows overlap, the rate of
    the first matching window is returned (rates are not summed).

    Args:
        times: Array of pulse start times.
        amount: Total individuals released per pulse.
        duration: Width of each rectangular pulse in days (must be > 0).

    Returns:
        Callable t -> u(t).

    Raises:
        ValueError: If duration <= 0.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    rate = amount / duration
    times = np.asarray(times, dtype=np.float64)

    def _u(t: float) -> float:
        for start in times:
            if start <= t < start + duration:
                return rate
        return 0.0

    return _u


def interpolated_control(
    t_grid: NDArray[np.float64],
    u_grid: NDArray[np.float64],
) -> ControlLaw:
    """Piecewise-linear interpolation of a discrete control profile.

    Useful for using the optimal control returned by GEKKO inside the
    SciPy integrator.

    Args:
        t_grid: Time grid where the control is known.
        u_grid: Control values at each grid point.

    Returns:
        Callable t -> u(t) linearly interpolated.
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64)
    if t_grid.shape != u_grid.shape:
        raise ValueError("t_grid and u_grid must have the same shape")

    def _u(t: float) -> float:
        return float(np.interp(t, t_grid, u_grid, left=0.0, right=0.0))

    return _u
