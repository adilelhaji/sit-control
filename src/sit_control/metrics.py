"""Metrics for evaluating control strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cost_L1(
    t: NDArray[np.float64],
    u: NDArray[np.float64],
) -> float:
    """L^1 cost: total number of sterile mosquitoes released.

    Args:
        t: Time grid.
        u: Control values at each grid point.

    Returns:
        Integral of u over [t[0], t[-1]].
    """
    return float(np.trapezoid(u, t))


def cost_L2(
    t: NDArray[np.float64],
    u: NDArray[np.float64],
    c: float = 1.0,
) -> float:
    """L^2 cost: quadratic penalty on the release profile.

    Args:
        t: Time grid.
        u: Control values at each grid point.
        c: Quadratic penalty weight.

    Returns:
        Integral of (c/2) * u^2 over [t[0], t[-1]].
    """
    return float(np.trapezoid(0.5 * c * u**2, t))


def relative_error(value: float, reference: float) -> float:
    """Relative error between a computed value and a reference.

    Args:
        value: Computed value.
        reference: Reference value (non-zero).

    Returns:
        Absolute relative error |value - reference| / |reference|.
    """
    if reference == 0:
        raise ValueError("reference must be non-zero")
    return abs(value - reference) / abs(reference)


def suppression_time(
    t: NDArray[np.float64],
    F: NDArray[np.float64],
    epsilon: float,
    tol: float = 0.0,
) -> float | None:
    """First time at which F(t) reaches the suppression threshold.

    Args:
        t: Time grid.
        F: Female trajectory.
        epsilon: Suppression threshold.
        tol: Absolute tolerance added to epsilon. Use when the solver
             converges to F_min = epsilon ± tol (e.g. bisection with
             tolerance 1e-3) to avoid spurious None results.

    Returns:
        Smallest t such that F(t) <= epsilon + tol, or None if never reached.
    """
    indices = np.where(epsilon + tol >= F)[0]
    if len(indices) == 0:
        return None
    return float(t[indices[0]])
