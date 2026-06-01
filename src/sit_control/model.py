"""Mosquito population dynamics: full (S2) and reduced (S1) models.

Implements the ODE systems of Almeida et al. (2022), equations (2) and (3).
The reduced model S1 uses the quasi-stationary approximation described in
Section 2.2 of the same article.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import BiologicalParameters


def recruitment(
    F: float | NDArray[np.float64],
    Ms: float | NDArray[np.float64],
    params: BiologicalParameters,
    *,
    singular_eps: float = 1e-12,
) -> float | NDArray[np.float64]:
    """Non-linear recruitment function f(F, M_s) of the reduced model S1.

    Implements equation (7) of Almeida et al. (2022). The function has a removable
    singularity at (F, M_s) = (0, 0); the implementation returns the
    asymptotic value -delta_F * F when the denominator falls below
    `singular_eps`.

    Args:
        F: Female population (scalar or array).
        Ms: Sterile male population (scalar or array).
        params: Biological parameters.
        singular_eps: Threshold for the removable singularity.

    Returns:
        Recruitment rate at (F, M_s).

    Note:
        The function is vectorised for use with NumPy arrays, with a
        graceful scalar fallback for use inside ODE integrators.
    """
    p = params
    denom_E = p.beta_E * F / p.K + p.nu_E + p.delta_E
    numerator = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
    denom = denom_E * (
        (1.0 - p.nu) * p.nu_E * p.beta_E * F
        + p.delta_M * p.gamma_s * Ms * denom_E
    )

    # The recruitment term has a removable singularity ONLY at (F, Ms) = (0, 0)
    # (both numerator and denom -> 0, ratio -> 0). Use a RELATIVE threshold so a
    # small-but-legitimate denominator (finite, well-defined ratio) is never
    # discarded; the asymptotic limit -delta_F*F is applied only at the genuine
    # singularity. In the operating regime denom is many orders above threshold,
    # so this never fires and changes no result.
    threshold = singular_eps * (1.0 + numerator)
    if np.isscalar(F):
        if denom < threshold:
            return -p.delta_F * F
        return numerator / denom - p.delta_F * F

    out = np.where(
        denom < threshold,
        -p.delta_F * F,
        np.divide(numerator, denom, where=denom >= threshold) - p.delta_F * F,
    )
    return out


def rhs_reduced(
    t: float,
    state: NDArray[np.float64],
    u_func: callable,
    params: BiologicalParameters,
    singular_eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Right-hand side of the reduced system S1.

    Args:
        t: Current time.
        state: Vector [F, Ms] of state variables.
        u_func: Control law, a callable t -> u(t).
        params: Biological parameters.
        singular_eps: Threshold for the singularity of f.

    Returns:
        Time derivative [dF/dt, dMs/dt].
    """
    F, Ms = state
    u = float(u_func(t))
    dF = recruitment(F, Ms, params, singular_eps=singular_eps)
    dMs = u - params.delta_s * Ms
    return np.array([dF, dMs], dtype=np.float64)


def rhs_full(
    t: float,
    state: NDArray[np.float64],
    u_func: callable,
    params: BiologicalParameters,
) -> NDArray[np.float64]:
    """Right-hand side of the full system S2.

    State vector: [E, M, F, M_s].

    Args:
        t: Current time.
        state: Vector [E, M, F, Ms] of state variables.
        u_func: Control law, a callable t -> u(t).
        params: Biological parameters.

    Returns:
        Time derivative [dE/dt, dM/dt, dF/dt, dMs/dt].
    """
    E, M, F, Ms = state
    p = params
    u = float(u_func(t))

    dE = p.beta_E * F * (1.0 - E / p.K) - (p.nu_E + p.delta_E) * E
    dM = (1.0 - p.nu) * p.nu_E * E - p.delta_M * M
    mating = M / (M + p.gamma_s * Ms) if (M + p.gamma_s * Ms) > 1e-12 else 0.0
    dF = p.nu * p.nu_E * E * mating - p.delta_F * F
    dMs = u - p.delta_s * Ms

    return np.array([dE, dM, dF, dMs], dtype=np.float64)