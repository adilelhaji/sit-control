"""Bisection algorithm for the optimal control problem.

Reproduces Algorithm 2 (Figure 2) of Almeida et al. (2022). Exploits
Theorem 3.3: for T > T_opt ≈ 103 days the optimal structure is RIGHT-
ALIGNED — u*=0 on [0, T-τ₁] then singular arc on [T-τ₁, T]. The
bang-ON-first structure applies only for T ≈ T_min ≈ 60 days. The
bisection searches over τ₁ (length of singular arc) in O(log N) time,
roughly 20× faster than GEKKO at the same discretisation (Almeida 2022,
Table 4: 1.44 s vs 29.4 s at N=300).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .controls import constant_control, interpolated_control
from .model import recruitment
from .parameters import BiologicalParameters, ControlConfig
from .simulator import Simulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BisectionResult:
    """Result of the bisection algorithm.

    Attributes:
        tau1: Length of the singular arc.
        t0: Start of the singular arc.
        t1: End of the singular arc.
        F_min: Minimum value of F reached on [0, T].
        iterations: Number of bisection iterations performed.
        converged: Whether F_min <= epsilon was reached.
    """

    tau1: float
    t0: float
    t1: float
    F_min: float
    iterations: int
    converged: bool


def singular_control(
    F: float,
    Ms: float,
    params: BiologicalParameters,
) -> float:
    """Compute the singular control u_sing(F, M_s).

    Implements equation (7) of Almeida et al. (2022), derived from
    the conditions of the Pontryagin Maximum Principle on the
    singular arc.

    Args:
        F: Female population.
        Ms: Sterile male population.
        params: Biological parameters.

    Returns:
        Singular control value at (F, M_s).
    """
    p = params
    df_dF, df_dMs, d2f_dMs2, d2f_dMsdF = _compute_partials(F, Ms, params)

    if abs(d2f_dMs2) < 1e-30:
        # Degenerate curvature — return 0 to stay within admissible set.
        return 0.0

    f_value = _compute_f(F, Ms, params)
    numerator = (
        df_dMs * df_dF + d2f_dMs2 * p.delta_s * Ms - d2f_dMsdF * f_value
    )
    return numerator / d2f_dMs2


def _compute_f(F: float, Ms: float, p: BiologicalParameters) -> float:
    """Internal: value of the recruitment f at (F, M_s)."""
    denom_E = p.beta_E * F / p.K + p.nu_E + p.delta_E
    numerator = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
    denom = denom_E * (
        (1.0 - p.nu) * p.nu_E * p.beta_E * F
        + p.delta_M * p.gamma_s * Ms * denom_E
    )
    if denom < 1e-12:
        return -p.delta_F * F
    return numerator / denom - p.delta_F * F


def _compute_partials(
    F: float,
    Ms: float,
    p: BiologicalParameters,
) -> tuple[float, float, float, float]:
    """Internal: partial derivatives of f at (F, M_s).

    Computed via numerical differentiation with step h = 1e-4.
    For production use, replace with symbolic differentiation.
    """
    h = 1e-4
    f0 = _compute_f(F, Ms, p)
    df_dF = (_compute_f(F + h, Ms, p) - f0) / h
    df_dMs = (_compute_f(F, Ms + h, p) - f0) / h
    d2f_dMs2 = (
        _compute_f(F, Ms + h, p) - 2 * f0 + _compute_f(F, Ms - h, p)
    ) / h**2
    d2f_dMsdF = (
        _compute_f(F + h, Ms + h, p) - _compute_f(F + h, Ms - h, p)
        - _compute_f(F - h, Ms + h, p) + _compute_f(F - h, Ms - h, p)
    ) / (4 * h**2)
    return df_dF, df_dMs, d2f_dMs2, d2f_dMsdF


def solve_by_bisection(
    params: BiologicalParameters,
    control_config: ControlConfig,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
) -> BisectionResult:
    """Solve the optimal control problem by bisection on tau_1.

    Reproduces Algorithm 2 of Almeida et al. (2022) for the case
    T > T_opt, where the optimal structure is zero-arc → singular arc
    (right-aligned): u*(t) = 0 on [0, T-τ₁], singular on [T-τ₁, T].
    The bang-ON-first structure only applies when T ≈ T_min (~60 days).

    Args:
        params: Biological parameters.
        control_config: Operational configuration (T, U_max, epsilon).
        max_iterations: Maximum number of bisection steps.
        tolerance: Convergence tolerance on F_min - epsilon.

    Returns:
        BisectionResult with the optimal switching times.
    """
    cfg = control_config
    epsilon = cfg.epsilon if cfg.epsilon is not None else params.F_bar / 4.0

    tau1_min, tau1_max = 0.0, cfg.T
    simulator = Simulator(params)

    logger.info(
        "Starting bisection: T=%g, U_max=%g, epsilon=%g",
        cfg.T, cfg.U_max, epsilon,
    )

    for i in range(max_iterations):
        tau1_test = 0.5 * (tau1_min + tau1_max)
        # Bang-zero for [0, t0], singular for [t0, T] (right-aligned, Thm 3.3)
        F_min = _simulate_with_tau1(
            params, cfg, tau1_test, simulator,
        )

        if abs(F_min - epsilon) < tolerance:
            return BisectionResult(
                tau1=tau1_test,
                t0=cfg.T - tau1_test,
                t1=cfg.T,
                F_min=F_min,
                iterations=i + 1,
                converged=True,
            )

        if F_min < epsilon:
            tau1_max = tau1_test
        else:
            tau1_min = tau1_test

    return BisectionResult(
        tau1=0.5 * (tau1_min + tau1_max),
        t0=cfg.T - 0.5 * (tau1_min + tau1_max),
        t1=cfg.T,
        F_min=F_min,
        iterations=max_iterations,
        converged=False,
    )


def _simulate_with_tau1(
    params: BiologicalParameters,
    cfg: ControlConfig,
    tau1: float,
    simulator: Simulator,
) -> float:
    """Simulate the bang-singular control for a given tau_1 and return F_min.

    Implements the inner loop of Algorithm 2 (Figure 2) of Almeida et al.
    (2022). The control has a right-aligned two-phase structure with
    t_0 = T - tau_1 (for T > T_opt ≈ 103 days, Almeida 2022, Thm 3.3):

    * **Phase 1 (bang-zero):** t in [0, t_0],  u(t) = 0.
    * **Phase 2 (singular):** t in [t_0, T], u(t) = u_sing(F(t), M_s(t))
      clipped to [0, U_max] to enforce the admissibility constraint.

    Note: bang-ON-first structure applies only for T ≈ T_min ≈ 60 days.

    Parameters
    ----------
    params : BiologicalParameters
        Biological parameters of the mosquito population model.
    cfg : ControlConfig
        Operational configuration (T, U_max, epsilon).
    tau1 : float
        Duration of the singular arc in days (= t_1 - t_0 = T - t_0).
    simulator : Simulator
        Simulator instance that provides the numerical configuration.

    Returns
    -------
    float
        Minimum female population min_{t in [0, T]} F(t).

    References
    ----------
    TFG Algorithm 2 and Almeida et al. (2022), Figure 2.
    """
    from scipy.integrate import solve_ivp

    T = cfg.T
    t0 = T - tau1
    num_cfg = simulator.config
    initial = np.array([params.F_bar, 0.0], dtype=np.float64)

    # ---- Phase 1: bang-zero [0, t0] with u = 0 (Algorithm 2, Thm 3.3) ----
    F_min = params.F_bar
    state_t0 = initial.copy()

    if t0 > 1e-10:
        res1 = simulator.simulate(
            T=t0,
            u_func=constant_control(0.0),
            model="S1",
            initial_state=initial,
        )
        state_t0 = res1.state[:, -1].copy()
        F_min = float(np.min(res1.state[0]))

    if tau1 < 1e-10:
        return F_min

    # ---- Phase 2: singular arc [t0, T] with state-feedback u_sing --------
    def _rhs_singular(t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        F_s = float(max(state[0], 0.0))
        Ms_s = float(max(state[1], 0.0))
        u_s = float(np.clip(
            singular_control(F_s, Ms_s, params), 0.0, cfg.U_max,
        ))
        dF = float(recruitment(
            F_s, Ms_s, params, singular_eps=num_cfg.singular_eps,
        ))
        dMs = u_s - params.delta_s * Ms_s
        return np.array([dF, dMs], dtype=np.float64)

    sol2 = solve_ivp(
        fun=_rhs_singular,
        t_span=(t0, T),
        y0=state_t0,
        method=num_cfg.solver_method,
        rtol=num_cfg.rtol,
        atol=num_cfg.atol,
    )

    if sol2.success and sol2.y.shape[1] > 0:
        F_min = min(F_min, float(np.min(sol2.y[0])))

    return F_min