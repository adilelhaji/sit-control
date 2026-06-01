"""Bisection algorithm for the optimal control problem.

Reproduces Algorithm 2 (Figure 2) of Almeida et al. (2022). Exploits
Theorem 3.3: for T > T_opt ≈ 103 days the optimal structure is RIGHT-
ALIGNED — u*=0 on [0, T-τ₂] then singular arc on [T-τ₂, T-τ₂+τ₁] then
u*=0 on [T-τ₂+τ₁, T]. The bang-ON-first structure applies only for
T ≈ T_min ≈ 60 days. The bisection searches over τ₁ (length of singular
arc) in O(log N) time, roughly 20× faster than GEKKO at the same
discretisation (Almeida 2022, Table 4: 1.44 s vs 29.4 s at N=300).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .controls import constant_control, interpolated_control
from .model import recruitment
from .parameters import BiologicalParameters, ControlConfig
from .simulator import Simulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BisectionResult:
    """Result of the bisection algorithm.

    Attributes:
        tau1: Duration of the singular arc.
        tau2: Duration from start of singular arc to minimum of F
              (tau2 >= tau1; the terminal u=0 phase has length tau2-tau1).
        t0: Start of the singular arc = T - tau2.
        t1: End of the singular arc   = T - tau2 + tau1.
        F_min: Minimum value of F reached on [0, T].
        iterations: Number of bisection iterations performed.
        converged: Whether F_min <= epsilon was reached.
    """

    tau1: float
    tau2: float
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
    """Analytical partial derivatives of f at (F, M_s).

    Let:
        D_E = beta_E*F/K + nu_E + delta_E
        A   = nu*(1-nu)*beta_E^2*nu_E^2*F^2        (numerator of f + delta_F*F)
        B   = (1-nu)*nu_E*beta_E*F
        Cs  = delta_M*gamma_s*Ms
        D   = D_E*(B + Cs*D_E)                     (denominator)

    Then f = A/D - delta_F*F and the four derivatives are:

        df/dF      = (A'*D - A*D') / D^2  - delta_F
        df/dMs     = -A * delta_M*gamma_s * D_E^2 / D^2
        d2f/dMs2   =  2*A*(delta_M*gamma_s)^2 * D_E^4 / D^3
        d2f/dMs dF = delta_M*gamma_s * [h'(F)/D^2 + 2*A*D_E^2*D'/D^3]

    where h'(F) = -(dA/dF * D_E^2 + 2*A*D_E * dD_E/dF).

    All expressions are closed-form; no finite differences are used.
    This reduces singular_control evaluation from O(6 ODE calls) to O(1).

    Returns:
        (df/dF, df/dMs, d²f/dMs², d²f/dMs dF)
    """
    F  = max(F, 1e-12)
    Ms = max(Ms, 0.0)

    # ── Intermediate quantities ───────────────────────────────────────────
    dE  = p.beta_E * F / p.K + p.nu_E + p.delta_E          # D_E
    A   = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
    B   = (1.0 - p.nu) * p.nu_E * p.beta_E * F
    Cs  = p.delta_M * p.gamma_s * Ms
    D   = dE * (B + Cs * dE)

    if D < 1e-30:
        return -p.delta_F, 0.0, 0.0, 0.0

    # ── Derivatives of intermediates w.r.t. F ────────────────────────────
    ddE_dF = p.beta_E / p.K
    dA_dF  = 2.0 * p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F
    dB_dF  = (1.0 - p.nu) * p.nu_E * p.beta_E
    # D = dE*(B + Cs*dE)  →  dD/dF = ddE_dF*(B+Cs*dE) + dE*(dB_dF+Cs*ddE_dF)
    dD_dF  = ddE_dF * (B + Cs * dE) + dE * (dB_dF + Cs * ddE_dF)

    # D w.r.t. Ms:  dD/dMs = delta_M*gamma_s * dE^2
    dD_dMs = p.delta_M * p.gamma_s * dE**2

    # ── ∂f/∂F ────────────────────────────────────────────────────────────
    df_dF = (dA_dF * D - A * dD_dF) / D**2 - p.delta_F

    # ── ∂f/∂Ms  = -A * delta_M*gamma_s * D_E^2 / D^2 ────────────────────
    df_dMs = -A * dD_dMs / D**2

    # ── ∂²f/∂Ms²  = 2*A*(delta_M*gamma_s)^2 * D_E^4 / D^3 ───────────────
    d2f_dMs2 = 2.0 * A * (p.delta_M * p.gamma_s)**2 * dE**4 / D**3

    # ── ∂²f/∂Ms∂F  (differentiate ∂f/∂Ms w.r.t. F) ─────────────────────
    # h(F) = -A*delta_M*gamma_s*D_E^2  →  h'(F) = -delta_M*gamma_s*(A'*dE^2 + 2*A*dE*dE')
    h_prime = -p.delta_M * p.gamma_s * (dA_dF * dE**2 + 2.0 * A * dE * ddE_dF)
    d2f_dMsdF = h_prime / D**2 + 2.0 * A * dD_dMs * dD_dF / D**3

    return df_dF, df_dMs, d2f_dMs2, d2f_dMsdF


def _simulate_singular_from_Fbar(
    params: BiologicalParameters,
    cfg: ControlConfig,
    tau1: float,
    num_cfg,
    n_points: int = 500,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Simulate singular control for τ₁ days starting from (F̄, 0).

    This is the inner simulation of Algorithm 2: it produces the
    control profile u^τ₁ and state trajectory that will later be
    time-shifted to build the right-aligned formula (9) control.

    Returns:
        t_sing : time grid on [0, tau1]
        F_sing : female trajectory
        Ms_sing: sterile-male trajectory
        u_sing : singular control values (clipped to [0, U_max])
    """
    initial = np.array([params.F_bar, 0.0], dtype=np.float64)

    # tau1 == 0: no singular arc. Return a single point at (F_bar, 0) instead
    # of integrating an empty [0, 0] span, which yields an empty solution and
    # an IndexError downstream.
    if tau1 <= 0.0:
        u0 = float(np.clip(
            singular_control(params.F_bar, 0.0, params), 0.0, cfg.U_max
        ))
        return (np.array([0.0]), np.array([params.F_bar]),
                np.array([0.0]), np.array([u0]))

    def _rhs(t: float, state: NDArray) -> NDArray:
        F_s  = float(max(state[0], 0.0))
        Ms_s = float(max(state[1], 0.0))
        u_s  = float(np.clip(singular_control(F_s, Ms_s, params), 0.0, cfg.U_max))
        dF   = float(recruitment(F_s, Ms_s, params, singular_eps=num_cfg.singular_eps))
        dMs  = u_s - params.delta_s * Ms_s
        return np.array([dF, dMs], dtype=np.float64)

    t_eval = np.linspace(0.0, tau1, max(10, n_points))
    sol = solve_ivp(
        fun=_rhs,
        t_span=(0.0, tau1),
        y0=initial,
        method=num_cfg.solver_method,
        t_eval=t_eval,
        rtol=num_cfg.rtol,
        atol=num_cfg.atol,
    )

    t_s  = sol.t
    F_s  = sol.y[0]
    Ms_s = sol.y[1]
    u_s  = np.array([
        float(np.clip(singular_control(
            float(max(F_s[i], 0.0)), float(max(Ms_s[i], 0.0)), params
        ), 0.0, cfg.U_max))
        for i in range(len(t_s))
    ])
    return t_s, F_s, Ms_s, u_s


def _find_tau2(
    params: BiologicalParameters,
    cfg: ControlConfig,
    tau1: float,
    state_at_tau1: NDArray,
    num_cfg,
    extra_time: float | None = None,
) -> tuple[float, float]:
    """Find τ₂ = argmin F(t) after applying u=0 from end of singular arc.

    After the singular arc ends (at time τ₁ from (F̄,0)), the control
    switches to u=0. F continues to decrease for a while then rebounds.
    τ₂ is the time of minimum F counting from t=0 (start of singular arc).

    Returns:
        tau2   : time of minimum F (measured from start of singular arc)
        F_min  : minimum F value = F(tau2)
    """
    if extra_time is None:
        extra_time = max(cfg.T, 2.0 * tau1)

    def _rhs_zero(t: float, state: NDArray) -> NDArray:
        F_s  = float(max(state[0], 0.0))
        Ms_s = float(max(state[1], 0.0))
        dF   = float(recruitment(F_s, Ms_s, params, singular_eps=num_cfg.singular_eps))
        dMs  = -params.delta_s * Ms_s
        return np.array([dF, dMs], dtype=np.float64)

    n_ext = max(200, int(extra_time * 10))
    t_eval_ext = np.linspace(tau1, tau1 + extra_time, n_ext)

    sol = solve_ivp(
        fun=_rhs_zero,
        t_span=(tau1, tau1 + extra_time),
        y0=state_at_tau1,
        method=num_cfg.solver_method,
        t_eval=t_eval_ext,
        rtol=num_cfg.rtol,
        atol=num_cfg.atol,
    )

    F_ext = sol.y[0]
    t_ext = sol.t

    # F_min over the complete trajectory [0, tau1+extra_time]
    # The singular phase min is F_sing[-1] (F is decreasing during singular arc).
    # After the arc, under u=0, F decreases further until it reaches its minimum.
    idx_min = int(np.argmin(F_ext))
    F_min   = float(F_ext[idx_min])
    tau2    = float(t_ext[idx_min])  # measured from t=0 (start of singular arc)

    return tau2, F_min


def solve_by_bisection(
    params: BiologicalParameters,
    control_config: ControlConfig,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
) -> BisectionResult:
    """Solve the optimal control problem by bisection on tau_1.

    Reproduces Algorithm 2 of Almeida et al. (2022) for the case
    T > T_opt, where the optimal structure (formula 9) has three phases:

        u*(t) = 0          on [0,   T - τ₂]
        u*(t) = u^τ₁(t-t₀) on [t₀,  t₁]       t₀ = T-τ₂, t₁ = T-τ₂+τ₁
        u*(t) = 0          on [t₁,  T]

    The bisection searches for τ₁ (singular arc duration) such that
    F_min = ε. For each τ₁, τ₂ is determined by finding the time of
    minimum F when the singular arc starts from (F̄, 0) and is followed
    by u=0 until F stops decreasing.

    Args:
        params: Biological parameters.
        control_config: Operational configuration (T, U_max, epsilon).
        max_iterations: Maximum number of bisection steps.
        tolerance: Convergence tolerance on |F_min - epsilon|.

    Returns:
        BisectionResult with the optimal switching times.
    """
    cfg = control_config
    epsilon = cfg.epsilon if cfg.epsilon is not None else params.F_bar / 4.0

    tau1_min, tau1_max = 0.0, cfg.T
    simulator = Simulator(params)
    num_cfg   = simulator.config

    logger.info(
        "Starting bisection: T=%g, U_max=%g, epsilon=%g",
        cfg.T, cfg.U_max, epsilon,
    )

    F_min = params.F_bar
    tau2  = cfg.T

    for i in range(max_iterations):
        tau1_test = 0.5 * (tau1_min + tau1_max)

        # Simulate singular arc from (F̄,0) for tau1_test days
        t_s, F_s, Ms_s, u_s = _simulate_singular_from_Fbar(
            params, cfg, tau1_test, num_cfg,
        )
        state_end = np.array([float(F_s[-1]), float(Ms_s[-1])], dtype=np.float64)

        # Extend with u=0 to find τ₂ = argmin F
        tau2_test, F_min_test = _find_tau2(
            params, cfg, tau1_test, state_end, num_cfg,
        )

        F_min = F_min_test

        logger.debug(
            "iter %d: tau1=%.3f, tau2=%.3f, F_min=%.4f (eps=%.4f)",
            i, tau1_test, tau2_test, F_min_test, epsilon,
        )

        if abs(F_min_test - epsilon) < tolerance:
            tau2 = tau2_test
            t0   = cfg.T - tau2
            t1   = t0 + tau1_test
            return BisectionResult(
                tau1=tau1_test,
                tau2=tau2,
                t0=max(t0, 0.0),
                t1=min(t1, cfg.T),
                F_min=F_min_test,
                iterations=i + 1,
                converged=True,
            )

        if F_min_test < epsilon:
            tau1_max = tau1_test
        else:
            tau1_min = tau1_test

        tau2 = tau2_test

    tau1_final = 0.5 * (tau1_min + tau1_max)
    t0_final   = cfg.T - tau2
    t1_final   = t0_final + tau1_final

    return BisectionResult(
        tau1=tau1_final,
        tau2=tau2,
        t0=max(t0_final, 0.0),
        t1=min(t1_final, cfg.T),
        F_min=F_min,
        iterations=max_iterations,
        converged=False,
    )


def build_formula9_trajectory(
    params: BiologicalParameters,
    cfg: ControlConfig,
    bis: BisectionResult,
    num_cfg,
    n_eval: int = 2000,
) -> tuple[NDArray, NDArray, NDArray]:
    """Reconstruct the formula (9) control and state trajectory over [0, T].

    The right-aligned three-phase control:
        Phase 1  [0, t0]   : u = 0
        Phase 2  [t0, t1]  : u = u^τ₁(t - t0)  (singular arc, time-shifted)
        Phase 3  [t1, T]   : u = 0

    Args:
        params  : Biological parameters.
        cfg     : Control configuration (T, U_max, epsilon).
        bis     : Result of solve_by_bisection.
        num_cfg : Numerical configuration.
        n_eval  : Total number of output points.

    Returns:
        t     : time grid over [0, T]
        F_traj: female population trajectory
        u_traj: control profile
    """
    T  = cfg.T
    t0 = bis.t0
    t1 = bis.t1

    # ── Phase 1: u=0 on [0, t0] ──────────────────────────────────────────
    n1 = max(2, int(t0 / T * n_eval)) if t0 > 1e-10 else 2
    initial = np.array([params.F_bar, 0.0], dtype=np.float64)

    t_ph1 = np.array([0.0, min(t0, T)])
    F_ph1 = np.array([params.F_bar, params.F_bar])
    u_ph1 = np.zeros(2)
    state_at_t0 = initial.copy()

    if t0 > 1e-10:
        def _rhs_zero(t: float, state: NDArray) -> NDArray:
            F_s  = float(max(state[0], 0.0))
            Ms_s = float(max(state[1], 0.0))
            dF   = float(recruitment(F_s, Ms_s, params, singular_eps=num_cfg.singular_eps))
            dMs  = -params.delta_s * Ms_s
            return np.array([dF, dMs], dtype=np.float64)

        t_eval_1 = np.linspace(0.0, t0, n1)
        sol1 = solve_ivp(
            fun=_rhs_zero,
            t_span=(0.0, t0),
            y0=initial,
            method=num_cfg.solver_method,
            t_eval=t_eval_1,
            rtol=num_cfg.rtol,
            atol=num_cfg.atol,
        )
        t_ph1         = sol1.t
        F_ph1         = sol1.y[0]
        u_ph1         = np.zeros_like(sol1.t)
        state_at_t0   = sol1.y[:, -1].copy()

    # ── Phase 2: singular arc on [t0, t1] ────────────────────────────────
    tau1 = bis.tau1
    n2   = max(10, int(tau1 / T * n_eval))

    def _rhs_singular(t: float, state: NDArray) -> NDArray:
        F_s  = float(max(state[0], 0.0))
        Ms_s = float(max(state[1], 0.0))
        u_s  = float(np.clip(singular_control(F_s, Ms_s, params), 0.0, cfg.U_max))
        dF   = float(recruitment(F_s, Ms_s, params, singular_eps=num_cfg.singular_eps))
        dMs  = u_s - params.delta_s * Ms_s
        return np.array([dF, dMs], dtype=np.float64)

    t_eval_2 = np.linspace(t0, t1, n2)
    sol2 = solve_ivp(
        fun=_rhs_singular,
        t_span=(t0, t1),
        y0=state_at_t0,
        method=num_cfg.solver_method,
        t_eval=t_eval_2,
        rtol=num_cfg.rtol,
        atol=num_cfg.atol,
    )
    t_ph2 = sol2.t
    F_ph2 = sol2.y[0]
    Ms_ph2 = sol2.y[1]
    u_ph2 = np.array([
        float(np.clip(singular_control(
            float(max(F_ph2[i], 0.0)), float(max(Ms_ph2[i], 0.0)), params
        ), 0.0, cfg.U_max))
        for i in range(len(t_ph2))
    ])
    state_at_t1 = sol2.y[:, -1].copy()

    # ── Phase 3: u=0 on [t1, T] ──────────────────────────────────────────
    t_ph3 = np.array([t1, T])
    F_ph3 = np.array([float(state_at_t1[0]), float(state_at_t1[0])])
    u_ph3 = np.zeros(2)

    if T - t1 > 1e-10:
        n3 = max(2, int((T - t1) / T * n_eval))

        def _rhs_zero3(t: float, state: NDArray) -> NDArray:
            F_s  = float(max(state[0], 0.0))
            Ms_s = float(max(state[1], 0.0))
            dF   = float(recruitment(F_s, Ms_s, params, singular_eps=num_cfg.singular_eps))
            dMs  = -params.delta_s * Ms_s
            return np.array([dF, dMs], dtype=np.float64)

        t_eval_3 = np.linspace(t1, T, n3)
        sol3 = solve_ivp(
            fun=_rhs_zero3,
            t_span=(t1, T),
            y0=state_at_t1,
            method=num_cfg.solver_method,
            t_eval=t_eval_3,
            rtol=num_cfg.rtol,
            atol=num_cfg.atol,
        )
        t_ph3 = sol3.t
        F_ph3 = sol3.y[0]
        u_ph3 = np.zeros_like(sol3.t)

    # ── Concatenate phases (avoid duplicate boundary points) ─────────────
    if t0 > 1e-10 and T - t1 > 1e-10:
        t_full = np.concatenate([t_ph1, t_ph2[1:], t_ph3[1:]])
        F_full = np.concatenate([F_ph1, F_ph2[1:], F_ph3[1:]])
        u_full = np.concatenate([u_ph1, u_ph2[1:], u_ph3[1:]])
    elif t0 > 1e-10:
        t_full = np.concatenate([t_ph1, t_ph2[1:]])
        F_full = np.concatenate([F_ph1, F_ph2[1:]])
        u_full = np.concatenate([u_ph1, u_ph2[1:]])
    elif T - t1 > 1e-10:
        t_full = np.concatenate([t_ph2, t_ph3[1:]])
        F_full = np.concatenate([F_ph2, F_ph3[1:]])
        u_full = np.concatenate([u_ph2, u_ph3[1:]])
    else:
        t_full = t_ph2
        F_full = F_ph2
        u_full = u_ph2

    return t_full, F_full, u_full


def _simulate_with_tau1(
    params: BiologicalParameters,
    cfg: ControlConfig,
    tau1: float,
    simulator: Simulator,
) -> float:
    """Simulate and return F_min for the given tau_1 (legacy inner loop).

    Kept for backward compatibility. The bisection now uses
    _simulate_singular_from_Fbar + _find_tau2 directly.
    """
    num_cfg = simulator.config
    t_s, F_s, Ms_s, u_s = _simulate_singular_from_Fbar(params, cfg, tau1, num_cfg)
    state_end = np.array([float(F_s[-1]), float(Ms_s[-1])], dtype=np.float64)
    _, F_min = _find_tau2(params, cfg, tau1, state_end, num_cfg)
    return F_min
