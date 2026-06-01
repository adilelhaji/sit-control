"""Forward-Backward Sweep (FBS) contrast solver for the L1 SIT problem.

This is the classic bang-bang Forward-Backward Sweep (Lenhart & Workman,
2007), used here as an *independent qualitative contrast method*:
  1. Forward sweep  : integrate (F, Ms) on [0, T] with current u(t)
  2. Backward sweep : integrate (λ₁, λ₂) on [T, 0] with terminal values
  3. Control update : u_new = U_max  if σ = 1 + λ₂ < 0, else 0  (bang-bang)
  4. Blend          : u ← α·u_new + (1-α)·u  (dampens bang-bang chattering)
  5. Repeat until ||Δu||_∞ < tol

IMPORTANT — approximate method, NOT the authoritative solver. The control
update is purely bang-bang and has no singular branch, so it CANNOT
reproduce the bang-singular-bang structure of the L1 optimum. For horizons
in the singular regime (T ≥ T* ≈ 104 d in the reference case) the control
chatters in the singular arc, the sweep does NOT converge, and the returned
cost J is method-dependent (it varies with α and overestimates the
optimum). The authoritative optimal solver is
``bisection.solve_by_bisection``, which builds the exact
bang-singular-bang solution. Use this module only for qualitative
cross-checking, never as the source of the reported optimal cost.

The terminal constraint F(T) = ε is imposed through a *linear signed*
multiplier on λ₁(T):  λ₁(T) = M · (F(T) − ε)  (no max(0,·); the sign drives
F up or down toward the active equality). The PMP sign condition
λ₁(T) ≥ 0 therefore holds only at the fixed point F(T) = ε, not during
intermediate iterations.

References:
    Lenhart, S. & Workman, J. T. (2007). Optimal Control Applied to
    Biological Models. Chapman & Hall/CRC.

    Almeida, L. et al. (2022). Optimal control strategies for the sterile
    mosquitoes technique. J. Differential Equations, 311, 229-266.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .parameters import BiologicalParameters, ControlConfig, NumericalConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FBSweepResult:
    """Result of a Forward-Backward Sweep run.

    Attributes:
        t: Time grid.
        u_opt: Optimal control profile.
        F_opt: Female trajectory.
        Ms_opt: Sterile male trajectory.
        lambda1: Adjoint variable λ₁(t).
        lambda2: Adjoint variable λ₂(t) (switching: σ = 1 + λ₂).
        cost: J(u*) = ∫u dt evaluated by trapezoidal rule.
        wall_time: Wall clock time (s).
        converged: Whether the inner FBS loop converged.
        n_iterations: Total inner iterations performed.
        F_terminal: F(T) at the end of the sweep.
    """

    t: NDArray[np.float64]
    u_opt: NDArray[np.float64]
    F_opt: NDArray[np.float64]
    Ms_opt: NDArray[np.float64]
    lambda1: NDArray[np.float64]
    lambda2: NDArray[np.float64]
    cost: float
    wall_time: float
    converged: bool
    n_iterations: int
    F_terminal: float


class ForwardBackwardSweep:
    """Approximate bang-bang PMP solver (contrast method) for L1 SIT control.

    Classic Forward-Backward Sweep (Lenhart & Workman, 2007). It does NOT
    capture the singular arc of the L1 optimum and does not converge in the
    singular regime (see the module docstring); it is kept only as an
    independent qualitative cross-check. The authoritative optimal solver is
    ``bisection.solve_by_bisection``.
    """

    def __init__(
        self,
        params: BiologicalParameters,
        num_config: NumericalConfig | None = None,
    ) -> None:
        self.params = params
        self.num_config = num_config or NumericalConfig()

    # ── S1 dynamics and partial derivatives ──────────────────────────────────

    def _f1(self, F: float, Ms: float) -> float:
        """Growth rate f₁(F, Ms) of the reduced model S1."""
        p = self.params
        F = max(F, 0.0)
        Ms = max(Ms, 0.0)
        DE = p.beta_E * F / p.K + p.nu_E + p.delta_E
        num = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
        den = (
            DE * ((1.0 - p.nu) * p.nu_E * p.beta_E * F
                  + p.delta_M * p.gamma_s * Ms * DE)
            + self.num_config.singular_eps
        )
        return num / den - p.delta_F * F

    def _df1_dF(self, F: float, Ms: float) -> float:
        """∂f₁/∂F via central finite differences."""
        h = max(1e-4 * abs(F), 1.0)
        return (self._f1(F + h, Ms) - self._f1(F - h, Ms)) / (2.0 * h)

    def _df1_dMs(self, F: float, Ms: float) -> float:
        """∂f₁/∂Ms via central finite differences."""
        h = max(1e-4 * abs(Ms), 1.0)
        return (self._f1(F, Ms + h) - self._f1(F, Ms - h)) / (2.0 * h)

    # ── Forward and backward ODE passes ──────────────────────────────────────

    def _forward(
        self,
        t_grid: NDArray,
        u_grid: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Integrate state equations forward from (F_bar, 0)."""
        p = self.params
        nc = self.num_config
        u_func = interp1d(
            t_grid, u_grid, kind="linear",
            bounds_error=False, fill_value=(u_grid[0], u_grid[-1]),
        )

        def rhs(t: float, x: list) -> list:
            F, Ms = max(x[0], 0.0), max(x[1], 0.0)
            return [self._f1(F, Ms), u_func(t) - p.delta_s * Ms]

        sol = solve_ivp(
            rhs, [t_grid[0], t_grid[-1]], [p.F_bar, 0.0],
            t_eval=t_grid, method=nc.solver_method,
            rtol=nc.rtol, atol=nc.atol,
        )
        return sol.y[0], sol.y[1]

    def _backward(
        self,
        t_grid: NDArray,
        F_grid: NDArray,
        Ms_grid: NDArray,
        mu: float,
    ) -> tuple[NDArray, NDArray]:
        """Integrate adjoint equations backward from (μ, 0) at T.

        Adjoint system (forward in reversed time s = T − t):
            dλ₁/ds = +λ₁ · ∂f₁/∂F
            dλ₂/ds = +λ₁ · ∂f₁/∂Ms − λ₂ · δs
        """
        p = self.params
        nc = self.num_config
        T = t_grid[-1]

        F_func = interp1d(
            t_grid, F_grid, kind="linear",
            bounds_error=False, fill_value=(F_grid[0], F_grid[-1]),
        )
        Ms_func = interp1d(
            t_grid, Ms_grid, kind="linear",
            bounds_error=False, fill_value=(Ms_grid[0], Ms_grid[-1]),
        )

        def rhs_rev(s: float, lam: list) -> list:
            t = T - s
            F = max(F_func(t), 1e-10)
            Ms = max(Ms_func(t), 0.0)
            lam1, lam2 = lam
            dl1 = lam1 * self._df1_dF(F, Ms)
            dl2 = lam1 * self._df1_dMs(F, Ms) - lam2 * p.delta_s
            return [dl1, dl2]

        s_eval = T - t_grid[::-1]  # s = T - t, increasing
        sol = solve_ivp(
            rhs_rev, [0.0, T], [mu, 0.0],
            t_eval=s_eval, method=nc.solver_method,
            rtol=nc.rtol, atol=nc.atol,
        )
        # Map back to t order
        lam1 = sol.y[0][::-1]
        lam2 = sol.y[1][::-1]
        return lam1, lam2

    # ── Main solver ───────────────────────────────────────────────────────────

    def solve_L1(
        self,
        control_config: ControlConfig,
        u_init: tuple[NDArray, NDArray] | None = None,
        n_iter: int = 400,
        alpha: float = 0.3,
        tol: float = 1e-4,
        M_penalty: float = 1e4,
        n_grid: int = 500,
    ) -> FBSweepResult:
        """Solve the L1 optimal control problem via Forward-Backward Sweep.

        Args:
            control_config: Problem configuration (T, U_max, epsilon).
            u_init: Optional warm-start (t_array, u_array). If None, uses a
                ramp initialisation matching the bang-singular structure.
            n_iter: Maximum number of FBS iterations.
            alpha: Blending weight for control update (0 < alpha ≤ 1).
                Smaller values dampen chattering but slow convergence.
            tol: Convergence criterion: max||u_new − u||.
            M_penalty: Penalty weight for the terminal constraint F(T) = ε.
                Sets the signed multiplier λ₁(T) = M·(F(T)−ε) (no max(0,·)).
            n_grid: Number of time points for the FBS discretisation.

        Returns:
            FBSweepResult with the (approximate) trajectories and diagnostics.
            ``converged`` is typically False in the singular regime; the cost
            is method-dependent and is not the optimal cost (use bisection).

        Raises:
            ValueError: If n_iter < 1.
        """
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        p = self.params
        cfg = control_config
        epsilon = cfg.epsilon if cfg.epsilon is not None else p.F_bar / 4.0

        logger.info(
            "FBS: T=%g, U_max=%g, epsilon=%g, n_iter=%d, alpha=%g",
            cfg.T, cfg.U_max, epsilon, n_iter, alpha,
        )

        t = np.linspace(0.0, cfg.T, n_grid)

        # Initialise control
        if u_init is not None:
            t0, u0 = u_init
            u = np.clip(np.interp(t, t0, u0), 0.0, cfg.U_max)
            logger.info("FBS: warm-started from provided control profile.")
        else:
            u = np.where(
                t < cfg.T / 2, 0.0,
                cfg.U_max / 2 * (t - cfg.T / 2) / (cfg.T / 2),
            )

        t_start = time.perf_counter()
        converged = False
        lam1 = np.zeros(n_grid)
        lam2 = np.zeros(n_grid)

        for k in range(n_iter):
            # Forward pass
            F, Ms = self._forward(t, u)

            # Two-sided penalty: λ₁(T) = M·(F(T)−ε).
            # Signed version (not max(0,...)) because at the optimum the
            # constraint is active F(T)=ε, so we solve the equality problem.
            # Positive μ → push F down; negative μ → push F up.
            mu = M_penalty * (float(F[-1]) - epsilon)

            # Backward pass
            lam1, lam2 = self._backward(t, F, Ms, mu)

            # Control update: minimise H = u·(1+λ₂) + ... over u ∈ [0, U_max]
            sigma = 1.0 + lam2  # switching function
            u_bang = np.where(sigma < 0.0, cfg.U_max, 0.0)
            u_new = alpha * u_bang + (1.0 - alpha) * u
            u_new = np.clip(u_new, 0.0, cfg.U_max)

            diff = float(np.max(np.abs(u_new - u)))
            u = u_new

            if k % 50 == 0:
                logger.debug(
                    "FBS iter %d: ||Δu||=%.2e, F(T)=%.2f, mu=%.2e",
                    k, diff, float(F[-1]), mu,
                )

            if diff < tol:
                converged = True
                logger.info("FBS converged at iteration %d (||Δu||=%.2e)", k, diff)
                break

        # Final forward pass with converged u
        F, Ms = self._forward(t, u)
        wall_time = time.perf_counter() - t_start
        cost = float(np.trapezoid(u, t))

        logger.info(
            "FBS finished in %.2fs, %d iters, J=%.4e, F(T)=%.2f (ε=%.2f), converged=%s",
            wall_time, k + 1, cost, float(F[-1]), epsilon, converged,
        )

        return FBSweepResult(
            t=t,
            u_opt=u,
            F_opt=F,
            Ms_opt=Ms,
            lambda1=lam1,
            lambda2=lam2,
            cost=cost,
            wall_time=wall_time,
            converged=converged,
            n_iterations=k + 1,
            F_terminal=float(F[-1]),
        )
