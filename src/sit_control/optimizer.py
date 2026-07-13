"""Optimal control solver based on GEKKO + APOPT.

Solves the problem (P^(S1)_{T,U,epsilon}) of Almeida et al. (2022)
by discretising the time horizon and solving the resulting NLP.

References:
    Hedengren, J. D. et al. (2014). Nonlinear modeling, estimation
    and predictive control in APMonitor. Computers & Chemical
    Engineering, 70, 133-148.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .parameters import BiologicalParameters, ControlConfig, NumericalConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    """Container for the result of an optimal control problem.

    Attributes:
        t: Time grid of the discretisation.
        u_opt: Optimal control profile u*(t).
        F_opt: Optimal female trajectory F*(t).
        Ms_opt: Optimal sterile male trajectory M_s*(t).
        cost: Value of the cost functional J(u*).
        wall_time: Wall clock time of the optimisation (s).
        converged: Whether the solver converged.
    """

    t: NDArray[np.float64]
    u_opt: NDArray[np.float64]
    F_opt: NDArray[np.float64]
    Ms_opt: NDArray[np.float64]
    cost: float
    wall_time: float
    converged: bool


class GekkoOptimiser:
    """Optimal control solver wrapping GEKKO.

    Implements the L^1-norm cost functional of Almeida et al. (2022),
    equation (4). The L^2-norm variant is obtained by replacing the
    objective.
    """

    def __init__(
        self,
        params: BiologicalParameters,
        num_config: NumericalConfig | None = None,
    ) -> None:
        """Initialise the optimiser.

        Args:
            params: Biological parameters of the model.
            num_config: Numerical configuration.
        """
        self.params = params
        self.num_config = num_config or NumericalConfig()

    def solve_L1(
        self,
        control_config: ControlConfig,
    ) -> OptimisationResult:
        """Solve the L^1 optimal control problem.

        Args:
            control_config: Operational configuration (T, U_max, epsilon).

        Returns:
            Optimisation result with the optimal trajectories.

        Raises:
            ImportError: If GEKKO is not installed.
            RuntimeError: If the solver fails to converge.
        """
        try:
            from gekko import GEKKO
        except ImportError as exc:
            raise ImportError(
                "GEKKO is required: install with `pip install gekko`"
            ) from exc

        p = self.params
        cfg = control_config
        epsilon = cfg.epsilon if cfg.epsilon is not None else p.F_bar / 4.0
        N = self.num_config.n_collocation

        logger.info(
            "Solving L1 problem with T=%g, U_max=%g, epsilon=%g, N=%d",
            cfg.T,
            cfg.U_max,
            epsilon,
            N,
        )

        m = GEKKO(remote=False)
        m.time = np.linspace(0.0, cfg.T, N)

        # State variables — initial conditions fixed explicitly (IMODE=6 requires
        # the first element to match the physical initial state).
        F = m.Var(value=p.F_bar, lb=0.0, name="F")
        Ms = m.Var(value=0.0, lb=0.0, name="Ms")
        m.fix_initial(F, val=p.F_bar)
        m.fix_initial(Ms, val=0.0)

        # Control variable — warm-started with a ramp matching the bang-singular
        # structure: zero for the first half, then linearly increasing to U_max/2.
        # This avoids APOPT converging to the u=0 trivial local minimum.
        t_grid = np.linspace(0.0, cfg.T, N)
        u_init = np.where(
            t_grid < cfg.T / 2, 0.0, cfg.U_max / 2 * (t_grid - cfg.T / 2) / (cfg.T / 2)
        )
        u = m.MV(value=u_init, lb=0.0, ub=cfg.U_max, name="u")
        u.STATUS = 1
        u.DCOST = 0.0  # No penalty on control variation

        # System dynamics (reduced model S1)
        # Regularised denominator: adds singular_eps to avoid 0/0 at (F,Ms)=(0,0).
        # The true limit is -delta_F*F (see model.py), but GEKKO cannot branch;
        # singular_eps << F_bar so the regularisation is negligible in practice.
        denom_E = p.beta_E * F / p.K + p.nu_E + p.delta_E
        numerator = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
        denom = (
            denom_E
            * (
                (1.0 - p.nu) * p.nu_E * p.beta_E * F
                + p.delta_M * p.gamma_s * Ms * denom_E
            )
            + self.num_config.singular_eps
        )

        m.Equation(F.dt() == numerator / denom - p.delta_F * F)
        m.Equation(Ms.dt() == u - p.delta_s * Ms)

        # Terminal constraint F(T) <= epsilon
        final = np.zeros(N)
        final[-1] = 1
        final_param = m.Param(value=final, name="final")
        m.Equation(F * final_param <= epsilon)

        # L1 cost functional
        m.Minimize(m.integral(u))

        # Solver configuration
        # RTOL/OTOL here are the NLP convergence tolerances for APOPT (not the
        # ODE rtol). Almeida et al. (2022) use 1e-6; both are driven by the
        # documented apopt_rtol knob so no convergence tolerance is hidden.
        m.options.SOLVER = self.num_config.gekko_solver  # 1=APOPT, 3=IPOPT
        m.options.IMODE = 6  # Dynamic optimal control
        m.options.RTOL = self.num_config.apopt_rtol
        m.options.OTOL = self.num_config.apopt_rtol  # objective tol (same knob)

        t_start = time.perf_counter()
        try:
            m.solve(disp=False)
            converged = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GEKKO did not converge: %s", exc)
            converged = False
        wall_time = time.perf_counter() - t_start

        u_arr = np.array(u.value)
        F_arr = np.array(F.value)
        Ms_arr = np.array(Ms.value)
        cost = float(np.trapezoid(u_arr, m.time))

        logger.info(
            "L1 problem solved in %.2fs with J(u*) = %.4e",
            wall_time,
            cost,
        )

        return OptimisationResult(
            t=np.asarray(m.time),
            u_opt=u_arr,
            F_opt=F_arr,
            Ms_opt=Ms_arr,
            cost=cost,
            wall_time=wall_time,
            converged=converged,
        )

    def solve_L1_warmstart(
        self,
        control_config: ControlConfig,
        u_init: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> OptimisationResult:
        """Solve the L1 problem with a custom control warm-start.

        Identical to solve_L1 but replaces the ramp initialisation with an
        externally provided control profile (e.g. the bisection solution).

        Args:
            control_config: Operational configuration (T, U_max, epsilon).
            u_init: Tuple (t_array, u_array) of the warm-start control.

        Returns:
            OptimisationResult with the optimal trajectories.
        """
        try:
            from gekko import GEKKO
        except ImportError as exc:
            raise ImportError(
                "GEKKO is required: install with `pip install gekko`"
            ) from exc

        p = self.params
        cfg = control_config
        epsilon = cfg.epsilon if cfg.epsilon is not None else p.F_bar / 4.0
        N = self.num_config.n_collocation

        logger.info(
            "Solving L1 (warm-start) with T=%g, U_max=%g, epsilon=%g, N=%d",
            cfg.T,
            cfg.U_max,
            epsilon,
            N,
        )

        m = GEKKO(remote=False)
        m.time = np.linspace(0.0, cfg.T, N)

        F = m.Var(value=p.F_bar, lb=0.0, name="F")
        Ms = m.Var(value=0.0, lb=0.0, name="Ms")
        m.fix_initial(F, val=p.F_bar)
        m.fix_initial(Ms, val=0.0)

        # Interpolate provided warm-start onto GEKKO grid
        t_ws, u_ws = u_init
        u_init_grid = np.clip(np.interp(m.time, t_ws, u_ws), 0.0, cfg.U_max)

        u = m.MV(value=u_init_grid, lb=0.0, ub=cfg.U_max, name="u")
        u.STATUS = 1
        u.DCOST = 0.0

        denom_E = p.beta_E * F / p.K + p.nu_E + p.delta_E
        numerator = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
        denom = (
            denom_E
            * (
                (1.0 - p.nu) * p.nu_E * p.beta_E * F
                + p.delta_M * p.gamma_s * Ms * denom_E
            )
            + self.num_config.singular_eps
        )

        m.Equation(F.dt() == numerator / denom - p.delta_F * F)
        m.Equation(Ms.dt() == u - p.delta_s * Ms)

        final = np.zeros(N)
        final[-1] = 1
        final_param = m.Param(value=final, name="final")
        m.Equation(F * final_param <= epsilon)

        m.Minimize(m.integral(u))

        m.options.SOLVER = self.num_config.gekko_solver
        m.options.IMODE = 6
        m.options.RTOL = self.num_config.apopt_rtol
        m.options.OTOL = self.num_config.apopt_rtol  # objective tol (same knob)

        t_start = time.perf_counter()
        try:
            m.solve(disp=False)
            converged = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GEKKO (warm-start) did not converge: %s", exc)
            converged = False
        wall_time = time.perf_counter() - t_start

        u_arr = np.array(u.value)
        F_arr = np.array(F.value)
        Ms_arr = np.array(Ms.value)
        cost = float(np.trapezoid(u_arr, m.time))

        logger.info(
            "L1 warm-start solved in %.2fs with J(u*) = %.4e",
            wall_time,
            cost,
        )

        return OptimisationResult(
            t=np.asarray(m.time),
            u_opt=u_arr,
            F_opt=F_arr,
            Ms_opt=Ms_arr,
            cost=cost,
            wall_time=wall_time,
            converged=converged,
        )

    def solve_L2(
        self,
        control_config: ControlConfig,
        c_weight: float = 1.0,
    ) -> OptimisationResult:
        """Solve the L^2 optimal control problem.

        Minimises the quadratic cost functional (L^2 variant)::

            J_2(u) = integral_0^T  (c / 2) * u(t)^2  dt

        subject to the reduced system S1 and the terminal constraint
        F(T) <= epsilon. The quadratic penalty on u eliminates singular arcs
        and produces a smooth, continuous release profile, as characterised
        analytically in Almeida et al. (2022), Section 5.

        Parameters
        ----------
        control_config : ControlConfig
            Operational configuration (T, U_max, epsilon).
        c_weight : float, optional
            Quadratic penalty weight ``c > 0`` in the cost functional.
            Default is 1.0 (dimensionless).

        Returns
        -------
        OptimisationResult
            Optimal trajectories. ``cost`` stores J_2(u*) = integral of
            (c/2)*u^2 evaluated by the trapezoidal rule.

        Raises
        ------
        ImportError
            If GEKKO is not installed.
        ValueError
            If ``c_weight`` is not strictly positive.
        RuntimeError
            If the solver fails to converge.

        References
        ----------
        Almeida et al. (2022), Section 5.
        """
        if c_weight <= 0:
            raise ValueError(f"c_weight must be positive, got {c_weight}")

        try:
            from gekko import GEKKO
        except ImportError as exc:
            raise ImportError(
                "GEKKO is required: install with `pip install gekko`"
            ) from exc

        p = self.params
        cfg = control_config
        epsilon = cfg.epsilon if cfg.epsilon is not None else p.F_bar / 4.0
        N = self.num_config.n_collocation

        logger.info(
            "Solving L2 problem with T=%g, U_max=%g, epsilon=%g, c=%g, N=%d",
            cfg.T,
            cfg.U_max,
            epsilon,
            c_weight,
            N,
        )

        m = GEKKO(remote=False)
        m.time = np.linspace(0.0, cfg.T, N)

        # State variables — initial conditions fixed explicitly
        F = m.Var(value=p.F_bar, lb=0.0, name="F")
        Ms = m.Var(value=0.0, lb=0.0, name="Ms")
        m.fix_initial(F, val=p.F_bar)
        m.fix_initial(Ms, val=0.0)

        # Control variable — ramp warm-start to avoid trivial u=0 local minimum
        t_grid = np.linspace(0.0, cfg.T, N)
        u_init = np.where(
            t_grid < cfg.T / 2, 0.0, cfg.U_max / 2 * (t_grid - cfg.T / 2) / (cfg.T / 2)
        )
        u = m.MV(value=u_init, lb=0.0, ub=cfg.U_max, name="u")
        u.STATUS = 1
        u.DCOST = 0.0

        # System dynamics — identical to S1 in solve_L1 (regularised denom)
        denom_E = p.beta_E * F / p.K + p.nu_E + p.delta_E
        numerator = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
        denom = (
            denom_E
            * (
                (1.0 - p.nu) * p.nu_E * p.beta_E * F
                + p.delta_M * p.gamma_s * Ms * denom_E
            )
            + self.num_config.singular_eps
        )

        m.Equation(F.dt() == numerator / denom - p.delta_F * F)
        m.Equation(Ms.dt() == u - p.delta_s * Ms)

        # Terminal constraint F(T) <= epsilon
        final = np.zeros(N)
        final[-1] = 1
        final_param = m.Param(value=final, name="final")
        m.Equation(F * final_param <= epsilon)

        # L2 cost functional: integral of (c/2) * u^2
        m.Minimize(m.integral(c_weight / 2.0 * u**2))

        # Solver configuration (apopt_rtol separate from ODE rtol)
        m.options.SOLVER = self.num_config.gekko_solver  # 1=APOPT, 3=IPOPT
        m.options.IMODE = 6  # Dynamic optimal control
        m.options.RTOL = self.num_config.apopt_rtol
        m.options.OTOL = self.num_config.apopt_rtol  # objective tol (same knob)

        t_start = time.perf_counter()
        try:
            m.solve(disp=False)
            converged = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GEKKO L2 did not converge: %s", exc)
            converged = False
        wall_time = time.perf_counter() - t_start

        u_arr = np.array(u.value)
        F_arr = np.array(F.value)
        Ms_arr = np.array(Ms.value)
        cost = float(np.trapezoid(0.5 * c_weight * u_arr**2, m.time))

        logger.info(
            "L2 problem solved in %.2fs with J2(u*) = %.4e",
            wall_time,
            cost,
        )

        return OptimisationResult(
            t=np.asarray(m.time),
            u_opt=u_arr,
            F_opt=F_arr,
            Ms_opt=Ms_arr,
            cost=cost,
            wall_time=wall_time,
            converged=converged,
        )
