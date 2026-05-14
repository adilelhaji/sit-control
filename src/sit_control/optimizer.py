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

    Implements the L^1-norm cost functional of equation (2.10) of
    the TFG. The L^2-norm variant is obtained by replacing the
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
            cfg.T, cfg.U_max, epsilon, N,
        )

        m = GEKKO(remote=False)
        m.time = np.linspace(0.0, cfg.T, N)

        # State variables
        F = m.Var(value=p.F_bar, lb=0.0, name="F")
        Ms = m.Var(value=0.0, lb=0.0, name="Ms")

        # Control variable
        u = m.MV(value=0.0, lb=0.0, ub=cfg.U_max, name="u")
        u.STATUS = 1
        u.DCOST = 0.0  # No penalty on control variation

        # System dynamics (reduced model S1)
        denom_E = p.beta_E * F / p.K + p.nu_E + p.delta_E
        numerator = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
        denom = denom_E * (
            (1.0 - p.nu) * p.nu_E * p.beta_E * F
            + p.delta_M * p.gamma_s * Ms * denom_E
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
        m.options.SOLVER = 1  # APOPT
        m.options.IMODE = 6   # Dynamic optimal control
        m.options.RTOL = self.num_config.rtol
        m.options.OTOL = 1e-6

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
        cost = float(np.trapz(u_arr, m.time))

        logger.info(
            "L1 problem solved in %.2fs with J(u*) = %.4e",
            wall_time, cost,
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

        Minimises the quadratic cost functional of equation (2.10b) / (J_L2)
        of the TFG::

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
        TFG eq. (J_L2) and Almeida et al. (2022), Section 5.
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
            cfg.T, cfg.U_max, epsilon, c_weight, N,
        )

        m = GEKKO(remote=False)
        m.time = np.linspace(0.0, cfg.T, N)

        # State variables
        F = m.Var(value=p.F_bar, lb=0.0, name="F")
        Ms = m.Var(value=0.0, lb=0.0, name="Ms")

        # Control variable — same admissible set as L1
        u = m.MV(value=0.0, lb=0.0, ub=cfg.U_max, name="u")
        u.STATUS = 1
        u.DCOST = 0.0

        # System dynamics — identical to S1 in solve_L1
        denom_E = p.beta_E * F / p.K + p.nu_E + p.delta_E
        numerator = p.nu * (1.0 - p.nu) * p.beta_E**2 * p.nu_E**2 * F**2
        denom = denom_E * (
            (1.0 - p.nu) * p.nu_E * p.beta_E * F
            + p.delta_M * p.gamma_s * Ms * denom_E
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

        # Solver configuration
        m.options.SOLVER = 1   # APOPT
        m.options.IMODE = 6    # Dynamic optimal control
        m.options.RTOL = self.num_config.rtol
        m.options.OTOL = 1e-6

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
        cost = float(np.trapz(0.5 * c_weight * u_arr**2, m.time))

        logger.info(
            "L2 problem solved in %.2fs with J2(u*) = %.4e",
            wall_time, cost,
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