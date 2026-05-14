"""Numerical integration of the SIT population models.

Wraps SciPy's solve_ivp with the project's parameter and configuration
objects. Provides a clean separation between the integration step and
the choice of control law.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .model import rhs_full, rhs_reduced
from .parameters import BiologicalParameters, NumericalConfig

logger = logging.getLogger(__name__)

ControlLaw = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Container for the result of an ODE integration.

    Attributes:
        t: Array of time points.
        state: Array of shape (n_states, n_points) with state trajectories.
        control: Array of control values at each time point.
        model: Model used, 'S1' (reduced) or 'S2' (full).
        success: Whether the integration converged.
    """

    t: NDArray[np.float64]
    state: NDArray[np.float64]
    control: NDArray[np.float64]
    model: Literal["S1", "S2"]
    success: bool


class Simulator:
    """Numerical simulator for the SIT models.

    Decouples integration from the choice of control law, enabling
    direct comparison of different release strategies under identical
    numerical conditions.
    """

    def __init__(
        self,
        params: BiologicalParameters,
        config: NumericalConfig | None = None,
    ) -> None:
        """Initialise the simulator.

        Args:
            params: Biological parameters of the model.
            config: Numerical configuration; defaults to NumericalConfig().
        """
        self.params = params
        self.config = config or NumericalConfig()

    def simulate(
        self,
        T: float,
        u_func: ControlLaw,
        model: Literal["S1", "S2"] = "S1",
        initial_state: NDArray[np.float64] | None = None,
        n_eval: int = 1000,
    ) -> SimulationResult:
        """Integrate the chosen model over [0, T] under control u_func.

        Args:
            T: Time horizon.
            u_func: Control law, a callable t -> u(t).
            model: 'S1' for the reduced model, 'S2' for the full one.
            initial_state: Initial condition; defaults to the persistence
                equilibrium with M_s = 0.
            n_eval: Number of evaluation points for output.

        Returns:
            SimulationResult with the integrated trajectories.

        Raises:
            ValueError: If the model identifier is invalid.
            RuntimeError: If the integrator fails to converge.
        """
        if model not in ("S1", "S2"):
            raise ValueError(f"Unknown model '{model}'; use 'S1' or 'S2'.")

        if initial_state is None:
            initial_state = self._default_initial_state(model)

        rhs = rhs_reduced if model == "S1" else rhs_full
        rhs_args = (
            (u_func, self.params, self.config.singular_eps)
            if model == "S1"
            else (u_func, self.params)
        )

        t_eval = np.linspace(0.0, T, n_eval)

        logger.debug(
            "Integrating %s on [0, %g] with %d evaluation points",
            model, T, n_eval,
        )

        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, T),
            y0=initial_state,
            method=self.config.solver_method,
            t_eval=t_eval,
            args=rhs_args,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        control_values = np.array([u_func(t) for t in sol.t])

        return SimulationResult(
            t=sol.t,
            state=sol.y,
            control=control_values,
            model=model,
            success=sol.success,
        )

    def _default_initial_state(
        self,
        model: Literal["S1", "S2"],
    ) -> NDArray[np.float64]:
        """Return the persistence equilibrium with M_s = 0."""
        p = self.params
        F_bar = p.F_bar

        if model == "S1":
            return np.array([F_bar, 0.0], dtype=np.float64)

        E_bar = p.K * (1.0 - 1.0 / p.R0)
        M_bar = (1.0 - p.nu) * p.nu_E * E_bar / p.delta_M
        return np.array([E_bar, M_bar, F_bar, 0.0], dtype=np.float64)