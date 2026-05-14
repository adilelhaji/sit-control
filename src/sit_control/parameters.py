"""Immutable biological and numerical parameters for the SIT model.

Parameter values follow Almeida et al. (2022), J. Differential Equations,
which in turn cites Strugarek et al. (2019) for the biological data of
*Aedes polynesiensis*.

References:
    Almeida, L., Duprez, M., Privat, Y., & Vauchelet, N. (2022).
    Optimal control strategies for the sterile mosquitoes technique.
    Journal of Differential Equations, 311, 229-266.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class BiologicalParameters:
    """Biological parameters of the mosquito population model.

    All rates are expressed in day^-1. Values are taken from
    Almeida et al. (2022), Table 1.

    Attributes:
        beta_E: Oviposition rate.
        delta_E: Mortality rate of aquatic phase (eggs).
        delta_M: Mortality rate of wild males.
        delta_F: Mortality rate of females.
        delta_s: Mortality rate of sterile males.
        nu_E: Hatching rate of eggs.
        nu: Probability of emerging as female.
        gamma_s: Competitiveness of sterile males.
        K: Environmental carrying capacity (individuals).
    """

    beta_E: float = 10.0
    delta_E: float = 0.03
    delta_M: float = 0.1
    delta_F: float = 0.04
    delta_s: float = 0.12
    nu_E: float = 0.05
    nu: float = 0.49
    gamma_s: float = 1.0
    K: float = 22_200.0

    def __post_init__(self) -> None:
        """Validate parameter values on construction."""
        for name in ("beta_E", "delta_E", "delta_M", "delta_F",
                    "delta_s", "nu_E", "gamma_s", "K"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0 < self.nu < 1:
            raise ValueError(f"nu must be in (0, 1), got {self.nu}")

    @property
    def R0(self) -> float:
        """Basic offspring number (entomological R0)."""
        return (self.nu * self.beta_E * self.nu_E) / (
            self.delta_F * (self.nu_E + self.delta_E)
        )

    @property
    def F_bar(self) -> float:
        """Female population at persistence equilibrium."""
        E_bar = self.K * (1.0 - 1.0 / self.R0)
        return (self.nu * self.nu_E / self.delta_F) * E_bar


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Configuration of the optimal control problem.

    Attributes:
        T: Time horizon in days.
        U_max: Maximum release rate (sterile males per day).
        epsilon: Suppression threshold (females). If None, computed
            as F_bar / 4 following Almeida et al. (2022).
    """

    T: float
    U_max: float
    epsilon: float | None = None

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.U_max <= 0:
            raise ValueError(f"U_max must be positive, got {self.U_max}")
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")


@dataclass(frozen=True, slots=True)
class NumericalConfig:
    """Numerical configuration for solvers.

    Attributes:
        rtol: Relative tolerance for ODE integration.
        atol: Absolute tolerance for ODE integration.
        n_collocation: Number of GEKKO collocation points.
        singular_eps: Threshold for the removable singularity of f.
        solver_method: SciPy solve_ivp method.
    """

    rtol: float = 1e-8
    atol: float = 1e-8
    n_collocation: int = 300
    singular_eps: float = 1e-12
    solver_method: str = "RK45"


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)