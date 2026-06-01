"""Immutable biological and numerical parameters for the SIT model.

Parameter values follow Almeida et al. (2022), J. Differential Equations,
which in turn cites Strugarek et al. (2019) for the biological data of
*Aedes polynesiensis*.

Note on the carrying capacity K:
    K is NOT a fixed biological constant. Strugarek et al. (2019), Sec. 4.1,
    defines it as a site-specific calibration parameter, derived from the
    male population estimate M*_+ obtained by mark-release-recapture (MRR)
    field experiments on the target area. Different studies adopt different
    K values reflecting different field scenarios:

    - Almeida et al. (2022) does not publish K in Table 1, but their
      Tables 2-4 explicitly report F_bar = 11037 with nu_E = 0.05. From
      the persistence-equilibrium formula F_bar = nu*nu_E*K*(1-1/R0)/delta_F,
      this uniquely determines K ~ 18258.
    - Join, Almeida & Fliess (2026), Sec. 5.2, adopts K = 22200, citing
      Strugarek (2019).

    The default value used here is K = 18258, consistent with the numerical
    experiments of Almeida (2022) that this implementation aims to reproduce.
    Use ``BiologicalParameters.join2026()`` for the alternative scenario.

References:
    Almeida, L., Duprez, M., Privat, Y., & Vauchelet, N. (2022).
    Optimal control strategies for the sterile mosquitoes technique.
    Journal of Differential Equations, 311, 229-266.

    Strugarek, M., Bossin, H., & Dumont, Y. (2019). On the use of the
    sterile insect release technique to reduce or eliminate mosquito
    populations. Applied Mathematical Modelling, 68, 443-470.

    Join, C., Almeida, L., & Fliess, M. (2026). Sterile mosquito release
    via intelligent proportional controllers. arXiv:2604.01355.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


#: Carrying capacity consistent with Almeida et al. (2022) Tables 2-4
#: (derived from F_bar = 11037 via the persistence-equilibrium formula).
K_ALMEIDA_2022: float = 18_258.0

#: Carrying capacity cited in Join, Almeida & Fliess (2026), Sec. 5.2.
K_JOIN_2026: float = 22_200.0


@dataclass(frozen=True, slots=True)
class BiologicalParameters:
    """Biological parameters of the mosquito population model.

    All rates are expressed in day^-1. Values follow Almeida et al.
    (2022), Table 1, with the exception of K (see module docstring),
    whose default is set so the persistence equilibrium reproduces
    F_bar = 11037 reported in Almeida (2022) Tables 2-4.

    Attributes:
        beta_E:  Oviposition rate (day^-1).
        delta_E: Mortality rate of aquatic phase (day^-1).
        delta_M: Mortality rate of wild males (day^-1).
        delta_F: Mortality rate of females (day^-1).
        delta_s: Mortality rate of sterile males (day^-1).
        nu_E:    Hatching rate of eggs (day^-1).
        nu:      Probability of emerging as female (dimensionless).
        gamma_s: Competitiveness of sterile males (dimensionless).
        K:       Environmental carrying capacity (individuals). Default
            K = 18258 reproduces the F_bar = 11037 reported in
            Almeida (2022) Tables 2-4. K is a site-specific calibration
            parameter, not a fixed biological constant
            (see module docstring and Strugarek et al. 2019, Sec. 4.1).
    """

    beta_E: float = 10.0
    delta_E: float = 0.03
    delta_M: float = 0.1
    delta_F: float = 0.04
    delta_s: float = 0.12
    nu_E: float = 0.05
    nu: float = 0.49
    gamma_s: float = 1.0
    K: float = K_ALMEIDA_2022

    def __post_init__(self) -> None:
        """Validate parameter values on construction."""
        for name in ("beta_E", "delta_E", "delta_M", "delta_F",
                    "delta_s", "nu_E", "K"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0 < self.nu < 1:
            raise ValueError(f"nu must be in (0, 1), got {self.nu}")
        if not 0 < self.gamma_s <= 1:
            raise ValueError(
                f"gamma_s must be in (0, 1], got {self.gamma_s}"
            )

    @classmethod
    def almeida2022(cls, **overrides: float) -> "BiologicalParameters":
        """Preset matching the numerical experiments of Almeida et al. (2022).

        Sets K = 18258 so that F_bar = 11037 (Almeida 2022, Tables 2-4).
        This is the canonical preset for reproducing the article's results.

        Args:
            **overrides: Optional field overrides (e.g. ``nu_E=0.1``).

        Returns:
            BiologicalParameters with K = K_ALMEIDA_2022 and any overrides.
        """
        return cls(K=K_ALMEIDA_2022, **overrides)

    @classmethod
    def join2026(cls, **overrides: float) -> "BiologicalParameters":
        """Preset matching Join, Almeida & Fliess (2026), Sec. 5.2.

        Sets K = 22200. This preset reproduces a different field-calibration
        scenario than Almeida (2022); the two are NOT interchangeable for
        cost-functional comparison.

        Args:
            **overrides: Optional field overrides.

        Returns:
            BiologicalParameters with K = K_JOIN_2026 and any overrides.
        """
        return cls(K=K_JOIN_2026, **overrides)

    @property
    def R0(self) -> float:
        """Basic offspring number (entomological R0).

        Defined in Almeida et al. (2022), Hypothesis (H):
        R0 = nu * beta_E * nu_E / (delta_F * (nu_E + delta_E)).
        """
        return (self.nu * self.beta_E * self.nu_E) / (
            self.delta_F * (self.nu_E + self.delta_E)
        )

    @property
    def F_bar(self) -> float:
        """Female population at the persistence equilibrium.

        Derived from Almeida et al. (2022), Proposition 2.1, eq. (2):
        E_bar = K * (1 - 1/R0), F_bar = (nu * nu_E / delta_F) * E_bar.
        """
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
        rtol: Relative tolerance for ODE integration (RK45).
        atol: Absolute tolerance for ODE integration (RK45).
        n_collocation: Number of GEKKO collocation points.
        singular_eps: Threshold for the removable singularity of f.
        solver_method: SciPy solve_ivp method.
        apopt_rtol: Relative tolerance for the APOPT NLP solver inside
            GEKKO. Kept separate from ``rtol`` because APOPT and RK45
            operate on different scales; Almeida et al. (2022) use 1e-6.
    """

    rtol: float = 1e-8
    atol: float = 1e-8
    n_collocation: int = 300
    singular_eps: float = 1e-12
    solver_method: str = "RK45"
    apopt_rtol: float = 1e-6
    gekko_solver: int = 1  # 1=APOPT, 3=IPOPT
    #: Maximum internal integration step for solve_ivp. Defaults to inf
    #: (unbounded, SciPy default). Set it to <= pulse_width/2 for
    #: discontinuous controls (impulsive/periodic releases) so the adaptive
    #: RK45 step cannot skip a narrow pulse; see Simulator.simulate.
    max_step: float = float("inf")


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