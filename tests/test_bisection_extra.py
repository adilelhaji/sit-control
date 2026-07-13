"""Additional tests for the bisection solver's uncovered paths.

This file complements ``tests/test_bisection.py``. It targets the parts of
:mod:`sit_control.bisection` that the existing suite leaves uncovered:

* the ``D < 1e-30`` degenerate-denominator guard inside ``_compute_partials``;
* the inner singular-arc simulation ``_simulate_singular_from_Fbar``;
* the formula-(9) trajectory assembler ``build_formula9_trajectory`` and each of
  its four phase-concatenation branches (interior arc, right-flush arc,
  left-flush arc, and full-span arc);
* the full bisection SOLVE loop ``solve_by_bisection`` (both the converged and
  the exhausted-iterations return paths).

Everything uses SciPy ``solve_ivp`` only (NO GEKKO). The helper/trajectory tests
run on short time spans and stay fast (unmarked, so they contribute coverage to
CI's ``-m "not slow"`` run). Only the full-solve tests are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.bisection import (
    BisectionResult,
    _compute_partials,
    _simulate_singular_from_Fbar,
    build_formula9_trajectory,
    solve_by_bisection,
)
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
)


@pytest.fixture
def params() -> BiologicalParameters:
    """Default parameters from Almeida et al. (2022)."""
    return BiologicalParameters()


@pytest.fixture
def num_cfg() -> NumericalConfig:
    """Default numerical configuration (RK45, tight tolerances)."""
    return NumericalConfig()


# ---------------------------------------------------------------------------
# _compute_partials — degenerate denominator guard (line 133-134)
# ---------------------------------------------------------------------------


def test_compute_partials_degenerate_denominator_guard() -> None:
    """When the denominator D collapses below 1e-30, partials degrade safely.

    The guard must return ``(-delta_F, 0, 0, 0)`` rather than dividing by a
    near-zero denominator. It is reached with a contrived parameter set whose
    tiny hatching/oviposition rates shrink D far below the threshold; the point
    is to pin the safe-degradation contract, not a realistic scenario.
    """
    tiny = BiologicalParameters(beta_E=1e-6, nu_E=1e-18)
    df_dF, df_dMs, d2f_dMs2, d2f_dMsdF = _compute_partials(1e-12, 0.0, tiny)
    assert df_dF == pytest.approx(-tiny.delta_F)
    assert df_dMs == 0.0
    assert d2f_dMs2 == 0.0
    assert d2f_dMsdF == 0.0


# ---------------------------------------------------------------------------
# _simulate_singular_from_Fbar
# ---------------------------------------------------------------------------


def test_simulate_singular_tau1_zero_single_point(
    params: BiologicalParameters,
    num_cfg: NumericalConfig,
) -> None:
    """tau1=0 returns a single sample at (F_bar, 0) without integrating."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    t_s, F_s, Ms_s, u_s = _simulate_singular_from_Fbar(params, cfg, 0.0, num_cfg)
    assert t_s.shape == (1,)
    assert F_s[0] == pytest.approx(params.F_bar)
    assert Ms_s[0] == 0.0
    assert u_s[0] >= 0.0


def test_simulate_singular_short_arc_structure(
    params: BiologicalParameters,
    num_cfg: NumericalConfig,
) -> None:
    """A short singular arc integrates from (F_bar, 0) with releases building up.

    F starts at F_bar; sterile males start at 0 and must grow (releases are
    non-negative), and the control stays within [0, U_max].
    """
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    t_s, F_s, Ms_s, u_s = _simulate_singular_from_Fbar(
        params, cfg, tau1=3.0, num_cfg=num_cfg
    )
    assert t_s[0] == pytest.approx(0.0)
    assert t_s[-1] == pytest.approx(3.0)
    assert F_s[0] == pytest.approx(params.F_bar)
    assert Ms_s[0] == pytest.approx(0.0)
    assert Ms_s[-1] >= 0.0
    assert np.all(u_s >= 0.0)
    assert np.all(u_s <= cfg.U_max + 1e-9)


# ---------------------------------------------------------------------------
# build_formula9_trajectory — the four phase-concatenation branches
# ---------------------------------------------------------------------------


def _assert_valid_trajectory(
    t: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    cfg: ControlConfig,
) -> None:
    """Shared structural checks on a formula-(9) trajectory."""
    assert t.shape == F.shape == u.shape
    assert t.ndim == 1 and t.size >= 2
    # Monotone non-decreasing time grid spanning [0, T].
    assert np.all(np.diff(t) >= -1e-9)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(cfg.T)
    # Control admissible everywhere.
    assert np.all(u >= -1e-9)
    assert np.all(u <= cfg.U_max + 1e-6)


def test_build_trajectory_interior_arc(
    params: BiologicalParameters,
    num_cfg: NumericalConfig,
) -> None:
    """Full three-phase case: 0 < t0 < t1 < T (interior singular arc)."""
    cfg = ControlConfig(T=10.0, U_max=5000.0)
    bis = BisectionResult(
        tau1=3.0, tau2=8.0, t0=2.0, t1=5.0, F_min=1000.0, iterations=1, converged=True
    )
    t, F, u = build_formula9_trajectory(params, cfg, bis, num_cfg, n_eval=200)
    _assert_valid_trajectory(t, F, u, cfg)
    # Phase 1 (u=0 on [0, t0]): F is unchanged at F_bar until t0.
    assert F[0] == pytest.approx(params.F_bar)
    assert u[0] == pytest.approx(0.0)


def test_build_trajectory_right_flush_arc(
    params: BiologicalParameters,
    num_cfg: NumericalConfig,
) -> None:
    """t1 == T: the terminal u=0 phase is absent (t0>0 branch)."""
    cfg = ControlConfig(T=10.0, U_max=5000.0)
    bis = BisectionResult(
        tau1=8.0, tau2=8.0, t0=2.0, t1=10.0, F_min=900.0, iterations=1, converged=True
    )
    t, F, u = build_formula9_trajectory(params, cfg, bis, num_cfg, n_eval=200)
    _assert_valid_trajectory(t, F, u, cfg)


def test_build_trajectory_left_flush_arc(
    params: BiologicalParameters,
    num_cfg: NumericalConfig,
) -> None:
    """t0 == 0: the initial u=0 phase is absent (T-t1>0 branch)."""
    cfg = ControlConfig(T=10.0, U_max=5000.0)
    bis = BisectionResult(
        tau1=5.0, tau2=5.0, t0=0.0, t1=5.0, F_min=900.0, iterations=1, converged=True
    )
    t, F, u = build_formula9_trajectory(params, cfg, bis, num_cfg, n_eval=200)
    _assert_valid_trajectory(t, F, u, cfg)
    # Singular arc starts immediately, so F leaves F_bar from the first step.
    assert F[0] == pytest.approx(params.F_bar)


def test_build_trajectory_full_span_arc(
    params: BiologicalParameters,
    num_cfg: NumericalConfig,
) -> None:
    """t0 == 0 and t1 == T: the singular arc spans the whole horizon."""
    cfg = ControlConfig(T=10.0, U_max=5000.0)
    bis = BisectionResult(
        tau1=10.0, tau2=10.0, t0=0.0, t1=10.0, F_min=800.0, iterations=1, converged=True
    )
    t, F, u = build_formula9_trajectory(params, cfg, bis, num_cfg, n_eval=200)
    _assert_valid_trajectory(t, F, u, cfg)


# ---------------------------------------------------------------------------
# solve_by_bisection — full solve loop (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_solve_by_bisection_converged_structure(
    params: BiologicalParameters,
) -> None:
    """A loose tolerance lets the solver converge; the result must be coherent.

    Structural invariants (Almeida 2022, Theorem 3.3 / formula 9):
    ``0 <= t0 <= t1 <= T``, ``tau2 >= tau1`` (terminal u=0 phase has length
    tau2 - tau1 >= 0), and the reached ``F_min`` never exceeds the persistence
    equilibrium F_bar (the u=0 baseline the search starts from).
    """
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = solve_by_bisection(params, cfg, max_iterations=40, tolerance=200.0)
    assert isinstance(result, BisectionResult)
    assert result.converged
    assert 0.0 <= result.t0 <= result.t1 <= cfg.T
    assert result.tau2 >= result.tau1 - 1e-9
    assert result.F_min <= params.F_bar + 1e-6
    assert 1 <= result.iterations <= 40


@pytest.mark.slow
def test_solve_by_bisection_exhausts_iterations(
    params: BiologicalParameters,
) -> None:
    """An unreachable tolerance in few steps exits via the non-converged path.

    With a tiny tolerance and a capped iteration budget the bisection cannot
    hit |F_min - epsilon| < tolerance, so it must return converged=False after
    exactly ``max_iterations`` steps while still yielding a valid structure.
    """
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = solve_by_bisection(params, cfg, max_iterations=3, tolerance=1e-9)
    assert not result.converged
    assert result.iterations == 3
    assert 0.0 <= result.t0 <= result.t1 <= cfg.T
    assert np.isfinite(result.F_min)
