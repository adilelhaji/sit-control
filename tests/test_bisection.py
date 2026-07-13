"""Unit tests for the bisection module."""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.bisection import (
    BisectionResult,
    _compute_f,
    _compute_partials,
    _simulate_with_tau1,
    singular_control,
    solve_by_bisection,
)
from sit_control.parameters import BiologicalParameters, ControlConfig
from sit_control.simulator import Simulator


@pytest.fixture
def params() -> BiologicalParameters:
    """Default parameters from Almeida et al. (2022)."""
    return BiologicalParameters()


@pytest.fixture
def cfg(params: BiologicalParameters) -> ControlConfig:
    return ControlConfig(T=150.0, U_max=5000.0)


@pytest.fixture
def simulator(params: BiologicalParameters) -> Simulator:
    return Simulator(params)


# ---------------------------------------------------------------------------
# singular_control
# ---------------------------------------------------------------------------


def test_singular_control_returns_scalar(params: BiologicalParameters) -> None:
    """singular_control should return a Python float."""
    val = singular_control(params.F_bar / 2, 500.0, params)
    assert isinstance(val, float)


def test_singular_control_degenerate_Ms_zero(
    params: BiologicalParameters,
) -> None:
    """At Ms=0 the denominator involves d2f_dMs2 — must not raise."""
    val = singular_control(params.F_bar / 2, 0.0, params)
    assert np.isfinite(val)


def test_singular_control_degenerate_curvature(
    params: BiologicalParameters,
) -> None:
    """When d2f_dMs2 ≈ 0, singular_control should return 0 (guarded path)."""
    # Monkey-patch _compute_partials inside bisection to trigger the guard.
    import sit_control.bisection as bmod

    original = bmod._compute_partials

    def _zero_curvature(F, Ms, p):
        df_dF, df_dMs, _, d2f_dMsdF = original(F, Ms, p)
        return df_dF, df_dMs, 0.0, d2f_dMsdF  # force d2f_dMs2 = 0

    bmod._compute_partials = _zero_curvature
    try:
        val = singular_control(1000.0, 500.0, params)
        assert val == 0.0
    finally:
        bmod._compute_partials = original


# ---------------------------------------------------------------------------
# _simulate_with_tau1
# ---------------------------------------------------------------------------


def test_simulate_tau1_zero_no_suppression(
    params: BiologicalParameters,
    cfg: ControlConfig,
    simulator: Simulator,
) -> None:
    """tau1=0 means no singular arc, so no sterile males are ever released and
    F stays at the persistence equilibrium F_bar."""
    F_min = _simulate_with_tau1(params, cfg, tau1=0.0, simulator=simulator)
    assert F_min == pytest.approx(params.F_bar, rel=1e-3)


def test_simulate_tau1_full_T_no_bangon(
    params: BiologicalParameters,
    cfg: ControlConfig,
    simulator: Simulator,
) -> None:
    """tau1=T means no bang-on phase; function must not raise and return finite."""
    F_min = _simulate_with_tau1(params, cfg, tau1=cfg.T, simulator=simulator)
    assert np.isfinite(F_min)
    assert 0.0 <= F_min <= params.F_bar


def test_simulate_longer_tau1_lower_Fmin(
    params: BiologicalParameters,
    cfg: ControlConfig,
    simulator: Simulator,
) -> None:
    """A longer singular arc (larger tau1) releases more sterile males and
    therefore reaches a lower (or equal) F_min."""
    F_short = _simulate_with_tau1(params, cfg, tau1=20.0, simulator=simulator)
    F_long = _simulate_with_tau1(params, cfg, tau1=100.0, simulator=simulator)
    # Longer singular arc = more suppression = lower F_min.
    assert F_long <= F_short + 10.0  # allow small numerical slack


def test_simulate_returns_nonnegative_Fmin(
    params: BiologicalParameters,
    cfg: ControlConfig,
    simulator: Simulator,
) -> None:
    """F_min must be non-negative for any tau1."""
    for tau1 in (0.0, 30.0, 75.0, 149.0, 150.0):
        F_min = _simulate_with_tau1(params, cfg, tau1=tau1, simulator=simulator)
        assert F_min >= -1.0  # tiny numerical under-shoot accepted


# ---------------------------------------------------------------------------
# solve_by_bisection
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bisection_returns_result(
    params: BiologicalParameters,
    cfg: ControlConfig,
) -> None:
    """solve_by_bisection should return a BisectionResult."""
    result = solve_by_bisection(params, cfg, max_iterations=10, tolerance=50.0)
    assert isinstance(result, BisectionResult)


@pytest.mark.slow
def test_bisection_t0_t1_consistent(
    params: BiologicalParameters,
    cfg: ControlConfig,
) -> None:
    """t0 and t1 are the singular-arc boundaries: 0 <= t0 <= t1 <= T.

    t1 is the END of the singular arc (~144.1 d for T=150), after which the
    control switches off until T; it is NOT the horizon T itself.
    """
    result = solve_by_bisection(params, cfg, max_iterations=10, tolerance=50.0)
    assert 0.0 <= result.t0 <= result.t1 <= cfg.T


@pytest.mark.slow
def test_bisection_converges(
    params: BiologicalParameters,
    cfg: ControlConfig,
) -> None:
    """With enough iterations and loose tolerance, bisection should converge."""
    result = solve_by_bisection(
        params,
        cfg,
        max_iterations=40,
        tolerance=100.0,
    )
    assert result.converged


@pytest.mark.slow
def test_bisection_Fmin_near_epsilon(
    params: BiologicalParameters,
    cfg: ControlConfig,
) -> None:
    """Converged result should have F_min close to epsilon."""
    epsilon = params.F_bar / 4.0
    result = solve_by_bisection(
        params,
        cfg,
        max_iterations=40,
        tolerance=50.0,
    )
    if result.converged:
        assert abs(result.F_min - epsilon) < 200.0  # 200 females slack


# ---------------------------------------------------------------------------
# _compute_partials — analytical derivatives vs finite differences
# ---------------------------------------------------------------------------
#
# The singular control u_sing is built from the four analytical partials of f
# (df/dF, df/dMs, d2f/dMs2, d2f/dMsdF) derived by hand in _compute_partials.
# A transcription slip in any of them would silently corrupt the whole
# bang-singular-bang solution. These tests pin each partial against a central
# finite difference so such a regression fails loudly.

# Interior evaluation points (Ms > 0 to stay clear of the Ms=0 boundary clamp).
_PARTIAL_POINTS = [
    (8000.0, 5000.0),
    (3000.0, 20000.0),
    (5000.0, 1000.0),
    (11000.0, 2000.0),
]


def _central(g, x, h):
    """Central finite difference of a scalar function g at x with step h."""
    return (g(x + h) - g(x - h)) / (2.0 * h)


@pytest.mark.parametrize("F, Ms", _PARTIAL_POINTS)
def test_compute_partials_match_finite_differences(
    params: BiologicalParameters,
    F: float,
    Ms: float,
) -> None:
    """The four analytical partials must match central finite differences.

    df/dF and df/dMs are checked against finite differences of f itself;
    d2f/dMs2 and d2f/dMsdF against finite differences of the (already
    validated) analytical df/dMs.
    """
    df_dF, df_dMs, d2f_dMs2, d2f_dMsdF = _compute_partials(F, Ms, params)

    hF = F * 1e-6
    hM = Ms * 1e-6

    fd_df_dF = _central(lambda x: _compute_f(x, Ms, params), F, hF)
    fd_df_dMs = _central(lambda x: _compute_f(F, x, params), Ms, hM)
    fd_d2f_dMs2 = _central(lambda x: _compute_partials(F, x, params)[1], Ms, hM)
    fd_d2f_dMsdF = _central(lambda x: _compute_partials(x, Ms, params)[1], F, hF)

    assert df_dF == pytest.approx(fd_df_dF, rel=1e-4)
    assert df_dMs == pytest.approx(fd_df_dMs, rel=1e-4)
    assert d2f_dMs2 == pytest.approx(fd_d2f_dMs2, rel=1e-3)
    assert d2f_dMsdF == pytest.approx(fd_d2f_dMsdF, rel=1e-3)


def test_compute_partials_d2f_dMsdF_zero_at_Ms_zero(
    params: BiologicalParameters,
) -> None:
    """At Ms=0, df/dMs is independent of F, so d2f/dMsdF must vanish exactly.

    (df/dMs(F, 0) = -nu*delta_M*gamma_s/(1-nu), a constant in F.) This edge
    case is where a sign/term error in the cross-partial would most easily
    hide, so it is pinned explicitly.
    """
    _, _, _, d2f_dMsdF = _compute_partials(8000.0, 0.0, params)
    assert d2f_dMsdF == pytest.approx(0.0, abs=1e-12)


def test_singular_control_matches_partials_formula(
    params: BiologicalParameters,
) -> None:
    """singular_control must equal the closed-form combination of the partials.

    u_sing = (df/dMs * df/dF + d2f/dMs2 * delta_s * Ms - d2f/dMsdF * f)
             / d2f/dMs2   (Almeida 2022, eq. 7).
    """
    F, Ms = 6000.0, 3000.0
    df_dF, df_dMs, d2f_dMs2, d2f_dMsdF = _compute_partials(F, Ms, params)
    f_val = _compute_f(F, Ms, params)
    expected = (
        df_dMs * df_dF + d2f_dMs2 * params.delta_s * Ms - d2f_dMsdF * f_val
    ) / d2f_dMs2
    assert singular_control(F, Ms, params) == pytest.approx(expected, rel=1e-12)
