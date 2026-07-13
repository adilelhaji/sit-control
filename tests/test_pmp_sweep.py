"""Unit tests for the pmp_sweep (Forward-Backward Sweep) module.

The Forward-Backward Sweep (FBS) is the approximate bang-bang PMP contrast
solver (Lenhart & Workman, 2007). These tests exercise its public API on
small, short-horizon problems so each runs fast: the S1 dynamics helper
``_f1`` and its finite-difference partials, the forward/backward ODE passes,
and the full ``solve_L1`` loop (warm start, cold start, convergence break,
and the ``n_iter`` guard).
"""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.parameters import BiologicalParameters, ControlConfig
from sit_control.pmp_sweep import FBSweepResult, ForwardBackwardSweep


@pytest.fixture
def params() -> BiologicalParameters:
    """Default parameters from Almeida et al. (2022)."""
    return BiologicalParameters()


@pytest.fixture
def sweep(params: BiologicalParameters) -> ForwardBackwardSweep:
    """A Forward-Backward Sweep solver with default numerical config."""
    return ForwardBackwardSweep(params)


@pytest.fixture
def cfg() -> ControlConfig:
    """Short-horizon control config for fast tests."""
    return ControlConfig(T=30.0, U_max=5000.0)


# ---------------------------------------------------------------------------
# _f1 — reduced S1 growth rate
# ---------------------------------------------------------------------------


def test_f1_returns_finite_scalar(sweep: ForwardBackwardSweep) -> None:
    """_f1 should return a finite Python float for interior states."""
    val = sweep._f1(sweep.params.F_bar / 2, 500.0)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_f1_zero_at_origin(sweep: ForwardBackwardSweep) -> None:
    """f1(0, 0) must be exactly 0 (numerator vanishes at F=0)."""
    assert sweep._f1(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_f1_clamps_negative_states(sweep: ForwardBackwardSweep) -> None:
    """Negative F/Ms are clamped to zero, so f1 matches the origin value."""
    assert sweep._f1(-100.0, -50.0) == pytest.approx(sweep._f1(0.0, 0.0), abs=1e-12)


def test_f1_equilibrium_near_zero(
    sweep: ForwardBackwardSweep, params: BiologicalParameters
) -> None:
    """At the persistence equilibrium F_bar (Ms=0) the growth rate is ~0."""
    val = sweep._f1(params.F_bar, 0.0)
    assert abs(val) < 1.0  # small residual relative to F_bar ~ 11037


def test_f1_more_sterile_males_lowers_growth(sweep: ForwardBackwardSweep) -> None:
    """More sterile males reduce recruitment, hence a lower growth rate."""
    low_ms = sweep._f1(6000.0, 500.0)
    high_ms = sweep._f1(6000.0, 20000.0)
    assert high_ms < low_ms


# ---------------------------------------------------------------------------
# _df1_dF / _df1_dMs — finite-difference partials
# ---------------------------------------------------------------------------


def test_df1_dF_matches_finite_difference(sweep: ForwardBackwardSweep) -> None:
    """_df1_dF must match an independent central finite difference of f1."""
    F, Ms = 6000.0, 3000.0
    h = 1.0
    fd = (sweep._f1(F + h, Ms) - sweep._f1(F - h, Ms)) / (2.0 * h)
    assert sweep._df1_dF(F, Ms) == pytest.approx(fd, rel=1e-3, abs=1e-6)


def test_df1_dMs_matches_finite_difference(sweep: ForwardBackwardSweep) -> None:
    """_df1_dMs must match an independent central finite difference of f1."""
    F, Ms = 6000.0, 3000.0
    h = 1.0
    fd = (sweep._f1(F, Ms + h) - sweep._f1(F, Ms - h)) / (2.0 * h)
    assert sweep._df1_dMs(F, Ms) == pytest.approx(fd, rel=1e-3, abs=1e-6)


def test_df1_dMs_negative(sweep: ForwardBackwardSweep) -> None:
    """Growth strictly decreases in Ms, so ∂f1/∂Ms < 0 for interior states."""
    assert sweep._df1_dMs(6000.0, 3000.0) < 0.0


# ---------------------------------------------------------------------------
# _forward — state integration
# ---------------------------------------------------------------------------


def test_forward_initial_conditions(
    sweep: ForwardBackwardSweep, params: BiologicalParameters
) -> None:
    """Forward pass starts at (F_bar, 0) by construction."""
    t = np.linspace(0.0, 30.0, 40)
    u = np.full(40, 1000.0)
    F, Ms = sweep._forward(t, u)
    assert F[0] == pytest.approx(params.F_bar, rel=1e-6)
    assert Ms[0] == pytest.approx(0.0, abs=1e-9)


def test_forward_shapes_and_finiteness(sweep: ForwardBackwardSweep) -> None:
    """Forward pass returns finite arrays matching the time grid length."""
    t = np.linspace(0.0, 30.0, 40)
    u = np.full(40, 1000.0)
    F, Ms = sweep._forward(t, u)
    assert F.shape == t.shape
    assert Ms.shape == t.shape
    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(Ms))


def test_forward_release_builds_sterile_males(sweep: ForwardBackwardSweep) -> None:
    """A positive constant release grows the sterile-male population from 0."""
    t = np.linspace(0.0, 30.0, 40)
    u = np.full(40, 2000.0)
    _, Ms = sweep._forward(t, u)
    assert Ms[-1] > Ms[0]
    assert np.all(Ms >= -1e-6)


def test_forward_more_release_suppresses_females(sweep: ForwardBackwardSweep) -> None:
    """A larger release drives F(T) lower than a smaller one."""
    t = np.linspace(0.0, 30.0, 40)
    _, _ = sweep._forward(t, np.zeros(40))
    F_small, _ = sweep._forward(t, np.full(40, 1000.0))
    F_large, _ = sweep._forward(t, np.full(40, 8000.0))
    assert F_large[-1] < F_small[-1]


# ---------------------------------------------------------------------------
# _backward — adjoint integration
# ---------------------------------------------------------------------------


def test_backward_terminal_condition(sweep: ForwardBackwardSweep) -> None:
    """Adjoint λ₁(T)=μ, λ₂(T)=0 at the terminal time (last grid point)."""
    t = np.linspace(0.0, 30.0, 40)
    F, Ms = sweep._forward(t, np.full(40, 1000.0))
    mu = 2.5
    lam1, lam2 = sweep._backward(t, F, Ms, mu)
    assert lam1[-1] == pytest.approx(mu, rel=1e-4)
    assert lam2[-1] == pytest.approx(0.0, abs=1e-6)


def test_backward_shapes_and_finiteness(sweep: ForwardBackwardSweep) -> None:
    """Backward pass returns finite arrays matching the time grid length."""
    t = np.linspace(0.0, 30.0, 40)
    F, Ms = sweep._forward(t, np.full(40, 1000.0))
    lam1, lam2 = sweep._backward(t, F, Ms, 1.0)
    assert lam1.shape == t.shape
    assert lam2.shape == t.shape
    assert np.all(np.isfinite(lam1))
    assert np.all(np.isfinite(lam2))


# ---------------------------------------------------------------------------
# solve_L1 — full Forward-Backward Sweep
# ---------------------------------------------------------------------------


def test_solve_L1_returns_result(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """solve_L1 should return an FBSweepResult."""
    result = sweep.solve_L1(cfg, n_iter=5, n_grid=40)
    assert isinstance(result, FBSweepResult)


def test_solve_L1_consistent_shapes(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """All trajectory arrays share the n_grid length."""
    n_grid = 40
    result = sweep.solve_L1(cfg, n_iter=5, n_grid=n_grid)
    for arr in (
        result.t,
        result.u_opt,
        result.F_opt,
        result.Ms_opt,
        result.lambda1,
        result.lambda2,
    ):
        assert arr.shape == (n_grid,)


def test_solve_L1_control_within_bounds(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """The control profile stays within [0, U_max] at every point."""
    result = sweep.solve_L1(cfg, n_iter=5, n_grid=40)
    assert np.all(result.u_opt >= -1e-9)
    assert np.all(result.u_opt <= cfg.U_max + 1e-9)


def test_solve_L1_diagnostics_finite(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """Scalar diagnostics (cost, wall_time, F_terminal) are finite and sane."""
    result = sweep.solve_L1(cfg, n_iter=5, n_grid=40)
    assert np.isfinite(result.cost)
    assert result.cost >= 0.0
    assert result.wall_time >= 0.0
    assert np.isfinite(result.F_terminal)
    assert result.F_terminal == pytest.approx(result.F_opt[-1], rel=1e-9)


def test_solve_L1_cost_matches_trapezoid(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """Reported cost equals the trapezoidal integral of the control."""
    result = sweep.solve_L1(cfg, n_iter=5, n_grid=40)
    expected = float(np.trapezoid(result.u_opt, result.t))
    assert result.cost == pytest.approx(expected, rel=1e-9)


def test_solve_L1_n_iterations_capped(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """Without convergence, n_iterations equals the n_iter budget."""
    n_iter = 4
    result = sweep.solve_L1(cfg, n_iter=n_iter, n_grid=40, tol=1e-12)
    assert result.n_iterations == n_iter
    assert not result.converged


def test_solve_L1_converges_on_loose_tol(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """A very loose tolerance triggers the convergence break on the first iter."""
    result = sweep.solve_L1(cfg, n_iter=5, n_grid=30, tol=1e12)
    assert result.converged
    assert result.n_iterations == 1


def test_solve_L1_warm_start(sweep: ForwardBackwardSweep, cfg: ControlConfig) -> None:
    """A provided warm-start control is accepted and clipped into bounds."""
    cold = sweep.solve_L1(cfg, n_iter=3, n_grid=40)
    warm = sweep.solve_L1(cfg, u_init=(cold.t, cold.u_opt), n_iter=3, n_grid=40)
    assert isinstance(warm, FBSweepResult)
    assert np.all(warm.u_opt <= cfg.U_max + 1e-9)
    assert np.all(warm.u_opt >= -1e-9)


def test_solve_L1_default_epsilon(sweep: ForwardBackwardSweep) -> None:
    """With epsilon=None the solver falls back to F_bar/4 without error."""
    cfg_no_eps = ControlConfig(T=30.0, U_max=5000.0, epsilon=None)
    result = sweep.solve_L1(cfg_no_eps, n_iter=3, n_grid=30)
    assert np.isfinite(result.cost)


def test_solve_L1_rejects_nonpositive_n_iter(
    sweep: ForwardBackwardSweep, cfg: ControlConfig
) -> None:
    """n_iter < 1 must raise ValueError (fail-loud on invalid config)."""
    with pytest.raises(ValueError, match="n_iter"):
        sweep.solve_L1(cfg, n_iter=0, n_grid=30)
