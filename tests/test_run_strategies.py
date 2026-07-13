"""Unit tests for the run_strategies script helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.controls import constant_control
from sit_control.parameters import BiologicalParameters, ControlConfig
from sit_control.simulator import Simulator


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters()


@pytest.fixture
def simulator(params: BiologicalParameters) -> Simulator:
    return Simulator(params)


@pytest.fixture
def cfg(params: BiologicalParameters) -> ControlConfig:
    return ControlConfig(T=30.0, U_max=5000.0)


# ---------------------------------------------------------------------------
# _variable_impulsive_control
# ---------------------------------------------------------------------------


def test_variable_impulsive_zero_outside_pulses() -> None:
    """Control should be zero outside pulse windows."""
    from scripts.run_strategies import _variable_impulsive_control

    times = np.array([0.0, 10.0, 20.0])
    amounts = np.array([100.0, 200.0, 300.0])
    u_func = _variable_impulsive_control(times, amounts, duration=0.5)
    assert u_func(5.0) == pytest.approx(0.0)
    assert u_func(15.0) == pytest.approx(0.0)


def test_variable_impulsive_correct_rate() -> None:
    """Within pulse window, rate should equal amount / duration."""
    from scripts.run_strategies import _variable_impulsive_control

    amount, duration = 500.0, 0.5
    u_func = _variable_impulsive_control(
        np.array([10.0]),
        np.array([amount]),
        duration=duration,
    )
    assert u_func(10.1) == pytest.approx(amount / duration)


def test_variable_impulsive_total_integral_approx() -> None:
    """Trapezoidal integral of control should approximate sum of amounts."""
    from scripts.run_strategies import _variable_impulsive_control

    times = np.array([0.0, 5.0, 10.0])
    amounts = np.array([100.0, 200.0, 300.0])
    duration = 0.5
    u_func = _variable_impulsive_control(times, amounts, duration=duration)
    t_grid = np.linspace(0, 15, 3000)
    u_grid = np.array([u_func(t) for t in t_grid])
    integral = float(np.trapezoid(u_grid, t_grid))
    assert integral == pytest.approx(sum(amounts), rel=0.01)


# ---------------------------------------------------------------------------
# _metrics
# ---------------------------------------------------------------------------


def test_metrics_finite_values(
    params: BiologicalParameters,
    simulator: Simulator,
    cfg: ControlConfig,
) -> None:
    """_metrics should return finite values for a valid simulation."""
    from scripts.run_strategies import _epsilon, _metrics

    eps = _epsilon(params, cfg)
    result = simulator.simulate(T=cfg.T, u_func=constant_control(1000.0), model="S1")
    m = _metrics(result.t, result.state[0], result.control, params, eps, 1.0)
    assert np.isfinite(m["J_L1"])
    assert np.isfinite(m["F_T_over_Fbar"])
    assert m["F_T_over_Fbar"] >= 0.0


def test_metrics_t_epsilon_none_when_not_suppressed(
    params: BiologicalParameters,
    simulator: Simulator,
    cfg: ControlConfig,
) -> None:
    """t_epsilon_days should be None when suppression is not reached."""
    from scripts.run_strategies import _epsilon, _metrics

    eps = _epsilon(params, cfg)
    # Zero control — F stays at F_bar, never reaches epsilon
    from sit_control.controls import zero_control

    result = simulator.simulate(T=cfg.T, u_func=zero_control(), model="S1")
    m = _metrics(result.t, result.state[0], result.control, params, eps)
    assert m["t_epsilon_days"] is None


# ---------------------------------------------------------------------------
# run_strategy_2
# ---------------------------------------------------------------------------


def test_strategy_2_preserves_cost(
    simulator: Simulator,
    cfg: ControlConfig,
) -> None:
    """Constant strategy should produce J ≈ J_reference (within 1%)."""
    from scripts.run_strategies import run_strategy_2

    J_ref = 50_000.0
    _, m = run_strategy_2(simulator, cfg, J_reference=J_ref)
    assert abs(m["J_L1"] - J_ref) / J_ref < 0.01


# ---------------------------------------------------------------------------
# run_strategy_3
# ---------------------------------------------------------------------------


def test_strategy_3_pulse_count(
    simulator: Simulator,
    cfg: ControlConfig,
) -> None:
    """Number of pulses equals the count of release times k*tau in [0, T)
    (releases at t=0, tau, 2*tau, ...), i.e. len(arange(0, T, tau))."""
    from scripts.run_strategies import run_strategy_3

    tau = 7.0
    expected_pulses = len(np.arange(0.0, cfg.T, tau))
    _, m = run_strategy_3(simulator, cfg, J_reference=50_000.0, tau=tau)
    assert m["n_pulses"] == expected_pulses


def test_strategy_3_preserves_total_cost(
    simulator: Simulator,
    cfg: ControlConfig,
) -> None:
    """Periodic impulsive strategy should preserve total cost within 2%."""
    from scripts.run_strategies import run_strategy_3

    J_ref = 50_000.0
    _, m = run_strategy_3(simulator, cfg, J_reference=J_ref, tau=5.0)
    assert abs(m["J_L1"] - J_ref) / J_ref < 0.02


# ---------------------------------------------------------------------------
# run_strategy_5 (no-GEKKO, small problem)
# ---------------------------------------------------------------------------


def test_strategy_5_returns_finite_metrics(
    simulator: Simulator,
    cfg: ControlConfig,
) -> None:
    """Strategy 5 should return finite metrics without raising."""
    from scripts.run_strategies import run_strategy_5

    _, m = run_strategy_5(
        simulator,
        cfg,
        tau=5.0,
        maxiter=20,
        J_initial_guess=50_000.0,
    )
    assert np.isfinite(m["J_L1"])
    assert len(m["amounts_per_pulse"]) == int(np.floor(cfg.T / 5.0))


def test_strategy_5_amounts_nonnegative(
    simulator: Simulator,
    cfg: ControlConfig,
) -> None:
    """All optimised batch amounts should be non-negative."""
    from scripts.run_strategies import run_strategy_5

    _, m = run_strategy_5(
        simulator,
        cfg,
        tau=5.0,
        maxiter=20,
        J_initial_guess=50_000.0,
    )
    assert all(c >= -1e-9 for c in m["amounts_per_pulse"])
