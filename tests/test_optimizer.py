"""Unit tests for the optimiser module.

Tests requiring GEKKO are marked as `slow` and are skipped if the
library is not installed.
"""

from __future__ import annotations

import pytest

from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
)

gekko = pytest.importorskip("gekko", reason="GEKKO not installed")

from sit_control.optimizer import GekkoOptimiser, OptimisationResult  # noqa: E402


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters()


@pytest.fixture
def num_config() -> NumericalConfig:
    """Reduced collocation for fast tests."""
    return NumericalConfig(n_collocation=80)


@pytest.fixture
def optimiser(
    params: BiologicalParameters,
    num_config: NumericalConfig,
) -> GekkoOptimiser:
    return GekkoOptimiser(params, num_config)


@pytest.mark.slow
def test_L1_returns_result(optimiser: GekkoOptimiser) -> None:
    """solve_L1 should return an OptimisationResult."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L1(cfg)
    assert isinstance(result, OptimisationResult)


@pytest.mark.slow
def test_L1_convergence(optimiser: GekkoOptimiser) -> None:
    """Standard problem should converge."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L1(cfg)
    assert result.converged


@pytest.mark.slow
def test_L1_terminal_constraint(
    optimiser: GekkoOptimiser,
    params: BiologicalParameters,
) -> None:
    """F(T) should be at most epsilon (within numerical tolerance)."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    epsilon = cfg.epsilon if cfg.epsilon is not None else params.F_bar / 4.0
    result = optimiser.solve_L1(cfg)
    assert result.F_opt[-1] <= epsilon * 1.05  # 5% slack


@pytest.mark.slow
def test_L1_control_bounds(optimiser: GekkoOptimiser) -> None:
    """The control should lie in [0, U_max]."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L1(cfg)
    assert result.u_opt.min() >= -1e-6
    assert result.u_opt.max() <= cfg.U_max + 1e-6


@pytest.mark.slow
def test_L1_cost_matches_reference(optimiser: GekkoOptimiser) -> None:
    """GEKKO/APOPT converges to a non-convex LOCAL minimum (~1.69e5), not the
    global optimum (1.328e5, found by bisection). This guards that the local
    minimum stays reproducible; it matches the Chapter 3 convergence table.
    """
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L1(cfg)
    reference = 1.69e5  # GEKKO/APOPT local minimum at K=18258 (Ch.3 conv. table)
    rel_error = abs(result.cost - reference) / reference
    assert rel_error < 0.05, f"Relative error {rel_error:.2%} exceeds 5%"


@pytest.mark.slow
def test_L1_cost_constant_for_large_T(optimiser: GekkoOptimiser) -> None:
    """J(u*) should be roughly constant for T > T*.

    Theorem 3.3 of Almeida et al. (2022) states that for T > T*,
    J(u*) is independent of T.
    """
    cfg_150 = ControlConfig(T=150.0, U_max=5000.0)
    cfg_200 = ControlConfig(T=200.0, U_max=5000.0)

    r_150 = optimiser.solve_L1(cfg_150)
    r_200 = optimiser.solve_L1(cfg_200)

    rel_diff = abs(r_150.cost - r_200.cost) / r_150.cost
    assert rel_diff < 0.02, f"J(u*) differs by {rel_diff:.2%} between T=150 and T=200"


def test_invalid_T_raises() -> None:
    """Negative T should raise ValueError."""
    with pytest.raises(ValueError, match="T must be positive"):
        ControlConfig(T=-10.0, U_max=5000.0)


def test_invalid_U_max_raises() -> None:
    """Negative U_max should raise ValueError."""
    with pytest.raises(ValueError, match="U_max must be positive"):
        ControlConfig(T=150.0, U_max=-1.0)


def test_invalid_epsilon_raises() -> None:
    """Negative epsilon should raise ValueError."""
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ControlConfig(T=150.0, U_max=5000.0, epsilon=-1.0)


# ---------------------------------------------------------------------------
# solve_L2 tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_L2_returns_result(optimiser: GekkoOptimiser) -> None:
    """solve_L2 should return an OptimisationResult."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L2(cfg)
    assert isinstance(result, OptimisationResult)


@pytest.mark.slow
def test_L2_convergence(optimiser: GekkoOptimiser) -> None:
    """Standard L2 problem should converge."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L2(cfg)
    assert result.converged


@pytest.mark.slow
def test_L2_terminal_constraint(
    optimiser: GekkoOptimiser,
    params: BiologicalParameters,
) -> None:
    """F(T) should satisfy the suppression constraint (5% numerical slack)."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    epsilon = cfg.epsilon if cfg.epsilon is not None else params.F_bar / 4.0
    result = optimiser.solve_L2(cfg)
    assert result.F_opt[-1] <= epsilon * 1.05


@pytest.mark.slow
def test_L2_control_bounds(optimiser: GekkoOptimiser) -> None:
    """The L2 control should lie in [0, U_max]."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L2(cfg)
    assert result.u_opt.min() >= -1e-6
    assert result.u_opt.max() <= cfg.U_max + 1e-6


@pytest.mark.slow
def test_L2_smoother_than_L1(optimiser: GekkoOptimiser) -> None:
    """L2 control should have lower variance than L1 (smoother profile).

    The quadratic penalty explicitly penalises large u values, so the
    resulting profile should vary less than the bang-singular-bang L1 solution.
    """
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    r_l1 = optimiser.solve_L1(cfg)
    r_l2 = optimiser.solve_L2(cfg)
    assert r_l2.u_opt.std() < r_l1.u_opt.std(), (
        f"L2 std={r_l2.u_opt.std():.1f} should be < L1 std={r_l1.u_opt.std():.1f}"
    )


@pytest.mark.slow
def test_L2_cost_is_quadratic_functional(optimiser: GekkoOptimiser) -> None:
    """The stored cost should equal (1/2)*integral(u^2) with c=1."""
    import numpy as np

    cfg = ControlConfig(T=150.0, U_max=5000.0)
    result = optimiser.solve_L2(cfg, c_weight=1.0)
    expected = float(np.trapezoid(0.5 * result.u_opt**2, result.t))
    assert abs(result.cost - expected) / (expected + 1e-12) < 1e-6


def test_L2_invalid_c_weight_raises(optimiser: GekkoOptimiser) -> None:
    """Non-positive c_weight should raise ValueError before calling GEKKO."""
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    with pytest.raises(ValueError, match="c_weight must be positive"):
        optimiser.solve_L2(cfg, c_weight=-1.0)
