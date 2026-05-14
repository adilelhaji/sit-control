"""Unit tests for the simulator module."""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.controls import constant_control, zero_control
from sit_control.parameters import BiologicalParameters, NumericalConfig
from sit_control.simulator import Simulator, SimulationResult


@pytest.fixture
def simulator() -> Simulator:
    """Default simulator with Almeida (2022) parameters."""
    return Simulator(BiologicalParameters())


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters()


def test_simulate_returns_correct_type(simulator: Simulator) -> None:
    """The simulator should return a SimulationResult."""
    result = simulator.simulate(T=10.0, u_func=zero_control(), model="S1")
    assert isinstance(result, SimulationResult)
    assert result.success


def test_simulate_S1_shape(simulator: Simulator) -> None:
    """S1 simulation should produce a (2, n_eval) state array."""
    result = simulator.simulate(
        T=10.0, u_func=zero_control(), model="S1", n_eval=100,
    )
    assert result.state.shape == (2, 100)
    assert result.t.shape == (100,)
    assert result.control.shape == (100,)


def test_simulate_S2_shape(simulator: Simulator) -> None:
    """S2 simulation should produce a (4, n_eval) state array."""
    result = simulator.simulate(
        T=10.0, u_func=zero_control(), model="S2", n_eval=100,
    )
    assert result.state.shape == (4, 100)


def test_equilibrium_stability_S1(
    simulator: Simulator,
    params: BiologicalParameters,
) -> None:
    """Starting from F_bar with u=0, F should remain near F_bar."""
    result = simulator.simulate(T=200.0, u_func=zero_control(), model="S1")
    F_final = result.state[0, -1]
    relative_drift = abs(F_final - params.F_bar) / params.F_bar
    assert relative_drift < 1e-3, (
        f"Equilibrium not preserved: drift={relative_drift:.2e}"
    )


def test_zero_control_preserves_Ms_at_zero(simulator: Simulator) -> None:
    """With u=0 and Ms(0)=0, Ms(t) should remain zero."""
    result = simulator.simulate(T=100.0, u_func=zero_control(), model="S1")
    assert np.max(np.abs(result.state[1])) < 1e-6


def test_constant_control_drives_Ms_to_equilibrium(
    simulator: Simulator,
    params: BiologicalParameters,
) -> None:
    """With constant u, Ms should converge to u/delta_s."""
    u_const = 1000.0
    expected_Ms = u_const / params.delta_s

    result = simulator.simulate(
        T=500.0,
        u_func=constant_control(u_const),
        model="S1",
    )
    Ms_final = result.state[1, -1]
    assert Ms_final == pytest.approx(expected_Ms, rel=1e-2)


def test_invalid_model_raises(simulator: Simulator) -> None:
    """An unknown model identifier should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        simulator.simulate(T=10.0, u_func=zero_control(), model="S3")  # type: ignore[arg-type]


def test_custom_initial_state(simulator: Simulator) -> None:
    """A user-provided initial state should be respected."""
    initial = np.array([5000.0, 100.0])
    result = simulator.simulate(
        T=0.1, u_func=zero_control(), model="S1",
        initial_state=initial, n_eval=2,
    )
    np.testing.assert_allclose(result.state[:, 0], initial, rtol=1e-6)


def test_solver_tolerances_respected() -> None:
    """Tighter tolerances should produce smaller equilibrium drift."""
    params = BiologicalParameters()

    loose = Simulator(params, NumericalConfig(rtol=1e-4, atol=1e-4))
    tight = Simulator(params, NumericalConfig(rtol=1e-10, atol=1e-10))

    r_loose = loose.simulate(T=300.0, u_func=zero_control(), model="S1")
    r_tight = tight.simulate(T=300.0, u_func=zero_control(), model="S1")

    drift_loose = abs(r_loose.state[0, -1] - params.F_bar) / params.F_bar
    drift_tight = abs(r_tight.state[0, -1] - params.F_bar) / params.F_bar

    assert drift_tight <= drift_loose