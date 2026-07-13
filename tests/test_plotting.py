"""Unit tests for the plotting module.

These tests exercise every public plotting function on minimal
synthetic inputs, using a headless Matplotlib backend so no display
is required. Each test asserts that the code path runs without
raising, returns a ``Figure`` and, when a ``save_path`` is given,
writes non-empty output files.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from sit_control.optimizer import OptimisationResult  # noqa: E402
from sit_control.plotting import (  # noqa: E402
    plot_convergence,
    plot_model_comparison,
    plot_optimal_solution,
    plot_strategy_comparison,
)
from sit_control.simulator import SimulationResult  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close all figures after each test to avoid resource warnings."""
    yield
    plt.close("all")


@pytest.fixture
def opt_result() -> OptimisationResult:
    """A minimal optimisation result with monotone synthetic data."""
    t = np.linspace(0.0, 10.0, 11)
    return OptimisationResult(
        t=t,
        u_opt=np.linspace(1.0, 0.0, 11),
        F_opt=np.linspace(1000.0, 100.0, 11),
        Ms_opt=np.linspace(0.0, 500.0, 11),
        cost=1.23e4,
        wall_time=0.5,
        converged=True,
    )


@pytest.fixture
def sim_S1() -> SimulationResult:
    """A minimal S1 (reduced) simulation with a 2-state trajectory."""
    t = np.linspace(0.0, 10.0, 11)
    state = np.vstack([np.linspace(1000.0, 200.0, 11), np.linspace(0.0, 300.0, 11)])
    return SimulationResult(
        t=t,
        state=state,
        control=np.linspace(1.0, 0.0, 11),
        model="S1",
        success=True,
    )


@pytest.fixture
def sim_S2() -> SimulationResult:
    """A minimal S2 (full) simulation with F at state index 2."""
    t = np.linspace(0.0, 10.0, 11)
    state = np.vstack(
        [
            np.linspace(500.0, 100.0, 11),
            np.linspace(500.0, 100.0, 11),
            np.linspace(1000.0, 150.0, 11),
        ]
    )
    return SimulationResult(
        t=t,
        state=state,
        control=np.linspace(1.0, 0.0, 11),
        model="S2",
        success=True,
    )


def _assert_saved(save_path: Path) -> None:
    """Assert the target file and its companion PNG exist and are non-empty."""
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    png_path = save_path.with_suffix(".png")
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_plot_optimal_solution_returns_figure(
    opt_result: OptimisationResult,
) -> None:
    """plot_optimal_solution should return a Figure without saving."""
    fig = plot_optimal_solution(opt_result)
    assert isinstance(fig, Figure)


def test_plot_optimal_solution_with_epsilon_and_save(
    opt_result: OptimisationResult, tmp_path: Path
) -> None:
    """The epsilon line and PDF/PNG saving paths should execute."""
    save_path = tmp_path / "optimal.pdf"
    fig = plot_optimal_solution(opt_result, epsilon=50.0, save_path=save_path)
    assert isinstance(fig, Figure)
    _assert_saved(save_path)


def test_plot_model_comparison_returns_figure(
    sim_S1: SimulationResult, sim_S2: SimulationResult
) -> None:
    """plot_model_comparison should return a Figure for valid S1/S2 inputs."""
    fig = plot_model_comparison(sim_S1, sim_S2)
    assert isinstance(fig, Figure)


def test_plot_model_comparison_saves(
    sim_S1: SimulationResult, sim_S2: SimulationResult, tmp_path: Path
) -> None:
    """Saving to a non-PNG path should also write a companion PNG."""
    save_path = tmp_path / "comparison.pdf"
    plot_model_comparison(sim_S1, sim_S2, save_path=save_path)
    _assert_saved(save_path)


def test_plot_model_comparison_wrong_models_raises(
    sim_S1: SimulationResult, sim_S2: SimulationResult
) -> None:
    """Swapping S1 and S2 should raise ValueError."""
    with pytest.raises(ValueError, match="S1 and S2"):
        plot_model_comparison(sim_S2, sim_S1)


def test_plot_convergence_returns_figure() -> None:
    """plot_convergence should return a Figure without a reference line."""
    N_values = np.array([10, 50, 100], dtype=np.int64)
    costs = np.array([1.0e4, 5.0e3, 4.0e3], dtype=np.float64)
    times = np.array([0.1, 0.5, 1.2], dtype=np.float64)
    fig = plot_convergence(N_values, costs, times)
    assert isinstance(fig, Figure)


def test_plot_convergence_with_reference_and_save(tmp_path: Path) -> None:
    """The reference-cost line and saving paths should execute."""
    N_values = np.array([10, 50, 100], dtype=np.int64)
    costs = np.array([1.0e4, 5.0e3, 4.0e3], dtype=np.float64)
    times = np.array([0.1, 0.5, 1.2], dtype=np.float64)
    save_path = tmp_path / "convergence.pdf"
    fig = plot_convergence(
        N_values, costs, times, reference_cost=4.2e3, save_path=save_path
    )
    assert isinstance(fig, Figure)
    _assert_saved(save_path)


def test_plot_strategy_comparison_returns_figure(
    sim_S1: SimulationResult,
) -> None:
    """plot_strategy_comparison should plot each strategy and return a Figure."""
    results = {"baseline": sim_S1}
    fig = plot_strategy_comparison(results)
    assert isinstance(fig, Figure)


def test_plot_strategy_comparison_with_epsilon_and_save(
    sim_S1: SimulationResult, sim_S2: SimulationResult, tmp_path: Path
) -> None:
    """The epsilon line, multiple strategies and saving should execute."""
    results = {"reduced": sim_S1, "full": sim_S2}
    save_path = tmp_path / "strategies.pdf"
    fig = plot_strategy_comparison(
        results,
        epsilon=100.0,
        save_path=save_path,
        state_index=0,
        state_label="Females $F(t)$",
    )
    assert isinstance(fig, Figure)
    _assert_saved(save_path)


def test_save_figure_png_only_writes_single_file(
    opt_result: OptimisationResult, tmp_path: Path
) -> None:
    """Saving directly to a .png path should not create a duplicate file."""
    save_path = tmp_path / "optimal.png"
    plot_optimal_solution(opt_result, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    pdf_siblings = list(tmp_path.glob("*.pdf"))
    assert pdf_siblings == []
