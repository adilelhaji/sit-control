"""Visualisation utilities for SIT simulations and optimisations.

All figures follow a consistent style suitable for academic
publication: monochromatic axes, vector output, LaTeX-rendered labels
when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .optimizer import OptimisationResult
from .simulator import SimulationResult

logger = logging.getLogger(__name__)


def _save_figure(fig: Figure, path: Path) -> None:
    """Save *fig* to *path* and write a companion PNG in the same directory.

    Both files share the rcParams settings (dpi=300, bbox=tight).
    If *path* already has a ``.png`` suffix only one file is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    logger.info("Saved: %s", path)
    if path.suffix.lower() != ".png":
        png_path = path.with_suffix(".png")
        fig.savefig(png_path)
        logger.info("Saved: %s", png_path)


# Style configuration (applied at import time)
plt.rcParams.update(
    {
        "figure.figsize": (10, 4),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
    }
)


def plot_optimal_solution(
    result: OptimisationResult,
    epsilon: float | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot the optimal control and the female trajectory.

    Reproduces the style of Figures 3--5 of Almeida et al. (2022).

    Args:
        result: Output of the optimiser.
        epsilon: Suppression threshold to draw as a horizontal line.
        save_path: If given, write the figure to this path.

    Returns:
        The created Matplotlib figure.
    """
    fig, (ax_F, ax_u) = plt.subplots(1, 2, figsize=(12, 4))

    ax_F.plot(result.t, result.F_opt, color="C0", label=r"$F^*(t)$")
    if epsilon is not None:
        ax_F.axhline(
            epsilon,
            color="C3",
            linestyle=":",
            label=rf"$\varepsilon = {epsilon:.0f}$",
        )
    ax_F.set_xlabel("Time (days)")
    ax_F.set_ylabel("Female population $F(t)$")
    ax_F.set_title("Optimal female trajectory")
    ax_F.legend(loc="best")

    ax_u.plot(result.t, result.u_opt, color="C1", drawstyle="steps-post")
    ax_u.set_xlabel("Time (days)")
    ax_u.set_ylabel(r"Control $u^*(t)$")
    ax_u.set_title("Optimal release rate")

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, Path(save_path))

    return fig


def plot_model_comparison(
    sim_S1: SimulationResult,
    sim_S2: SimulationResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Compare the female trajectories of S1 and S2.

    Args:
        sim_S1: Simulation result for the reduced model.
        sim_S2: Simulation result for the full model.
        save_path: If given, write the figure to this path.

    Returns:
        The created Matplotlib figure.

    Raises:
        ValueError: If the simulation results refer to wrong models.
    """
    if sim_S1.model != "S1" or sim_S2.model != "S2":
        raise ValueError("Provide simulations of S1 and S2 respectively")

    fig, ax = plt.subplots(figsize=(10, 4))

    # S1 has F as index 0; S2 has F as index 2
    ax.plot(sim_S1.t, sim_S1.state[0], label="S1 (reduced)", color="C0")
    ax.plot(
        sim_S2.t,
        sim_S2.state[2],
        label="S2 (full)",
        color="C1",
        linestyle="--",
    )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Female population $F(t)$")
    ax.set_title("S1 vs S2 comparison under identical control")
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, Path(save_path))

    return fig


def plot_convergence(
    N_values: NDArray[np.int64],
    costs: NDArray[np.float64],
    times: NDArray[np.float64],
    reference_cost: float | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot J(u*) and wall time as a function of N.

    Args:
        N_values: Array of discretisation sizes.
        costs: J(u*) for each N.
        times: Wall time (s) for each N.
        reference_cost: Reference value for J(u*) (horizontal line).
        save_path: If given, write the figure to this path.

    Returns:
        The created Matplotlib figure.
    """
    fig, (ax_J, ax_t) = plt.subplots(1, 2, figsize=(12, 4))

    ax_J.semilogx(N_values, costs, "o-", color="C0")
    if reference_cost is not None:
        ax_J.axhline(
            reference_cost,
            color="C3",
            linestyle=":",
            label=f"Reference: {reference_cost:.3e}",
        )
        ax_J.legend(loc="best")
    ax_J.set_xlabel("Discretisation $N$")
    ax_J.set_ylabel(r"$J(u^*)$")
    ax_J.set_title("Convergence of the cost functional")

    ax_t.loglog(N_values, times, "s-", color="C2")
    ax_t.set_xlabel("Discretisation $N$")
    ax_t.set_ylabel("Wall time (s)")
    ax_t.set_title("Computational cost")

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, Path(save_path))

    return fig


def plot_strategy_comparison(
    results: dict[str, SimulationResult],
    epsilon: float | None = None,
    save_path: Path | str | None = None,
    state_index: int = 0,
    state_label: str = "F(t)",
) -> Figure:
    """Compare multiple control strategies on the same axes.

    Args:
        results: Mapping of strategy name to simulation result.
        epsilon: Suppression threshold to draw.
        save_path: If given, write the figure to this path.
        state_index: Index of the state variable to plot.
        state_label: Y-axis label.

    Returns:
        The created Matplotlib figure.
    """
    fig, (ax_state, ax_u) = plt.subplots(1, 2, figsize=(12, 4))

    for name, result in results.items():
        ax_state.plot(result.t, result.state[state_index], label=name)
        ax_u.plot(result.t, result.control, label=name, drawstyle="steps-post")

    if epsilon is not None:
        ax_state.axhline(
            epsilon,
            color="black",
            linestyle=":",
            label=r"$\varepsilon$",
        )

    ax_state.set_xlabel("Time (days)")
    ax_state.set_ylabel(state_label)
    ax_state.set_title("State trajectories")
    ax_state.legend(loc="best")

    ax_u.set_xlabel("Time (days)")
    ax_u.set_ylabel("$u(t)$")
    ax_u.set_title("Release rate")
    ax_u.legend(loc="best")

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, Path(save_path))

    return fig
