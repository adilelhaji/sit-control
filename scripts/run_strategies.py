"""Execute and compare five SIT control strategies for Aedes polynesiensis.

Strategies
----------
1. Continuous optimal L^1 (GEKKO + APOPT)
2. Constant release with J = J*(u_L1) — same total cost as optimal
3. Periodic impulsive, tau in {3, 7, 14} days, same total cost
4. Continuous optimal L^2 (GEKKO + APOPT)
5. Optimal impulsive, tau = 7 days (scipy SLSQP over batch amounts {c_k})

Outputs
-------
results/strategies_metrics.json  — all metrics in JSON
results/fig_comparison.pdf       — F(t) overlay for all strategies

Usage
-----
python scripts/run_strategies.py --config configs/almeida2022.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from sit_control.bisection import build_formula9_trajectory, solve_by_bisection
from sit_control.controls import (
    constant_control,
    impulsive_control,
    interpolated_control,
)
from sit_control.metrics import cost_L1, suppression_time
from sit_control.optimizer import GekkoOptimiser, OptimisationResult
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)
from sit_control.plotting import plot_strategy_comparison
from sit_control.simulator import SimulationResult, Simulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epsilon(params: BiologicalParameters, cfg: ControlConfig) -> float:
    return cfg.epsilon if cfg.epsilon is not None else params.F_bar / 4.0


def _metrics(
    t: NDArray[np.float64],
    F: NDArray[np.float64],
    u: NDArray[np.float64],
    params: BiologicalParameters,
    epsilon: float,
    wall_time: float = 0.0,
) -> dict[str, Any]:
    """Standard performance metrics for one strategy."""
    t_eps = suppression_time(t, F, epsilon)
    return {
        "J_L1": cost_L1(t, u),
        "t_epsilon_days": t_eps,
        "F_T": float(F[-1]),
        "F_T_over_Fbar": float(F[-1]) / params.F_bar,
        "wall_time_s": round(wall_time, 3),
    }


def _variable_impulsive_control(
    times: NDArray[np.float64],
    amounts: NDArray[np.float64],
    duration: float = 0.5,
):
    """Impulsive control with individual amount per pulse (Strategy 5).

    Parameters
    ----------
    times : NDArray
        Release start times (days).
    amounts : NDArray
        Amount released at each pulse (individuals).
    duration : float
        Width of each rectangular pulse approximation (days).

    Returns
    -------
    Callable t -> u(t).
    """
    rates = np.asarray(amounts, dtype=np.float64) / duration
    times_arr = np.asarray(times, dtype=np.float64)

    def _u(t: float) -> float:
        for start, rate in zip(times_arr, rates):
            if start <= t < start + duration:
                return float(rate)
        return 0.0

    return _u


def _opt_to_sim(
    opt: OptimisationResult,
    simulator: Simulator,
    cfg: ControlConfig,
) -> SimulationResult:
    """Re-simulate with the GEKKO optimal control via SciPy for plotting."""
    u_func = interpolated_control(opt.t, opt.u_opt)
    return simulator.simulate(
        T=cfg.T,
        u_func=u_func,
        model="S1",
        initial_state=np.array([simulator.params.F_bar, 0.0]),
    )


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_strategy_1(
    optimiser: GekkoOptimiser,
    cfg: ControlConfig,
) -> tuple[OptimisationResult, dict[str, Any]]:
    """Strategy 1: continuous optimal L^1 via bisection (Section 4.1).

    Uses the structural optimum — bisection on the singular-arc duration
    (Almeida 2022, Algorithm 2) — instead of the GEKKO local minimum, so the
    reported cost is the true global optimum and coherent with Chapter 3.
    Valid for T > T* (singular structure); for short horizons T < T* the
    bisection does not apply.
    """
    params = optimiser.params
    logger.info("Strategy 1 — L1 optimal (bisection) | T=%g, U_max=%g", cfg.T, cfg.U_max)
    t0 = time.perf_counter()
    bis = solve_by_bisection(params, cfg)
    # n_eval=20000: the L1 cost is the trapezoidal integral of u over the
    # singular arc; the default grid (2000) under-resolves it and overstates
    # J by ~0.1% (1.330e5 vs the converged 1.328e5). The fine grid makes the
    # reported cost match the converged optimum used in Chapter 3.
    t, F, u = build_formula9_trajectory(
        params, cfg, bis, optimiser.num_config, n_eval=20000
    )
    wall = time.perf_counter() - t0
    result = OptimisationResult(
        t=t,
        u_opt=u,
        F_opt=F,
        Ms_opt=np.zeros_like(F),
        cost=float(np.trapezoid(u, t)),
        wall_time=wall,
        converged=bis.converged,
    )
    eps = _epsilon(params, cfg)
    m = _metrics(result.t, result.F_opt, result.u_opt, params, eps, wall)
    m.update({
        "converged": bis.converged,
        "method": "bisection",
        "tau1_days": bis.tau1,
        "t0_days": bis.t0,
        "t1_days": bis.t1,
    })
    return result, m


def run_strategy_2(
    simulator: Simulator,
    cfg: ControlConfig,
    J_reference: float,
) -> tuple[SimulationResult, dict[str, Any]]:
    """Strategy 2: constant release with total cost = J_reference (Section 4.2).

    Sets u_c = J_reference / T so that integral(u_c, 0, T) = J_reference.
    """
    u_c = J_reference / cfg.T
    logger.info("Strategy 2 — constant | u_c=%.2f", u_c)
    t0 = time.perf_counter()
    result = simulator.simulate(T=cfg.T, u_func=constant_control(u_c), model="S1")
    wall = time.perf_counter() - t0
    eps = _epsilon(simulator.params, cfg)
    m = _metrics(result.t, result.state[0], result.control, simulator.params, eps, wall)
    m["u_constant"] = u_c
    return result, m


def run_strategy_3(
    simulator: Simulator,
    cfg: ControlConfig,
    J_reference: float,
    tau: float,
) -> tuple[SimulationResult, dict[str, Any]]:
    """Strategy 3: periodic impulsive with period tau (Section 4.3).

    Distributes J_reference uniformly across all pulses.
    """
    times = np.arange(0.0, cfg.T, tau)
    amount = J_reference / len(times)
    logger.info(
        "Strategy 3 — impulsive | tau=%.0f d, %d pulses, c=%.1f each",
        tau, len(times), amount,
    )
    t0 = time.perf_counter()
    result = simulator.simulate(
        T=cfg.T,
        u_func=impulsive_control(times, amount),
        model="S1",
    )
    wall = time.perf_counter() - t0
    eps = _epsilon(simulator.params, cfg)
    m = _metrics(result.t, result.state[0], result.control, simulator.params, eps, wall)
    m.update({"tau_days": tau, "n_pulses": int(len(times)), "amount_per_pulse": amount})
    return result, m


def run_strategy_4(
    optimiser: GekkoOptimiser,
    cfg: ControlConfig,
    c_weight: float = 1.0,
) -> tuple[OptimisationResult, dict[str, Any]]:
    """Strategy 4: continuous optimal L^2 (Section 4.4)."""
    logger.info("Strategy 4 — L2 optimal | T=%g, c=%.2g", cfg.T, c_weight)
    t0 = time.perf_counter()
    result = optimiser.solve_L2(cfg, c_weight=c_weight)
    wall = time.perf_counter() - t0
    eps = _epsilon(optimiser.params, cfg)
    m = _metrics(result.t, result.F_opt, result.u_opt, optimiser.params, eps, wall)
    m.update({
        "J_L2": result.cost,
        "c_weight": c_weight,
        "converged": result.converged,
        "u_std": float(np.std(result.u_opt)),
    })
    return result, m


def run_strategy_5(
    simulator: Simulator,
    cfg: ControlConfig,
    tau: float = 7.0,
    pulse_duration: float = 0.5,
    J_initial_guess: float | None = None,
    maxiter: int = 300,
) -> tuple[SimulationResult, dict[str, Any]]:
    """Strategy 5: optimal impulsive — optimise per-pulse amounts (Section 4.5).

    Solves min Σ c_k subject to F(T) ≤ ε using scipy SLSQP.
    Each pulse is approximated as a rectangular release of width `pulse_duration`.

    Parameters
    ----------
    simulator : Simulator
    cfg : ControlConfig
    tau : float
        Release period in days.
    pulse_duration : float
        Rectangular approximation width for each Dirac impulse.
    J_initial_guess : float or None
        Total cost for the uniform initial guess; defaults to 2*ε*δ_s*T.
    maxiter : int
        Maximum SLSQP iterations.

    Returns
    -------
    (SimulationResult, metrics dict)
    """
    params = simulator.params
    eps = _epsilon(params, cfg)
    times = np.arange(0.0, cfg.T, tau)
    N = len(times)
    # Per-batch cap = full-period release budget (U_max sustained over tau),
    # not U_max * pulse_duration. The latter (≈2500) is far too small to ever
    # suppress and leaves the problem infeasible; U_max * tau (≈35000) lets
    # each batch carry a meaningful dose, matching Chapter 3's release budget.
    max_per_pulse = cfg.U_max * tau

    # Uniform initial guess — start from a feasible region (over-estimate)
    if J_initial_guess is None:
        J_initial_guess = 2.0 * eps * params.delta_s * cfg.T
    c_init = np.full(N, J_initial_guess / N)
    c_init = np.clip(c_init, 0.0, max_per_pulse)

    logger.info(
        "Strategy 5 — optimal impulsive | tau=%.0f d, %d pulses, maxiter=%d",
        tau, N, maxiter,
    )

    def _F_terminal(amounts: NDArray[np.float64]) -> float:
        u_func = _variable_impulsive_control(times, amounts, duration=pulse_duration)
        try:
            res = simulator.simulate(T=cfg.T, u_func=u_func, model="S1")
            return float(res.state[0, -1])
        except RuntimeError:
            return float(params.F_bar)

    t_start = time.perf_counter()
    opt = minimize(
        fun=lambda c: float(np.sum(c)),
        jac=lambda c: np.ones_like(c),
        x0=c_init,
        method="SLSQP",
        bounds=[(0.0, max_per_pulse)] * N,
        constraints={"type": "ineq", "fun": lambda c: eps - _F_terminal(c)},
        # eps=1.0: finite-difference step for the gradient. The default
        # (~1.5e-8) is far below the batch scale (1e4) and yields a zero
        # numerical gradient; a step of 1 individual recovers usable
        # sensitivities and gives F(T)=ε exactly (feasible).
        options={"maxiter": maxiter, "ftol": 1e-6, "eps": 1.0, "disp": False},
    )
    wall = time.perf_counter() - t_start

    best = np.clip(opt.x, 0.0, None)
    u_func = _variable_impulsive_control(times, best, duration=pulse_duration)
    result = simulator.simulate(T=cfg.T, u_func=u_func, model="S1")

    m = _metrics(result.t, result.state[0], result.control, params, eps, wall)
    m.update({
        "tau_days": tau,
        "n_pulses": N,
        "amounts_per_pulse": best.tolist(),
        "scipy_success": bool(opt.success),
        "scipy_message": str(opt.message),
    })
    logger.info(
        "Strategy 5 done | J=%.4e, F(T)=%.1f, success=%s",
        m["J_L1"], result.state[0, -1], opt.success,
    )
    return result, m


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(
    meta: dict[str, Any],
    params: BiologicalParameters,
    epsilon: float,
) -> None:
    col = 36
    header = f"{'Estrategia':<{col}} {'J (total)':>12} {'t_ε (días)':>12} {'F(T)/F̄':>10}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for key, m in meta.items():
        if not isinstance(m, dict) or "J_L1" not in m:
            continue
        t_eps = m.get("t_epsilon_days")
        t_str = f"{t_eps:.1f}" if t_eps is not None else "—"
        print(f"{key:<{col}} {m['J_L1']:>12.4e} {t_str:>12} {m['F_T_over_Fbar']:>10.4f}")
    print(f"{sep}\n")
    print(f"  Reference: F̄ = {params.F_bar:.0f},  ε = {epsilon:.0f}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure non-ASCII summary glyphs (ε, F̄) print on consoles whose default
    # encoding is not UTF-8 (e.g. Windows cp1252), which otherwise crashes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML configuration file")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Output directory")
    parser.add_argument("--tau-values", type=float, nargs="+",
                        default=[3.0, 7.0, 14.0],
                        help="Periods (days) for Strategy 3")
    parser.add_argument("--tau-optimal", type=float, default=7.0,
                        help="Release period (days) for Strategy 5")
    parser.add_argument("--c-weight", type=float, default=1.0,
                        help="Quadratic weight c for Strategy 4 L2 cost")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    params = BiologicalParameters(**cfg_dict["biological"])
    num_cfg = NumericalConfig(**cfg_dict["numerical"])
    control_cfg = ControlConfig(
        T=cfg_dict["control"]["T"],
        U_max=cfg_dict["control"]["U_max"],
        epsilon=cfg_dict["control"].get("epsilon"),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    eps = _epsilon(params, control_cfg)
    simulator = Simulator(params, num_cfg)
    optimiser = GekkoOptimiser(params, num_cfg)

    all_metrics: dict[str, Any] = {
        "_config": {
            "T": control_cfg.T,
            "U_max": control_cfg.U_max,
            "epsilon": eps,
            "F_bar": params.F_bar,
        }
    }
    # Mapping strategy label → SimulationResult (for comparison plot)
    plot_sims: dict[str, SimulationResult] = {}

    # ---- Strategy 1 -------------------------------------------------------
    opt1, m1 = run_strategy_1(optimiser, control_cfg)
    all_metrics["strategy_1_L1_optimal"] = m1
    plot_sims["Óptima $L^1$"] = _opt_to_sim(opt1, simulator, control_cfg)
    J_ref = m1["J_L1"]

    # ---- Strategy 2 -------------------------------------------------------
    sim2, m2 = run_strategy_2(simulator, control_cfg, J_ref)
    all_metrics["strategy_2_constant"] = m2
    plot_sims["Constante"] = sim2

    # ---- Strategy 3 -------------------------------------------------------
    for tau in args.tau_values:
        sim3, m3 = run_strategy_3(simulator, control_cfg, J_ref, tau)
        key = f"strategy_3_impulsive_tau{int(tau)}d"
        all_metrics[key] = m3
        plot_sims[rf"Periódica $\tau$={int(tau)}d"] = sim3

    # ---- Strategy 4 -------------------------------------------------------
    opt4, m4 = run_strategy_4(optimiser, control_cfg, c_weight=args.c_weight)
    all_metrics["strategy_4_L2_optimal"] = m4
    plot_sims["Óptima $L^2$"] = _opt_to_sim(opt4, simulator, control_cfg)

    # ---- Strategy 5 -------------------------------------------------------
    sim5, m5 = run_strategy_5(
        simulator, control_cfg, tau=args.tau_optimal, J_initial_guess=J_ref,
    )
    all_metrics[f"strategy_5_impulsive_optimal_tau{int(args.tau_optimal)}d"] = m5
    plot_sims[rf"Impulsiva óptima $\tau$={int(args.tau_optimal)}d"] = sim5

    # ---- Persist metrics --------------------------------------------------
    out_json = args.output / "strategies_metrics.json"
    out_json.write_text(json.dumps(all_metrics, indent=2, default=str))
    logger.info("Metrics saved → %s", out_json)

    # ---- Summary table ----------------------------------------------------
    _print_summary(all_metrics, params, eps)

    # ---- Comparison figure ------------------------------------------------
    fig = plot_strategy_comparison(
        results=plot_sims,
        epsilon=eps,
        save_path=args.output / "fig_comparison.pdf",
        state_index=0,
        state_label=r"Hembras $F(t)$",
    )
    plt.close(fig)
    logger.info("Comparison figure saved → %s", args.output / "fig_comparison.pdf")


if __name__ == "__main__":
    main()
