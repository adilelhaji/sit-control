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
from dataclasses import replace
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
    j_l1: float | None = None,
) -> dict[str, Any]:
    """Standard performance metrics for one strategy.

    For impulsive strategies pass ``j_l1`` = the sum of released amounts, which
    is the exact integral of the rectangular-pulse control. The trapezoidal
    integral of ``u`` sampled on the coarse output grid overestimates narrow
    pulses (boundary effect, up to +2.4% for tau=14 d), so it must not be used
    as the reported cost for impulsive strategies.
    """
    t_eps = suppression_time(t, F, epsilon)
    return {
        "J_L1": cost_L1(t, u) if j_l1 is None else float(j_l1),
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
    pulse_duration = 0.5
    # Cap the integrator step to half the pulse width so the adaptive RK45
    # cannot under-resolve the narrow pulses (the default max_step=inf can step
    # over them, see Strategy 6).
    capped_sim = Simulator(
        simulator.params,
        replace(simulator.config, max_step=pulse_duration / 2.0),
    )
    t0 = time.perf_counter()
    result = capped_sim.simulate(
        T=cfg.T,
        u_func=impulsive_control(times, amount, duration=pulse_duration),
        model="S1",
    )
    wall = time.perf_counter() - t0
    eps = _epsilon(simulator.params, cfg)
    # Exact L1 cost = sum of released amounts (= J_reference by construction),
    # not the trapezoid of u on the coarse grid (which overestimates pulses).
    m = _metrics(
        result.t, result.state[0], result.control, simulator.params, eps, wall,
        j_l1=J_reference,
    )
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

    # Cap the integrator step to half the pulse width so the adaptive RK45 does
    # not under-resolve the narrow pulses. Used in every constraint evaluation
    # below, so the SLSQP optimises against an accurate F(T).
    capped_sim = Simulator(params, replace(simulator.config,
                                           max_step=pulse_duration / 2.0))

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
            res = capped_sim.simulate(T=cfg.T, u_func=u_func, model="S1")
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
    result = capped_sim.simulate(T=cfg.T, u_func=u_func, model="S1")

    # Exact L1 cost = sum of optimised amounts (the SLSQP objective itself),
    # not the trapezoid of u on the coarse grid.
    m = _metrics(
        result.t, result.state[0], result.control, params, eps, wall,
        j_l1=float(np.sum(best)),
    )
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
# Strategy 6: impulsivized optimal
# ---------------------------------------------------------------------------

def _impulsivize_optimal(
    t_star: NDArray[np.float64],
    u_star: NDArray[np.float64],
    T: float,
    tau: float,
    placement: str = "start",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lump the continuous optimum u* into one impulse per tau-interval.

    Partition [0, T] into intervals of length ``tau`` and assign each interval a
    single impulse whose mass equals the optimal cost spent there:
        c_k = integral over the interval of u*(t) dt.
    Intervals where u* == 0 (the waiting phases) get c_k = 0, so by construction
    Σ c_k = J_L1(u*) and the result is compared with the continuous optimum at
    EQUAL budget.

    Returns
    -------
    (times, amounts) : impulse start/centre times and masses c_k.
    """
    edges = np.arange(0.0, T - 1e-9, tau)
    edges = np.append(edges, T)  # close the last (possibly partial) interval
    times, amounts = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (t_star >= a) & (t_star <= b)
        c_k = (float(np.trapezoid(u_star[mask], t_star[mask]))
               if mask.sum() > 1 else 0.0)
        times.append(a if placement == "start" else 0.5 * (a + b))
        amounts.append(max(c_k, 0.0))
    return np.asarray(times), np.asarray(amounts)


def run_strategy_6(
    simulator: Simulator,
    cfg: ControlConfig,
    opt1: OptimisationResult,
    tau: float = 7.0,
    placement: str = "start",
    pulse_duration: float = 0.5,
) -> tuple[SimulationResult, dict[str, Any]]:
    """Strategy 6: impulsivized optimal.

    Take the continuous optimum u*(t) from Strategy 1 and replace it by one
    impulse per ``tau``-interval carrying the optimal cost of that interval
    (c_k = integral of u* over the interval). The total budget equals J_L1(u*),
    so the strategy is compared with the continuous optimum at equal cost: it
    shows whether discretising the *optimal* effort still suppresses, and how
    close it lands to the SLSQP per-pulse optimisation (Strategy 5).
    """
    params = simulator.params
    eps = _epsilon(params, cfg)
    times, amounts = _impulsivize_optimal(
        opt1.t, opt1.u_opt, cfg.T, tau, placement
    )
    n_pulses = int((amounts > 0).sum())
    logger.info(
        "Strategy 6 — impulsivized optimal | tau=%.0f d, %d pulses, placement=%s",
        tau, n_pulses, placement,
    )
    u_func = _variable_impulsive_control(times, amounts, duration=pulse_duration)
    # Cap the integrator step to half the pulse width: with sparse impulses
    # (tau=7, 14) preceded by the long u*=0 initial phase, an unbounded adaptive
    # RK45 step jumps over the 0.5-day pulses and falsely reports F(T)=F_bar.
    # The simulator documents this requirement (max_step <= pulse_width / 2).
    capped_sim = Simulator(params, replace(simulator.config,
                                           max_step=pulse_duration / 2.0))
    result = capped_sim.simulate(T=cfg.T, u_func=u_func, model="S1")
    m = _metrics(
        result.t, result.state[0], result.control, params, eps,
        j_l1=float(np.sum(amounts)),
    )
    m.update({
        "tau_days": tau,
        "n_pulses": n_pulses,
        "placement": placement,
    })
    logger.info(
        "Strategy 6 done | J=%.4e, F(T)/Fbar=%.4f",
        m["J_L1"], m["F_T_over_Fbar"],
    )
    return result, m


# ---------------------------------------------------------------------------
# Strategy 5 initialisation sweep (non-convexity range)
# ---------------------------------------------------------------------------

def run_strategy_5_init_sweep(
    simulator: Simulator,
    cfg: ControlConfig,
    J_ref: float,
    scales: list[float],
    tau: float = 7.0,
    maxiter: int = 300,
) -> dict[str, Any]:
    """Initialisation sweep for the optimal-impulsive SLSQP (Section 4.5).

    Runs Strategy 5 from several uniform initial guesses (``scale * J_ref``) to
    characterise the non-convexity of the impulsive problem: it reports the
    spread of the converged cost across initialisations, expressed as a
    percentage over the continuous optimum J_ref.
    """
    runs: list[dict[str, Any]] = []
    for s in scales:
        logger.info("Init sweep | scale=%.2f (J0=%.0f)", s, s * J_ref)
        _, m = run_strategy_5(
            simulator, cfg, tau=tau, J_initial_guess=s * J_ref, maxiter=maxiter,
        )
        runs.append({
            "scale": s,
            "J_L1": m["J_L1"],
            "F_T_over_Fbar": m["F_T_over_Fbar"],
            "scipy_success": m["scipy_success"],
        })
    js = [r["J_L1"] for r in runs]
    j_min, j_max = min(js), max(js)
    summary = {
        "tau_days": tau,
        "J_ref": J_ref,
        "runs": runs,
        "J_min": j_min,
        "J_max": j_max,
        "pct_over_optimum_min": 100.0 * (j_min / J_ref - 1.0),
        "pct_over_optimum_max": 100.0 * (j_max / J_ref - 1.0),
    }
    print(f"\nStrategy 5 initialisation sweep (tau={tau:.0f} d, J_ref={J_ref:.0f})")
    print(f"{'scale':>6} {'J':>12} {'F(T)/Fbar':>10} {'% over opt':>11} "
          f"{'success':>8}")
    for r in runs:
        print(f"{r['scale']:>6.2f} {r['J_L1']:>12.0f} {r['F_T_over_Fbar']:>10.4f} "
              f"{100.0 * (r['J_L1'] / J_ref - 1.0):>10.1f}% "
              f"{str(r['scipy_success']):>8}")
    print(f"  Range over optimum: +{summary['pct_over_optimum_min']:.1f}% .. "
          f"+{summary['pct_over_optimum_max']:.1f}%\n")
    return summary


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(
    meta: dict[str, Any],
    params: BiologicalParameters,
    epsilon: float,
) -> None:
    col = 36
    header = f"{'Strategy':<{col}} {'J (total)':>12} {'t_ε (days)':>12} {'F(T)/F̄':>10}"
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
    parser.add_argument("--strategies", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6], choices=[1, 2, 3, 4, 5, 6],
                        help="Which strategies to run. Strategy 1 always runs "
                             "(it provides u* and J_ref for the others). "
                             "Metrics merge into the existing JSON, so a partial "
                             "run does not erase the others.")
    parser.add_argument("--s5-sweep", type=float, nargs="*", default=None,
                        metavar="SCALE",
                        help="Run ONLY the Strategy 5 initialisation sweep "
                             "(non-convexity range). Optionally pass the uniform "
                             "init scales (x J_ref); defaults to 1.0 1.5 2.0. "
                             "Each point is a full SLSQP run, so this is slow.")
    parser.add_argument("--s6-figure", action="store_true",
                        help="Run ONLY Strategy 1 + Strategy 6 and save "
                             "results/fig_impulsivizacion.pdf: F(t) of the "
                             "continuous optimum vs the impulsivized optimum for "
                             "each tau. Fast (no SLSQP).")
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

    run = set(args.strategies)

    # ---- Strategy 1 (always runs: provides u* and J_ref) ------------------
    opt1, m1 = run_strategy_1(optimiser, control_cfg)
    all_metrics["strategy_1_L1_optimal"] = m1
    plot_sims["Optimal $L^1$"] = _opt_to_sim(opt1, simulator, control_cfg)
    J_ref = m1["J_L1"]

    # ---- Strategy 5 initialisation sweep — runs alone and exits -----------
    if args.s5_sweep is not None:
        scales = args.s5_sweep if args.s5_sweep else [1.0, 1.5, 2.0]
        sweep = run_strategy_5_init_sweep(
            simulator, control_cfg, J_ref, scales, tau=args.tau_optimal,
        )
        sweep_path = args.output / "s5_init_sweep.json"
        sweep_path.write_text(json.dumps(sweep, indent=2, default=str))
        logger.info("Init sweep saved → %s", sweep_path)
        return

    # ---- Strategy 6 figure — runs alone and exits ------------------------
    if args.s6_figure:
        sims = {"Óptima continua $L^1$": _opt_to_sim(opt1, simulator, control_cfg)}
        for tau in args.tau_values:
            sim6, _ = run_strategy_6(simulator, control_cfg, opt1, tau=tau)
            sims[rf"Impulsivizada $\tau$={int(tau)}d"] = sim6
        fig_path = args.output / "fig_impulsivizacion.pdf"
        fig = plot_strategy_comparison(
            results=sims, epsilon=eps, save_path=fig_path,
            state_index=0, state_label=r"Females $F(t)$",
        )
        plt.close(fig)
        logger.info("Impulsivization figure saved → %s", fig_path)
        return

    # ---- Strategy 2 -------------------------------------------------------
    if 2 in run:
        sim2, m2 = run_strategy_2(simulator, control_cfg, J_ref)
        all_metrics["strategy_2_constant"] = m2
        plot_sims["Constant"] = sim2

    # ---- Strategy 3 -------------------------------------------------------
    if 3 in run:
        for tau in args.tau_values:
            sim3, m3 = run_strategy_3(simulator, control_cfg, J_ref, tau)
            key = f"strategy_3_impulsive_tau{int(tau)}d"
            all_metrics[key] = m3
            plot_sims[rf"Periodic $\tau$={int(tau)}d"] = sim3

    # ---- Strategy 4 -------------------------------------------------------
    if 4 in run:
        opt4, m4 = run_strategy_4(optimiser, control_cfg, c_weight=args.c_weight)
        all_metrics["strategy_4_L2_optimal"] = m4
        plot_sims["Optimal $L^2$"] = _opt_to_sim(opt4, simulator, control_cfg)

    # ---- Strategy 5 -------------------------------------------------------
    if 5 in run:
        sim5, m5 = run_strategy_5(
            simulator, control_cfg, tau=args.tau_optimal, J_initial_guess=J_ref,
        )
        all_metrics[f"strategy_5_impulsive_optimal_tau{int(args.tau_optimal)}d"] = m5
        plot_sims[rf"Optimal impulsive $\tau$={int(args.tau_optimal)}d"] = sim5

    # ---- Strategy 6: impulsivized optimal --------------------------------
    # Reuses the u* already computed in Strategy 1 (opt1); records metrics only,
    # the comparison figure is left unchanged.
    if 6 in run:
        for tau in args.tau_values:
            _, m6 = run_strategy_6(simulator, control_cfg, opt1, tau=tau)
            all_metrics[f"strategy_6_impulsivized_optimal_tau{int(tau)}d"] = m6

    # ---- Persist metrics (merge with existing JSON, non-destructive) ------
    out_json = args.output / "strategies_metrics.json"
    merged: dict[str, Any] = {}
    if out_json.is_file():
        try:
            merged = json.loads(out_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged.update(all_metrics)
    out_json.write_text(json.dumps(merged, indent=2, default=str))
    logger.info("Metrics saved (merged) → %s", out_json)

    # ---- Summary table ----------------------------------------------------
    _print_summary(merged, params, eps)

    # ---- Comparison figure (only on a full run, to protect the thesis fig) -
    if run >= {2, 3, 4, 5}:
        fig = plot_strategy_comparison(
            results=plot_sims,
            epsilon=eps,
            save_path=args.output / "fig_comparison.pdf",
            state_index=0,
            state_label=r"Females $F(t)$",
        )
        plt.close(fig)
        logger.info("Comparison figure saved → %s", args.output / "fig_comparison.pdf")
    else:
        logger.info("Partial run %s: comparison figure not regenerated.",
                    sorted(run))


if __name__ == "__main__":
    main()
