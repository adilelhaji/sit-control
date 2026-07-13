"""Comparison of three SIT optimal control methods over S1 and S2.

Compared methods:
  1. Bisection (Almeida 2022) — analytical solution via PMP + Algorithm 2
  2. FBS (Forward-Backward Sweep) — direct numerical PMP iteration
  3. GEKKO (ramp) — direct NLP with ramp warm-start
  4. GEKKO (bisection warm-start) — direct NLP initialized from u*(bisection)

Produces:
  - fig_comparacion_metodos.pdf  : F(t) of the 4 methods on S1 and S2
  - fig_controles_optimos.pdf    : u*(t) of the 4 methods
  - fig_solucion_optima.pdf      : 2 panels F*(t) and u*(t) (bisection vs GEKKO)
  - comparison_metrics.json      : cost, error and convergence metrics

Usage:
    python scripts/run_comparison_s1_s2.py --config configs/almeida2022.yaml
    python scripts/run_comparison_s1_s2.py --config configs/almeida2022.yaml --no-gekko
    python scripts/run_comparison_s1_s2.py --config configs/almeida2022.yaml --no-fbs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from sit_control.bisection import (
    BisectionResult,
    build_formula9_trajectory,
    solve_by_bisection,
)
from sit_control.controls import interpolated_control
from sit_control.optimizer import GekkoOptimiser, OptimisationResult
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)
from sit_control.plotting import _save_figure
from sit_control.pmp_sweep import FBSweepResult, ForwardBackwardSweep
from sit_control.simulator import Simulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────


def _run_bisection(
    params: BiologicalParameters,
    cfg: ControlConfig,
    num: NumericalConfig,
    n_eval: int = 2000,
) -> tuple[BisectionResult, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Bisection: T=%g ...", cfg.T)
    bis = solve_by_bisection(params, cfg)
    logger.info(
        "Bisection: tau1=%.2fd  tau2=%.2fd  t0=%.2fd  t1=%.2fd  "
        "F_min=%.2f  converged=%s",
        bis.tau1,
        bis.tau2,
        bis.t0,
        bis.t1,
        bis.F_min,
        bis.converged,
    )
    t, F, u = build_formula9_trajectory(params, cfg, bis, num, n_eval=n_eval)
    return bis, t, F, u


def _run_fbs(
    params: BiologicalParameters,
    cfg: ControlConfig,
    num: NumericalConfig,
    u_init: tuple[np.ndarray, np.ndarray] | None = None,
    n_iter: int = 400,
    alpha: float = 0.5,
    n_grid: int = 500,
) -> FBSweepResult:
    label = "FBS (bisection warm-start)" if u_init is not None else "FBS (ramp)"
    logger.info("Starting %s ...", label)
    fbs = ForwardBackwardSweep(params, num)
    result = fbs.solve_L1(
        cfg,
        u_init=u_init,
        n_iter=n_iter,
        alpha=alpha,
        n_grid=n_grid,
    )
    logger.info(
        "%s: J=%.4e  t=%.2fs  F(T)=%.2f  iters=%d  converged=%s",
        label,
        result.cost,
        result.wall_time,
        result.F_terminal,
        result.n_iterations,
        result.converged,
    )
    return result


def _run_gekko(
    params: BiologicalParameters,
    cfg: ControlConfig,
    num: NumericalConfig,
    u_init: tuple[np.ndarray, np.ndarray] | None = None,
) -> OptimisationResult:
    opt = GekkoOptimiser(params, num)
    if u_init is not None:
        logger.info("GEKKO (bisection warm-start, N=%d) ...", num.n_collocation)
        result = opt.solve_L1_warmstart(cfg, u_init)
        label = "GEKKO warm-start"
    else:
        logger.info("GEKKO (ramp, N=%d) ...", num.n_collocation)
        result = opt.solve_L1(cfg)
        label = "GEKKO ramp"
    logger.info(
        "%s: J=%.4e  t=%.2fs  converged=%s",
        label,
        result.cost,
        result.wall_time,
        result.converged,
    )
    return result


def _simulate_s2(
    params: BiologicalParameters,
    cfg: ControlConfig,
    num: NumericalConfig,
    t_ctrl: np.ndarray,
    u_ctrl: np.ndarray,
    label: str,
    n_eval: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    u_func = interpolated_control(t_ctrl, u_ctrl)
    sim = Simulator(params, num)
    res = sim.simulate(T=cfg.T, u_func=u_func, model="S2", n_eval=n_eval)
    logger.info("%s S2: F(T)=%.2f", label, res.state[2, -1])
    return res.t, res.state[2]


def _rel_error_s1_s2(
    t_s1: np.ndarray,
    F_s1: np.ndarray,
    t_s2: np.ndarray,
    F_s2: np.ndarray,
) -> tuple[float, float]:
    F_s2_i = interp1d(t_s2, F_s2, kind="linear", fill_value="extrapolate")(t_s1)
    rel = np.abs(F_s1 - F_s2_i) / np.maximum(F_s1, 1.0)
    return float(np.max(rel)), float(np.mean(rel))


# ── plotting ──────────────────────────────────────────────────────────────────


def _plot_comparison_methods(
    results: dict,
    epsilon: float,
    output_path: Path,
) -> None:
    """F(t) of all methods on S1, with S2 as dashed lines."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"bis": "C0", "fbs": "C2", "gk_ramp": "C1", "gk_ws": "C3"}
    labels_s1 = {
        "bis": r"Bisection — $S_1$",
        "fbs": r"FBS — $S_1$",
        "gk_ramp": r"GEKKO (ramp) — $S_1$",
        "gk_ws": r"GEKKO (WS) — $S_1$",
    }
    labels_s2 = {
        "bis": r"Bisection — $S_2$",
        "fbs": r"FBS — $S_2$",
        "gk_ramp": r"GEKKO (ramp) — $S_2$",
        "gk_ws": r"GEKKO (WS) — $S_2$",
    }

    for key, color in colors.items():
        if key not in results:
            continue
        t_s1, F_s1 = results[key]["t_s1"], results[key]["F_s1"]
        t_s2, F_s2 = results[key]["t_s2"], results[key]["F_s2"]
        ax.plot(t_s1, F_s1, color=color, lw=1.8, label=labels_s1[key])
        ax.plot(t_s2, F_s2, color=color, lw=1.4, linestyle="--", label=labels_s2[key])

    ax.axhline(
        epsilon,
        color="k",
        lw=1.1,
        linestyle="-.",
        label=rf"$\varepsilon = {epsilon:.0f}$",
    )

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel(r"$F(t)$ (fertile females)", fontsize=11)
    ax.set_title(
        r"Comparison of methods under the optimal control $u^*(t)$", fontsize=12
    )
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_xlim(0, results[next(iter(results))]["t_s1"][-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_controls(
    results: dict,
    bis: BisectionResult,
    output_path: Path,
) -> None:
    """u*(t) of all methods."""
    fig, ax = plt.subplots(figsize=(9, 4))

    colors = {"bis": "C0", "fbs": "C2", "gk_ramp": "C1", "gk_ws": "C3"}
    labels = {
        "bis": r"Bisection",
        "fbs": r"FBS (PMP)",
        "gk_ramp": r"GEKKO (ramp)",
        "gk_ws": r"GEKKO (WS bisection)",
    }

    for key, color in colors.items():
        if key not in results:
            continue
        t_u, u = results[key]["t_u"], results[key]["u"]
        ax.plot(t_u, u, color=color, lw=1.6, label=labels[key])

    ax.axvline(
        bis.t0, color="gray", lw=0.8, linestyle=":", label=f"$t_0={bis.t0:.1f}$d"
    )
    ax.axvline(
        bis.t1, color="gray", lw=0.8, linestyle="--", label=f"$t_1={bis.t1:.1f}$d"
    )

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel(r"$u(t)$ (sterile males/day)", fontsize=11)
    ax.set_title(
        r"Optimal control profile $u^*(t)$ — comparison of methods", fontsize=12
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, results[next(iter(results))]["t_u"][-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_optimal_solution(
    t_f9: np.ndarray,
    F_f9: np.ndarray,
    u_f9: np.ndarray,
    t_gk: np.ndarray,
    F_gk: np.ndarray,
    u_gk: np.ndarray,
    epsilon: float,
    bis: BisectionResult,
    output_path: Path,
) -> None:
    """2-panel figure: F*(t) top, u*(t) bottom."""
    fig, (ax_F, ax_u) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax_F.plot(t_f9, F_f9, color="C0", lw=1.8, label=r"Bisection — $S_1$")
    ax_F.plot(t_gk, F_gk, color="C1", lw=1.8, linestyle="--", label=r"GEKKO — $S_1$")
    ax_F.axhline(
        epsilon,
        color="C3",
        lw=1.2,
        linestyle="-.",
        label=rf"$\varepsilon = {epsilon:.0f}$",
    )
    ax_F.set_ylabel(r"$F(t)$ (fertile females)", fontsize=11)
    ax_F.legend(fontsize=9, loc="upper right")
    ax_F.set_ylim(bottom=0)

    ax_u.plot(t_f9, u_f9, color="C0", lw=1.8, label=r"Bisection")
    ax_u.plot(t_gk, u_gk, color="C1", lw=1.8, linestyle="--", label=r"GEKKO")
    ax_u.axvline(bis.t0, color="gray", lw=0.8, linestyle=":")
    ax_u.axvline(bis.t1, color="gray", lw=0.8, linestyle=":")

    ax_u.set_xlabel("Time (days)", fontsize=11)
    ax_u.set_ylabel(r"$u(t)$ (males/day)", fontsize=11)
    ax_u.legend(fontsize=9, loc="upper left")
    ax_u.set_xlim(0, t_f9[-1])
    ax_u.set_ylim(bottom=0)

    fig.suptitle(
        r"Optimal solution $u^*(t)$ and trajectory $F^*(t)$, $T=150\,\mathrm{d}$",
        fontsize=12,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, metavar="YAML")
    parser.add_argument(
        "--output", type=Path, default=Path("results/comparison"), metavar="DIR"
    )
    parser.add_argument(
        "--T", type=float, default=150.0, help="Time horizon in days (default: 150)"
    )
    parser.add_argument("--no-gekko", action="store_true", help="Skip GEKKO")
    parser.add_argument(
        "--no-fbs", action="store_true", help="Skip Forward-Backward Sweep"
    )
    parser.add_argument(
        "--gekko-N",
        type=int,
        default=None,
        metavar="N",
        help="Number of GEKKO collocation nodes",
    )
    parser.add_argument(
        "--gekko-solver",
        type=int,
        default=None,
        choices=[1, 3],
        help="GEKKO solver: 1=APOPT (default), 3=IPOPT",
    )
    parser.add_argument(
        "--fbs-iter",
        type=int,
        default=400,
        help="Maximum FBS iterations (default: 400)",
    )
    parser.add_argument(
        "--fbs-alpha",
        type=float,
        default=0.5,
        help="FBS blending factor (default: 0.5)",
    )
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    bio = BiologicalParameters(**cfg_dict["biological"])
    num_kw = dict(cfg_dict["numerical"])
    if args.gekko_N is not None:
        num_kw["n_collocation"] = args.gekko_N
    if args.gekko_solver is not None:
        num_kw["gekko_solver"] = args.gekko_solver
    num = NumericalConfig(**num_kw)
    epsilon = bio.F_bar / 4.0
    cfg = ControlConfig(T=args.T, U_max=cfg_dict["control"]["U_max"])

    logger.info(
        "F_bar=%.2f | epsilon=%.2f | T=%.0f | U_max=%.0f",
        bio.F_bar,
        epsilon,
        args.T,
        cfg.U_max,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    # Stores results per method: {key: {t_s1, F_s1, t_s2, F_s2, t_u, u}}
    results: dict = {}
    metrics: dict = {
        "T": args.T,
        "F_bar": bio.F_bar,
        "epsilon": epsilon,
        "U_max": cfg.U_max,
    }

    # ── 1. Bisection ─────────────────────────────────────────────────────────
    bis, t_bis, F_bis, u_bis = _run_bisection(bio, cfg, num)
    t_bis_s2, F_bis_s2 = _simulate_s2(bio, cfg, num, t_bis, u_bis, "Bisection")
    results["bis"] = {
        "t_s1": t_bis,
        "F_s1": F_bis,
        "t_s2": t_bis_s2,
        "F_s2": F_bis_s2,
        "t_u": t_bis,
        "u": u_bis,
    }
    J_bis = float(np.trapezoid(u_bis, t_bis))
    max_err_bis, mean_err_bis = _rel_error_s1_s2(t_bis, F_bis, t_bis_s2, F_bis_s2)
    logger.info("Bisection: J=%.4e | max_err_S1vS2=%.2f%%", J_bis, max_err_bis * 100)
    metrics["bisection"] = {
        "tau1": bis.tau1,
        "tau2": bis.tau2,
        "t0": bis.t0,
        "t1": bis.t1,
        "F_min": bis.F_min,
        "converged": bis.converged,
        "J": J_bis,
        "max_rel_err_s1_s2_pct": max_err_bis * 100,
    }

    # ── 2. FBS (Forward-Backward Sweep) ──────────────────────────────────────
    if not args.no_fbs:
        try:
            fbs_result = _run_fbs(
                bio,
                cfg,
                num,
                u_init=(t_bis, u_bis),  # warm-start with bisection
                n_iter=args.fbs_iter,
                alpha=args.fbs_alpha,
            )
            t_fbs_s2, F_fbs_s2 = _simulate_s2(
                bio,
                cfg,
                num,
                fbs_result.t,
                fbs_result.u_opt,
                "FBS",
            )
            results["fbs"] = {
                "t_s1": fbs_result.t,
                "F_s1": fbs_result.F_opt,
                "t_s2": t_fbs_s2,
                "F_s2": F_fbs_s2,
                "t_u": fbs_result.t,
                "u": fbs_result.u_opt,
            }
            max_err_fbs, mean_err_fbs = _rel_error_s1_s2(
                fbs_result.t,
                fbs_result.F_opt,
                t_fbs_s2,
                F_fbs_s2,
            )
            logger.info(
                "FBS: J=%.4e | max_err_S1vS2=%.2f%%", fbs_result.cost, max_err_fbs * 100
            )
            metrics["fbs"] = {
                "J": fbs_result.cost,
                "wall_time": fbs_result.wall_time,
                "n_iterations": fbs_result.n_iterations,
                "converged": fbs_result.converged,
                "F_terminal": fbs_result.F_terminal,
                "max_rel_err_s1_s2_pct": max_err_fbs * 100,
            }
        except Exception as exc:
            logger.warning("FBS failed (%s); omitting.", exc)
            args.no_fbs = True

    # ── 3. GEKKO (ramp warm-start) ────────────────────────────────────────────
    if not args.no_gekko:
        try:
            gk_ramp = _run_gekko(bio, cfg, num, u_init=None)
            t_gkr_s2, F_gkr_s2 = _simulate_s2(
                bio,
                cfg,
                num,
                gk_ramp.t,
                gk_ramp.u_opt,
                "GEKKO-ramp",
            )
            results["gk_ramp"] = {
                "t_s1": gk_ramp.t,
                "F_s1": gk_ramp.F_opt,
                "t_s2": t_gkr_s2,
                "F_s2": F_gkr_s2,
                "t_u": gk_ramp.t,
                "u": gk_ramp.u_opt,
            }
            max_err_gkr, _ = _rel_error_s1_s2(
                gk_ramp.t, gk_ramp.F_opt, t_gkr_s2, F_gkr_s2
            )
            metrics["gekko_ramp"] = {
                "J": gk_ramp.cost,
                "wall_time": gk_ramp.wall_time,
                "converged": gk_ramp.converged,
                "max_rel_err_s1_s2_pct": max_err_gkr * 100,
            }

            # ── 4. GEKKO (bisection warm-start) ──────────────────────────────
            gk_ws = _run_gekko(bio, cfg, num, u_init=(t_bis, u_bis))
            t_gkws_s2, F_gkws_s2 = _simulate_s2(
                bio,
                cfg,
                num,
                gk_ws.t,
                gk_ws.u_opt,
                "GEKKO-WS",
            )
            results["gk_ws"] = {
                "t_s1": gk_ws.t,
                "F_s1": gk_ws.F_opt,
                "t_s2": t_gkws_s2,
                "F_s2": F_gkws_s2,
                "t_u": gk_ws.t,
                "u": gk_ws.u_opt,
            }
            max_err_gkws, _ = _rel_error_s1_s2(
                gk_ws.t, gk_ws.F_opt, t_gkws_s2, F_gkws_s2
            )
            metrics["gekko_warmstart"] = {
                "J": gk_ws.cost,
                "wall_time": gk_ws.wall_time,
                "converged": gk_ws.converged,
                "max_rel_err_s1_s2_pct": max_err_gkws * 100,
            }
            logger.info(
                "GEKKO ramp J=%.4e | GEKKO WS J=%.4e",
                gk_ramp.cost,
                gk_ws.cost,
            )

        except Exception as exc:
            logger.warning("GEKKO failed (%s); omitting.", exc)
            args.no_gekko = True

    # ── 5. Cost summary table ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("COST SUMMARY J*(u)")
    logger.info("  Bisection :  J = %.4e", J_bis)
    if "fbs" in metrics:
        logger.info(
            "  FBS       :  J = %.4e  (F(T)=%.1f)",
            metrics["fbs"]["J"],
            metrics["fbs"]["F_terminal"],
        )
    if "gekko_ramp" in metrics:
        logger.info("  GEKKO ramp:  J = %.4e", metrics["gekko_ramp"]["J"])
    if "gekko_warmstart" in metrics:
        logger.info("  GEKKO WS  :  J = %.4e", metrics["gekko_warmstart"]["J"])
    logger.info("=" * 60)

    # ── 6. Figures ────────────────────────────────────────────────────────────
    _plot_comparison_methods(
        results,
        epsilon,
        args.output / "fig_comparacion_metodos.pdf",
    )
    _plot_controls(
        results,
        bis,
        args.output / "fig_controles_optimos.pdf",
    )

    # Classic bisection vs GEKKO figure (for the thesis)
    t_gk = gk_ramp.t if "gk_ramp" in results else t_bis
    F_gk = gk_ramp.F_opt if "gk_ramp" in results else F_bis
    u_gk = gk_ramp.u_opt if "gk_ramp" in results else u_bis
    _plot_optimal_solution(
        t_bis,
        F_bis,
        u_bis,
        t_gk,
        F_gk,
        u_gk,
        epsilon,
        bis,
        args.output / "fig_solucion_optima.pdf",
    )

    # ── 7. JSON ───────────────────────────────────────────────────────────────
    metrics_path = args.output / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
