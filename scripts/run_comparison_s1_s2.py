"""Compare S1 and S2 model trajectories: Formula (9), GEKKO S1, S2 with S1 control.

Reproduces Figure 4 of Almeida et al. (2022): three curves showing
F(t) under the optimal L1 control for T=150 days.

Produces:
  - fig_comparacion_s1_s2.pdf : F(t) trajectories — Formula(9) S1,
                                 GEKKO S1, and S2 driven by GEKKO S1 control.
  - fig_solucion_optima.pdf   : u*(t) and F*(t) from Formula (9).
  - comparison_metrics.json   : error metrics and solver diagnostics.

Workflow
--------
1. Run bisection (Algorithm 2) for the requested T → optimal (τ₁, τ₂, t₀, t₁).
2. Build formula (9) trajectory: 3-phase bang-off / singular / bang-off.
3. Solve GEKKO on S1 (NLP, direct collocation) → u_gekko(t), F_gekko_S1(t).
4. Simulate S2 driven by u_gekko(t) → F_S2(t).
5. Plot all three F(t) on the same axes.
6. Save figures and metrics JSON.

Usage:
    python scripts/run_comparison_s1_s2.py --config configs/almeida2022.yaml
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
from sit_control.plotting import _save_figure, plot_optimal_solution
from sit_control.simulator import SimulationResult, Simulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_formula9(
    params: BiologicalParameters,
    cfg: ControlConfig,
    num: NumericalConfig,
    n_eval: int = 2000,
) -> tuple[BisectionResult, np.ndarray, np.ndarray, np.ndarray]:
    """Run bisection and reconstruct the formula (9) trajectory.

    Returns:
        bis   : BisectionResult (contains tau1, tau2, t0, t1)
        t     : time grid over [0, T]
        F     : female trajectory
        u     : control profile (3 phases)
    """
    logger.info("Running bisection for T=%g ...", cfg.T)
    bis = solve_by_bisection(params, cfg)
    logger.info(
        "Bisection: tau1=%.2fd  tau2=%.2fd  t0=%.2fd  t1=%.2fd  "
        "F_min=%.2f  converged=%s",
        bis.tau1, bis.tau2, bis.t0, bis.t1, bis.F_min, bis.converged,
    )
    t, F, u = build_formula9_trajectory(params, cfg, bis, num, n_eval=n_eval)
    return bis, t, F, u


def _run_gekko_s1(
    params: BiologicalParameters,
    cfg: ControlConfig,
    num: NumericalConfig,
) -> OptimisationResult:
    """Solve the L1 problem on the reduced model S1 with GEKKO."""
    logger.info("Running GEKKO on S1 (N=%d) ...", num.n_collocation)
    opt = GekkoOptimiser(params, num)
    result = opt.solve_L1(cfg)
    logger.info(
        "GEKKO S1: J=%.4e  t=%.2fs  converged=%s",
        result.cost, result.wall_time, result.converged,
    )
    return result


def _simulate_s2_with_control(
    params: BiologicalParameters,
    cfg: ControlConfig,
    num: NumericalConfig,
    t_ctrl: np.ndarray,
    u_ctrl: np.ndarray,
    label: str = "S2",
    n_eval: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate S2 with an interpolated control signal.

    Returns:
        t_S2 : time grid
        F_S2 : female trajectory
    """
    u_func = interpolated_control(t_ctrl, u_ctrl)
    sim = Simulator(params, num)
    res = sim.simulate(T=cfg.T, u_func=u_func, model="S2", n_eval=n_eval)
    logger.info("%s simulation: F(T)=%.2f", label, res.state[2, -1])
    return res.t, res.state[2]


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_comparison(
    t_f9: np.ndarray, F_f9: np.ndarray,
    t_f9s2: np.ndarray, F_f9s2: np.ndarray,
    t_gk: np.ndarray, F_gk: np.ndarray,
    t_gks2: np.ndarray, F_gks2: np.ndarray,
    epsilon: float,
    output_path: Path,
) -> None:
    """Four-line comparison figure: bisection and GEKKO, each on S1 and S2."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(t_f9,   F_f9,   color="C0", lw=1.8,
            label=r"Bisección — $S_1$")
    ax.plot(t_f9s2, F_f9s2, color="C0", lw=1.8, linestyle="--",
            label=r"Bisección — $S_2$")
    ax.plot(t_gk,   F_gk,   color="C1", lw=1.8,
            label=r"GEKKO — $S_1$")
    ax.plot(t_gks2, F_gks2, color="C1", lw=1.8, linestyle="--",
            label=r"GEKKO — $S_2$")

    ax.axhline(epsilon, color="C3", lw=1.2, linestyle="-.",
               label=rf"$\varepsilon = {epsilon:.0f}$")

    ax.set_xlabel("Tiempo (días)", fontsize=11)
    ax.set_ylabel(r"$F(t)$ (hembras fértiles)", fontsize=11)
    ax.set_title(r"Comparación de modelos bajo el control $u^*(t)$", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0, t_f9[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_optimal_solution(
    t_f9: np.ndarray, F_f9: np.ndarray, u_f9: np.ndarray,
    t_gk: np.ndarray, F_gk: np.ndarray, u_gk: np.ndarray,
    epsilon: float,
    bis: "BisectionResult",
    output_path: Path,
) -> None:
    """Two-panel figure: F*(t) top, u*(t) bottom — matches Almeida Fig. 4 layout."""
    fig, (ax_F, ax_u) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # ── Panel superior: trayectoria F*(t) ────────────────────────────────
    ax_F.plot(t_f9, F_f9, color="C0", lw=1.8, label=r"Bisección — $S_1$")
    ax_F.plot(t_gk, F_gk, color="C1", lw=1.8, linestyle="--", label=r"GEKKO — $S_1$")
    ax_F.axhline(epsilon, color="C3", lw=1.2, linestyle="-.",
                 label=rf"$\varepsilon = {epsilon:.0f}$")
    ax_F.set_ylabel(r"$F(t)$ (hembras fértiles)", fontsize=11)
    ax_F.legend(fontsize=9, loc="upper right")
    ax_F.set_ylim(bottom=0)

    # ── Panel inferior: control u*(t) ─────────────────────────────────────
    ax_u.plot(t_f9, u_f9, color="C0", lw=1.8, label=r"Bisección")
    ax_u.plot(t_gk, u_gk, color="C1", lw=1.8, linestyle="--", label=r"GEKKO")

    # Marcar fases de la bisección
    ax_u.axvline(bis.t0, color="gray", lw=0.8, linestyle=":")
    ax_u.axvline(bis.t1, color="gray", lw=0.8, linestyle=":")
    ax_u.text(bis.t0 / 2, ax_u.get_ylim()[1] * 0.85 if ax_u.get_ylim()[1] > 0 else 100,
              r"$u=0$", ha="center", fontsize=8, color="gray")

    ax_u.set_xlabel("Tiempo (días)", fontsize=11)
    ax_u.set_ylabel(r"$u(t)$ (machos/día)", fontsize=11)
    ax_u.legend(fontsize=9, loc="upper left")
    ax_u.set_xlim(0, t_f9[-1])
    ax_u.set_ylim(bottom=0)

    fig.suptitle(r"Solución óptima $u^*(t)$ y trayectoria $F^*(t)$, $T=150\,\mathrm{d}$",
                 fontsize=12)
    fig.tight_layout()

    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, metavar="YAML")
    parser.add_argument("--output", type=Path, default=Path("results/comparison"),
                        metavar="DIR")
    parser.add_argument("--T", type=float, default=150.0,
                        help="Time horizon in days (default: 150)")
    parser.add_argument("--no-gekko", action="store_true",
                        help="Skip GEKKO (only formula-9 and S2 with formula-9 control)")
    parser.add_argument("--gekko-N", type=int, default=None, metavar="N",
                        help="Override number of GEKKO collocation nodes (default: from config)")
    parser.add_argument("--gekko-solver", type=int, default=None, choices=[1, 3],
                        help="GEKKO solver: 1=APOPT (default), 3=IPOPT")
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

    logger.info("F_bar=%.2f | epsilon=%.2f | T=%.0f | U_max=%.0f",
                bio.F_bar, epsilon, args.T, cfg.U_max)

    args.output.mkdir(parents=True, exist_ok=True)

    # ── 1. Formula (9): bisection + 3-phase trajectory ───────────────────
    bis, t_f9, F_f9, u_f9 = _run_formula9(bio, cfg, num)

    # ── 2. S2 with bisection control ─────────────────────────────────────
    t_f9s2, F_f9s2 = _simulate_s2_with_control(
        bio, cfg, num, t_f9, u_f9, label="Bisección-S2",
    )

    # ── 3. GEKKO on S1 ────────────────────────────────────────────────────
    if not args.no_gekko:
        try:
            gekko_result = _run_gekko_s1(bio, cfg, num)
            t_gk  = gekko_result.t
            F_gk  = gekko_result.F_opt
            u_gk  = gekko_result.u_opt
            t_gks2, F_gks2 = _simulate_s2_with_control(
                bio, cfg, num, t_gk, u_gk, label="GEKKO-S2",
            )
        except Exception as exc:
            logger.warning("GEKKO failed (%s); falling back to bisection.", exc)
            args.no_gekko = True

    if args.no_gekko:
        t_gk   = t_f9;   F_gk   = F_f9;   u_gk   = u_f9
        t_gks2 = t_f9s2; F_gks2 = F_f9s2

    # ── 4. Relative errors ────────────────────────────────────────────────
    F_f9s2_i = interp1d(t_f9s2, F_f9s2, kind="linear", fill_value="extrapolate")(t_f9)
    rel_bis   = np.abs(F_f9 - F_f9s2_i) / np.maximum(F_f9, 1.0)
    F_gks2_i  = interp1d(t_gks2, F_gks2, kind="linear", fill_value="extrapolate")(t_gk)
    rel_gk    = np.abs(F_gk - F_gks2_i) / np.maximum(F_gk, 1.0)
    max_rel_err  = float(np.max(rel_bis))
    mean_rel_err = float(np.mean(rel_bis))
    logger.info("Max rel error Bisección S1 vs S2: %.4f (%.2f%%)",
                max_rel_err, max_rel_err * 100)
    logger.info("Max rel error GEKKO S1 vs S2:     %.4f (%.2f%%)",
                float(np.max(rel_gk)), float(np.max(rel_gk)) * 100)

    # ── 5. fig:comparacion_s1_s2 ─────────────────────────────────────────
    path_cmp = args.output / "fig_comparacion_s1_s2.pdf"
    _plot_comparison(
        t_f9, F_f9, t_f9s2, F_f9s2,
        t_gk, F_gk, t_gks2, F_gks2,
        epsilon, path_cmp,
    )

    # ── 6. fig:solucion_optima (2 panels: F*(t) top, u*(t) bottom) ──────
    path_opt = args.output / "fig_solucion_optima.pdf"
    _plot_optimal_solution(
        t_f9, F_f9, u_f9, t_gk, F_gk, u_gk, epsilon, bis, path_opt,
    )

    # ── 7. Metrics JSON ───────────────────────────────────────────────────
    J_f9 = float(np.trapezoid(u_f9, t_f9))
    J_gk = float(np.trapezoid(u_gk, t_gk)) if not args.no_gekko else J_f9

    metrics = {
        "T": args.T,
        "F_bar": bio.F_bar,
        "epsilon": epsilon,
        "U_max": cfg.U_max,
        "bisection": {
            "tau1": bis.tau1,
            "tau2": bis.tau2,
            "t0": bis.t0,
            "t1": bis.t1,
            "F_min": bis.F_min,
            "converged": bis.converged,
            "J_formula9": J_f9,
        },
        "gekko_S1": {
            "J": J_gk,
            "converged": (not args.no_gekko),
        },
        "s1_vs_s2": {
            "max_relative_error": max_rel_err,
            "max_relative_error_pct": max_rel_err * 100,
            "mean_relative_error_pct": mean_rel_err * 100,
        },
        "figures": {
            "comparacion_s1_s2": str(path_cmp),
            "solucion_optima": str(path_opt),
        },
    }
    metrics_path = args.output / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", metrics_path)
    logger.info(
        "Summary: J*(F9)=%.4e | J*(GEKKO)=%.4e | max_err=%.2f%%",
        J_f9, J_gk, max_rel_err * 100,
    )


if __name__ == "__main__":
    main()
