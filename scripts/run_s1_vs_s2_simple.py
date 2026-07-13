"""Simple S1 vs S2 trajectory comparison under the bisection optimal control.

Produces a clean two-curve figure validating the quasi-stationary approximation
of Almeida et al. (2022), Section 2.2 / Proposition 2.3: applying the optimal
control u*(t) computed on the reduced system S1 to the full system S2 yields
trajectories that are visually indistinguishable.

This script is intentionally minimal: it runs ONLY the bisection optimum (no
FBS, no GEKKO) and plots F_S1(t) vs F_S2(t) — exactly the message needed for
Section 3.4.2 of the TFG verification chapter.

Outputs:
    figures/fig_s1_vs_s2.pdf       : single-panel comparison
    figures/fig_s1_vs_s2.png       : PNG copy
    figures/s1_vs_s2_metrics.json  : numerical metrics

Usage:
    python scripts/run_s1_vs_s2_simple.py --config configs/almeida2022.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sit_control.bisection import build_formula9_trajectory, solve_by_bisection
from sit_control.controls import interpolated_control
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)
from sit_control.plotting import _save_figure
from sit_control.simulator import Simulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _interp_F(
    t_source: np.ndarray, F_source: np.ndarray, t_target: np.ndarray
) -> np.ndarray:
    """Linear interpolation of F onto a target time grid (clipped at boundaries)."""
    return np.interp(t_target, t_source, F_source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, metavar="YAML")
    parser.add_argument("--output", type=Path, default=Path("figures"), metavar="DIR")
    parser.add_argument("--T", type=float, default=150.0, help="Time horizon (days).")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    bio = BiologicalParameters(**cfg_dict["biological"])
    num = NumericalConfig(**cfg_dict["numerical"])
    cfg = ControlConfig(T=args.T, U_max=cfg_dict["control"]["U_max"])

    epsilon = bio.F_bar / 4.0
    logger.info(
        "F_bar=%.2f | epsilon=%.2f | T=%.0f | U_max=%.0f | K=%.0f",
        bio.F_bar,
        epsilon,
        args.T,
        cfg.U_max,
        bio.K,
    )

    # ── 1. Solve optimum on S1 by bisection (Algorithm 2 of Almeida 2022) ────
    logger.info("Solving bisection on S1 ...")
    bis = solve_by_bisection(bio, cfg)
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

    # Reconstruct the full u*(t) profile via the 3-phase formula (9)
    t_s1, F_s1, u_star = build_formula9_trajectory(bio, cfg, bis, num, n_eval=2000)
    u_func = interpolated_control(t_s1, u_star)

    # ── 2. Apply the same u*(t) to the FULL system S2 ────────────────────────
    logger.info("Applying u*(S1) to the full system S2 ...")
    sim = Simulator(bio, num)
    res_s2 = sim.simulate(T=args.T, u_func=u_func, model="S2")
    t_s2 = res_s2.t
    F_s2 = res_s2.state[2]  # F is index 2 in [E, M, F, Ms]

    # ── 3. Quantify the S1 vs S2 discrepancy ─────────────────────────────────
    # Resample S1 onto the S2 grid for pointwise comparison
    F_s1_on_s2 = _interp_F(t_s1, F_s1, t_s2)
    abs_err = np.abs(F_s1_on_s2 - F_s2)
    rel_err_pct = 100.0 * abs_err / np.maximum(F_s1_on_s2, 1e-9)
    max_rel = float(rel_err_pct.max())
    mean_rel = float(rel_err_pct.mean())
    F_s1_T = float(F_s1[-1])
    F_s2_T = float(F_s2[-1])

    logger.info(
        "Comparison: F_S1(T)=%.2f  F_S2(T)=%.2f  max_rel=%.3f%%  mean_rel=%.3f%%",
        F_s1_T,
        F_s2_T,
        max_rel,
        mean_rel,
    )

    # ── 4. Plot: two trajectories, one panel ─────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(t_s1, F_s1, color="C0", lw=2.2, label=r"$F_{S_1}(t)$")
    ax.plot(t_s2, F_s2, color="C3", lw=2.2, linestyle="--", label=r"$F_{S_2}(t)$")
    ax.axhline(
        epsilon,
        color="0.35",
        lw=1.0,
        linestyle=":",
        label=rf"$\varepsilon = \bar{{F}}/4 = {epsilon:.0f}$",
    )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fertile females $F(t)$")
    ax.set_xlim(0, args.T)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_title(
        rf"$F(t)$ under $u^*(t)$ (bisection): $S_1$ vs. $S_2$  —  "
        rf"max. rel. error $= {max_rel:.2f}\,\%$"
    )
    fig.tight_layout()

    _save_figure(fig, args.output / "fig_s1_vs_s2.pdf")
    plt.close(fig)

    # ── 5. Persist metrics ───────────────────────────────────────────────────
    metrics = {
        "config": str(args.config),
        "T": args.T,
        "K": bio.K,
        "F_bar": bio.F_bar,
        "epsilon": epsilon,
        "bisection": {
            "tau1_days": bis.tau1,
            "tau2_days": bis.tau2,
            "t0_days": bis.t0,
            "t1_days": bis.t1,
            "F_min": bis.F_min,
            "converged": bis.converged,
        },
        "comparison": {
            "F_S1_terminal": F_s1_T,
            "F_S2_terminal": F_s2_T,
            "max_rel_error_pct": max_rel,
            "mean_rel_error_pct": mean_rel,
        },
    }
    out_json = args.output / "s1_vs_s2_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", out_json)


if __name__ == "__main__":
    main()
