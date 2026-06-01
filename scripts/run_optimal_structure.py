"""Optimal-control structure plot for the verification chapter (Sec. 3.4.3).

Resolves the optimal control of Almeida et al. (2022) by the bisection
algorithm (Algorithm 2, Theorem 3.3) and produces a two-panel figure
illustrating the bang-singular-bang structure:

    Top panel:    F*(t), with horizontal line at the suppression target
                  epsilon = F_bar/4 and vertical lines at t0 (start of
                  singular arc) and t1 (end).
    Bottom panel: u*(t), showing the three phases u=0, u_sing, u=0.

This script is intentionally minimal — it runs ONLY bisection, no FBS
and no GEKKO — to keep the verification message focused on whether the
implementation reproduces the structural prediction of Theorem 3.3.

Outputs:
    figures/fig_solucion_optima.pdf       : two-panel figure
    figures/fig_solucion_optima.png       : PNG copy
    figures/estructura_optima_metrics.json: switching times and J*

Usage:
    python scripts/run_estructura_optima.py --config configs/almeida2022.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sit_control.bisection import build_formula9_trajectory, solve_by_bisection
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)
from sit_control.plotting import _save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        bio.F_bar, epsilon, args.T, cfg.U_max, bio.K,
    )

    # ── 1. Solve bisection (Algorithm 2 of Almeida 2022) ─────────────────────
    logger.info("Solving bisection on S1 ...")
    bis = solve_by_bisection(bio, cfg)
    logger.info(
        "Bisection: tau1=%.2fd  tau2=%.2fd  t0=%.2fd  t1=%.2fd  F_min=%.2f  converged=%s",
        bis.tau1, bis.tau2, bis.t0, bis.t1, bis.F_min, bis.converged,
    )

    # ── 2. Reconstruct u*(t) and F*(t) via the formula (9) ───────────────────
    t, F, u = build_formula9_trajectory(bio, cfg, bis, num, n_eval=2000)

    # Trapezoidal cost on the singular arc [t0, t1]
    mask_sing = (t >= bis.t0) & (t <= bis.t1)
    J_star = float(np.trapezoid(u[mask_sing], t[mask_sing]))
    logger.info("J* = integral(u_sing on [t0, t1]) = %.4e", J_star)

    # ── 3. Plot: two stacked panels ──────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    fig, (ax_F, ax_u) = plt.subplots(
        2, 1, figsize=(8.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.85], "hspace": 0.08},
    )

    # Top: F*(t) ─────────────────────────────────────────────────────────────
    ax_F.plot(t, F, color="C0", lw=2.2, label=r"$F^*(t)$")
    ax_F.axhline(epsilon, color="0.35", lw=1.0, linestyle=":",
                 label=fr"$\varepsilon = \bar{{F}}/4 = {epsilon:.0f}$")
    ax_F.axvline(bis.t0, color="0.55", lw=0.9, linestyle="--")
    ax_F.axvline(bis.t1, color="0.55", lw=0.9, linestyle="--")
    ax_F.set_ylabel("Fertile females $F^*(t)$")
    ax_F.set_ylim(bottom=0)
    ax_F.grid(True, alpha=0.3)
    ax_F.legend(loc="upper right", framealpha=0.95)
    ax_F.set_title(
        fr"Optimal solution by bisection: $T={args.T:.0f}$ d, "
        fr"$\tau_1={bis.tau1:.2f}$ d, $\tau_2={bis.tau2:.2f}$ d, "
        fr"$J^*={J_star:.3e}$"
    )

    # Bottom: u*(t) ─────────────────────────────────────────────────────────
    ax_u.plot(t, u, color="C3", lw=2.2, label=r"$u^*(t)$")
    ax_u.axhline(cfg.U_max, color="0.55", lw=0.8, linestyle=":",
                 label=fr"$U_{{\max}}={cfg.U_max:.0f}$")
    ax_u.axvline(bis.t0, color="0.55", lw=0.9, linestyle="--")
    ax_u.axvline(bis.t1, color="0.55", lw=0.9, linestyle="--")

    # Annotate phases on the u panel
    y_mid = 0.55 * cfg.U_max
    ax_u.annotate(
        r"$u^*\!=\!0$", xy=(bis.t0 / 2.0, y_mid),
        ha="center", va="center", fontsize=11, color="0.30",
    )
    ax_u.annotate(
        r"singular arc", xy=((bis.t0 + bis.t1) / 2.0, 0.78 * cfg.U_max),
        ha="center", va="center", fontsize=11, color="0.30",
    )
    ax_u.annotate(
        r"$u^*\!=\!0$", xy=((bis.t1 + args.T) / 2.0, y_mid),
        ha="center", va="center", fontsize=11, color="0.30",
    )

    # Vertical labels for t0, t1
    ax_u.text(bis.t0, -0.10 * cfg.U_max, fr"$t_0={bis.t0:.1f}$",
              ha="center", va="top", fontsize=9, color="0.30")
    ax_u.text(bis.t1, -0.10 * cfg.U_max, fr"$t_1={bis.t1:.1f}$",
              ha="center", va="top", fontsize=9, color="0.30")

    ax_u.set_xlabel("Time (days)")
    ax_u.set_ylabel(r"Control $u^*(t)$ (males/day)")
    ax_u.set_xlim(0, args.T)
    ax_u.set_ylim(0, 1.1 * cfg.U_max)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(loc="upper left", framealpha=0.95)

    fig.tight_layout()
    _save_figure(fig, args.output / "fig_solucion_optima.pdf")
    plt.close(fig)

    # ── 4. Persist metrics ───────────────────────────────────────────────────
    metrics = {
        "config": str(args.config),
        "T": args.T,
        "K": bio.K,
        "F_bar": bio.F_bar,
        "epsilon": epsilon,
        "U_max": cfg.U_max,
        "bisection": {
            "tau1_days": bis.tau1,
            "tau2_days": bis.tau2,
            "t0_days": bis.t0,
            "t1_days": bis.t1,
            "F_min": bis.F_min,
            "iterations": bis.iterations,
            "converged": bis.converged,
        },
        "J_star": J_star,
    }
    out_json = args.output / "estructura_optima_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", out_json)


if __name__ == "__main__":
    main()
