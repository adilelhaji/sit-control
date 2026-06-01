"""Regenerate the Chapter 5 RL figures from the sweep outputs.

Reads the aggregated per-seed CSV and the rollout JSONs produced on the server
(see scripts/rl_sweep.sh and scripts/rl_rollout.py) and produces:

  * fig_rl_tradeoff : cost vs robustness trade-off (LR-constant, 1M configs),
                      one point per terminal-penalty value, vs Almeida's optimum.
  * fig_rl_policy   : the closed-loop RL policy (3 seeds of the best config) vs
                      the open-loop bang-singular-bang optimum.

Usage:
    poetry run python scripts/rl_figures.py \
        --sweep results/rl/sweep --out Memoria/Chapter5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sit_control.bisection import (  # noqa: E402
    build_formula9_trajectory,
    solve_by_bisection,
)
from sit_control.parameters import (  # noqa: E402
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)

# best robust config (see SUMMARY.txt): tp=8, LR constant, 1M
BEST_PREFIX = "tp8_ts1M_lrconstant"
ROW_RE = re.compile(r"tp(\d+)_ts1M_lrconstant_ent[0-9.]+$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, default=Path("results/rl/sweep"))
    ap.add_argument("--config", type=Path, default=Path("configs/almeida2022.yaml"))
    ap.add_argument("--out", type=Path, default=Path("Memoria/Chapter5"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 120})

    # Almeida optimum (bisection) at nominal parameters
    d = load_config(a.config)
    p = BiologicalParameters(**d["biological"])
    num = NumericalConfig(**d["numerical"])
    cc = ControlConfig(T=150.0, U_max=d["control"]["U_max"],
                       epsilon=d["control"].get("epsilon"))
    bis = solve_by_bisection(p, cc)
    t_a, F_a, u_a = build_formula9_trajectory(p, cc, bis, num, n_eval=20000)
    eps = p.F_bar / 4.0
    J_a = float(np.trapezoid(u_a, t_a))

    # ---- Figure 1: cost-robustness trade-off ----
    agg: dict[int, dict[str, list[float]]] = {}
    with open(a.sweep / "per_seed.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = ROW_RE.match(r["config"])
            if not m:
                continue
            tp = int(m.group(1))
            agg.setdefault(tp, {"s": [], "c": []})
            agg[tp]["s"].append(float(r["dr_success"]) * 100)
            agg[tp]["c"].append(float(r["dr_cost"]))
    tps = sorted(agg)
    xs = [mean(agg[tp]["c"]) / 1e5 for tp in tps]
    ys = [mean(agg[tp]["s"]) for tp in tps]
    yerr = [pstdev(agg[tp]["s"]) for tp in tps]
    fig1, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color="#1f77b4", capsize=4, ms=7,
                lw=1.5, label="Política RL (PPO, ±30 % DR)")
    for tp, x, y in zip(tps, xs, ys):
        ax.annotate(f"  tp={tp}", (x, y), fontsize=9, va="center")
    ax.axvline(J_a / 1e5, ls="--", color="crimson", lw=1.3)
    ax.text(J_a / 1e5 + 0.03, 18, "óptimo Almeida\n(lazo abierto, frágil)",
            color="crimson", fontsize=9)
    ax.set_xlabel(r"Coste medio $J$ bajo $\pm30\%$  [$\times10^5$ mosquitos]")
    ax.set_ylabel(r"Tasa de éxito bajo $\pm30\%$  [\%]")
    ax.set_title("Compromiso coste–robustez de la política RL")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    fig1.tight_layout()
    fig1.savefig(a.out / "fig_rl_tradeoff.pdf", bbox_inches="tight")

    # ---- Figure 2: RL closed-loop policy vs Almeida ----
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, s in enumerate((0, 1, 2)):
        roll = a.sweep / f"rollout_{BEST_PREFIX}_ent0.005_s{s}.json"
        if not roll.exists():
            roll = a.sweep / f"rollout_tp8_s{s}.json"  # fallback name
        j = json.loads(roll.read_text(encoding="utf-8"))
        lbl = "Política RL (3 semillas)" if i == 0 else None
        a1.plot(j["t"], j["F"], color="#1f77b4", alpha=0.7, lw=1.2, label=lbl)
        a2.plot(j["t"], j["u"], color="#1f77b4", alpha=0.7, lw=1.2, label=lbl)
    a1.plot(t_a, F_a, color="crimson", ls="--", lw=1.8,
            label="Óptimo Almeida (lazo abierto)")
    a2.plot(t_a, u_a, color="crimson", ls="--", lw=1.8,
            label="Óptimo Almeida (lazo abierto)")
    a1.axhline(eps, ls=":", color="grey")
    a1.text(3, eps * 1.25, r"$\varepsilon=\bar F/4$", color="grey", fontsize=9)
    a1.set_xlabel("t (días)")
    a1.set_ylabel(r"Hembras $F(t)$")
    a1.set_title("Estado")
    a1.legend(fontsize=9)
    a2.set_xlabel("t (días)")
    a2.set_ylabel(r"Control $u(t)$")
    a2.set_title("Control")
    a2.legend(fontsize=9)
    fig2.suptitle("Política RL en lazo cerrado (tp=8) frente al óptimo en lazo abierto")
    fig2.tight_layout()
    fig2.savefig(a.out / "fig_rl_policy.pdf", bbox_inches="tight")
    print(f"Figuras guardadas en {a.out}  (Almeida J={J_a:.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
