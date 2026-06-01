"""Parametric sensitivity analysis for the L^1 optimal control.

Varies delta_F, nu_E, and K by +/-20 % around baseline and measures how the
optimal cost J_{L^1}(u*) and the suppression time t_epsilon change.

Design choices
--------------
* epsilon is fixed at the baseline value epsilon_0 = F_bar_0 / 4 so that all
  runs target the same absolute suppression level, enabling fair comparison.
* The initial condition is F_bar of the *perturbed* system (the natural
  persistence equilibrium for that parameter set).

Outputs
-------
results/sensitivity_results.json  -- full metric table
results/fig_tornado.pdf           -- tornado chart of Delta_J / J_0 (%)

Usage
-----
python scripts/run_sensitivity.py --config configs/almeida2022.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sit_control.metrics import suppression_time
from sit_control.optimizer import GekkoOptimiser
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Parameters to vary and their display names
SENSITIVITY_PARAMS: dict[str, str] = {
    "delta_F": r"$\delta_F$",
    "nu_E":    r"$\nu_E$",
    "K":       r"$K$",
}
PERTURBATION_LEVELS = (-0.20, +0.20)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _perturb(params: BiologicalParameters, param: str, factor: float) -> BiologicalParameters:
    """Return a copy of params with one field multiplied by (1 + factor).

    Parameters
    ----------
    params : BiologicalParameters
        Baseline parameters.
    param : str
        Name of the field to perturb (must exist in BiologicalParameters).
    factor : float
        Relative perturbation, e.g. -0.20 for a 20 % reduction.

    Returns
    -------
    BiologicalParameters
        New (frozen) parameter object with the single field perturbed.
    """
    baseline_val = getattr(params, param)
    new_val = baseline_val * (1.0 + factor)
    return dataclasses.replace(params, **{param: new_val})


def _run_one(
    params: BiologicalParameters,
    num_cfg: NumericalConfig,
    control_cfg: ControlConfig,
    fixed_epsilon: float,
) -> dict[str, Any]:
    """Solve L^1 optimal control for one parameter set.

    Parameters
    ----------
    params : BiologicalParameters
        (Possibly perturbed) biological parameters.
    num_cfg : NumericalConfig
        Numerical solver configuration.
    control_cfg : ControlConfig
        Baseline control configuration (T, U_max). epsilon is overridden
        by fixed_epsilon so all runs target the same suppression level.
    fixed_epsilon : float
        Suppression threshold fixed at baseline value.

    Returns
    -------
    dict with keys: J, t_epsilon_days, F_T, F_bar, R0, converged, wall_time_s.
    """
    cfg = ControlConfig(
        T=control_cfg.T,
        U_max=control_cfg.U_max,
        epsilon=fixed_epsilon,
    )
    optimiser = GekkoOptimiser(params, num_cfg)

    t_start = time.perf_counter()
    result = optimiser.solve_L1(cfg)
    wall = time.perf_counter() - t_start

    t_eps = suppression_time(result.t, result.F_opt, fixed_epsilon)

    return {
        "J": result.cost,
        "t_epsilon_days": t_eps,
        "F_T": float(result.F_opt[-1]),
        "F_bar": params.F_bar,
        "R0": params.R0,
        "converged": result.converged,
        "wall_time_s": round(wall, 3),
    }


def run_sensitivity(
    baseline_params: BiologicalParameters,
    num_cfg: NumericalConfig,
    control_cfg: ControlConfig,
    fixed_epsilon: float,
    params_to_vary: dict[str, str] | None = None,
    levels: tuple[float, ...] = PERTURBATION_LEVELS,
) -> dict[str, Any]:
    """Run the full sensitivity analysis.

    Parameters
    ----------
    baseline_params : BiologicalParameters
        Baseline parameter set.
    num_cfg : NumericalConfig
        Numerical configuration shared across all runs.
    control_cfg : ControlConfig
        Operational configuration (T, U_max).
    fixed_epsilon : float
        Suppression threshold held constant for all runs.
    params_to_vary : dict or None
        Mapping of field name → display label. Defaults to SENSITIVITY_PARAMS.
    levels : tuple of float
        Relative perturbation levels (e.g. (-0.20, +0.20)).

    Returns
    -------
    dict with 'baseline' and 'perturbations' sub-dicts.
    """
    if params_to_vary is None:
        params_to_vary = SENSITIVITY_PARAMS

    # --- Baseline run -------------------------------------------------------
    logger.info("Running baseline (unperturbed)")
    baseline_result = _run_one(baseline_params, num_cfg, control_cfg, fixed_epsilon)
    J0 = baseline_result["J"]
    logger.info("Baseline J0 = %.4e", J0)

    # --- Perturbation runs --------------------------------------------------
    perturbations: dict[str, Any] = {}
    for param, label in params_to_vary.items():
        baseline_val = getattr(baseline_params, param)
        perturbations[param] = {"label": label, "baseline_value": baseline_val, "runs": {}}

        for level in levels:
            pct = int(round(level * 100))
            key = f"{pct:+d}%"
            perturbed = _perturb(baseline_params, param, level)
            perturbed_val = getattr(perturbed, param)
            logger.info(
                "  %s = %.4g → %.4g (%s)",
                param, baseline_val, perturbed_val, key,
            )
            m = _run_one(perturbed, num_cfg, control_cfg, fixed_epsilon)
            m["perturbed_value"] = perturbed_val
            m["delta_J_pct"] = 100.0 * (m["J"] - J0) / J0
            perturbations[param]["runs"][key] = m

    return {"baseline": baseline_result, "perturbations": perturbations}


# ---------------------------------------------------------------------------
# Tornado chart
# ---------------------------------------------------------------------------

def plot_tornado(
    sensitivity: dict[str, Any],
    save_path: Path | str | None = None,
) -> Figure:
    """Horizontal tornado chart of Delta_J / J_0 (%).

    Each parameter occupies one horizontal band; the two bars (−20 % and
    +20 % perturbation) are drawn with distinct colours and hatching.

    Parameters
    ----------
    sensitivity : dict
        Output of :func:`run_sensitivity`.
    save_path : Path or str or None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib Figure.
    """
    perturbations = sensitivity["perturbations"]
    J0 = sensitivity["baseline"]["J"]

    param_names = list(perturbations.keys())
    n = len(param_names)

    # Collect (low_delta_pct, high_delta_pct) per parameter
    low_vals: list[float] = []
    high_vals: list[float] = []
    labels: list[str] = []

    for param in param_names:
        entry = perturbations[param]
        runs = entry["runs"]
        # Assume levels are sorted; first is negative, second is positive
        sorted_keys = sorted(runs.keys())
        vals = [runs[k]["delta_J_pct"] for k in sorted_keys]
        low_vals.append(min(vals))
        high_vals.append(max(vals))
        labels.append(entry["label"])

    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 3 + 0.6 * n))

    bar_h = 0.35
    ax.barh(
        y_pos + bar_h / 2, low_vals, height=bar_h,
        color="C0", alpha=0.75, label="−20 % perturbation",
    )
    ax.barh(
        y_pos - bar_h / 2, high_vals, height=bar_h,
        color="C3", alpha=0.75, label="+20 % perturbation",
    )

    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(r"$\Delta J / J_0$ (%)", fontsize=11)
    ax.set_title(
        r"Sensitivity of $J_{L^1}(u^*)$ to $\pm 20\,\%$ parameter perturbations",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()

    if save_path is not None:
        from sit_control.plotting import _save_figure
        _save_figure(fig, Path(save_path))

    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(sensitivity: dict[str, Any]) -> None:
    """Print the sensitivity results as a formatted table to stdout."""
    J0 = sensitivity["baseline"]["J"]
    t0 = sensitivity["baseline"]["t_epsilon_days"]
    t0_str = f"{t0:.1f}" if t0 is not None else "—"

    col = 12
    header = (
        f"{'Parameter':<14} {'Level':>6}  "
        f"{'J_L1':>{col}}  {'t_ε (d)':>{col}}  {'ΔJ/J₀ (%)':>{col}}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"{'Baseline':<14} {'':>6}  {J0:>{col}.4e}  {t0_str:>{col}}  {'0.00':>{col}}")
    print(f"{'-' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")

    for param, entry in sensitivity["perturbations"].items():
        label = entry["label"]
        for level_key, m in sorted(entry["runs"].items()):
            t_eps = m.get("t_epsilon_days")
            t_str = f"{t_eps:.1f}" if t_eps is not None else "—"
            print(
                f"{label:<14} {level_key:>6}  "
                f"{m['J']:>{col}.4e}  {t_str:>{col}}  {m['delta_J_pct']:>{col}.2f}"
            )
        print(f"{'-' * len(header)}")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML configuration file")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Output directory")
    parser.add_argument(
        "--params", nargs="+",
        default=list(SENSITIVITY_PARAMS.keys()),
        choices=list(SENSITIVITY_PARAMS.keys()),
        help="Parameters to vary (default: all three)",
    )
    parser.add_argument(
        "--levels", type=float, nargs="+",
        default=list(PERTURBATION_LEVELS),
        help="Relative perturbation levels (e.g. -0.2 0.2)",
    )
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    baseline_params = BiologicalParameters(**cfg_dict["biological"])
    num_cfg = NumericalConfig(**cfg_dict["numerical"])
    control_cfg = ControlConfig(
        T=cfg_dict["control"]["T"],
        U_max=cfg_dict["control"]["U_max"],
        epsilon=cfg_dict["control"].get("epsilon"),
    )

    # Fix epsilon at the baseline equilibrium value for all runs
    fixed_epsilon = (
        control_cfg.epsilon
        if control_cfg.epsilon is not None
        else baseline_params.F_bar / 4.0
    )
    logger.info(
        "Sensitivity analysis | epsilon_0 = %.1f, T=%g, U_max=%g",
        fixed_epsilon, control_cfg.T, control_cfg.U_max,
    )

    params_to_vary = {p: SENSITIVITY_PARAMS[p] for p in args.params}

    args.output.mkdir(parents=True, exist_ok=True)

    sensitivity = run_sensitivity(
        baseline_params=baseline_params,
        num_cfg=num_cfg,
        control_cfg=control_cfg,
        fixed_epsilon=fixed_epsilon,
        params_to_vary=params_to_vary,
        levels=tuple(args.levels),
    )

    # --- Persist -----------------------------------------------------------
    out_json = args.output / "sensitivity_results.json"
    out_json.write_text(json.dumps(sensitivity, indent=2, default=str))
    logger.info("Results saved → %s", out_json)

    # --- Print table -------------------------------------------------------
    _print_summary(sensitivity)

    # --- Tornado chart -----------------------------------------------------
    fig = plot_tornado(
        sensitivity,
        save_path=args.output / "fig_tornado.pdf",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
