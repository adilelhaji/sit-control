"""Convergence study of the optimal control solver.

Reproduces Table 4 of Almeida et al. (2022): cost functional J(u*) and
wall time as a function of the discretisation size N.

Usage:
    python scripts/run_convergence.py --config configs/almeida2022.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from sit_control.metrics import relative_error
from sit_control.optimizer import GekkoOptimiser
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)
from sit_control.plotting import plot_convergence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Reference times reported by Almeida et al. (2022), Table 4.
# Hardware: Intel Core i5, 8th generation.
REFERENCE_TIMES = {50: 2.05, 100: 6.00, 200: 29.4, 400: 113.0}
REFERENCE_COST = 1.46e5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )
    parser.add_argument(
        "--N-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400],
        help="Discretisation sizes to evaluate",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bio = BiologicalParameters(**cfg["biological"])
    control_cfg = ControlConfig(
        T=cfg["control"]["T"],
        U_max=cfg["control"]["U_max"],
        epsilon=cfg["control"].get("epsilon"),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    costs = []
    times = []

    logger.info(
        "Convergence study: T=%g, U_max=%g, N=%s",
        control_cfg.T,
        control_cfg.U_max,
        args.N_values,
    )

    for N in args.N_values:
        num_config = NumericalConfig(
            rtol=cfg["numerical"]["rtol"],
            atol=cfg["numerical"]["atol"],
            n_collocation=N,
            singular_eps=cfg["numerical"]["singular_eps"],
            solver_method=cfg["numerical"]["solver_method"],
        )
        optimiser = GekkoOptimiser(bio, num_config)
        result = optimiser.solve_L1(control_cfg)

        ref_time = REFERENCE_TIMES.get(N)
        err = relative_error(result.cost, REFERENCE_COST)

        results[f"N={N}"] = {
            "J_this_work": result.cost,
            "J_reference": REFERENCE_COST,
            "relative_error": err,
            "wall_time_s": result.wall_time,
            "reference_time_s": ref_time,
            "converged": result.converged,
        }
        costs.append(result.cost)
        times.append(result.wall_time)

        logger.info(
            "N=%4d: J=%.4e, t=%6.2fs (ref %s)",
            N,
            result.cost,
            result.wall_time,
            f"{ref_time:.2f}s" if ref_time else "n/a",
        )

    out_json = args.output / "convergence.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Numerical results written to %s", out_json)

    plot_convergence(
        N_values=np.array(args.N_values),
        costs=np.array(costs),
        times=np.array(times),
        reference_cost=REFERENCE_COST,
        save_path=args.output / "convergence.pdf",
    )
    logger.info("Convergence figure saved to %s", args.output / "convergence.pdf")


if __name__ == "__main__":
    main()
