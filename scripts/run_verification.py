"""Reproduce the verification table of the TFG (Chapter 3).

Usage:
    python scripts/run_verification.py --config configs/almeida2022.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sit_control.metrics import relative_error
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

REFERENCE_VALUES = {
    "S1": {60: 1.46e5, 150: 1.46e5, 200: 1.46e5},
    "S2": {60: 1.47e5, 150: 1.47e5, 200: 1.47e5},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    bio = BiologicalParameters(**cfg["biological"])
    num = NumericalConfig(**cfg["numerical"])

    args.output.mkdir(parents=True, exist_ok=True)
    results = {}

    for T in (60.0, 150.0, 200.0):
        control_cfg = ControlConfig(T=T, U_max=cfg["control"]["U_max"])
        optimiser = GekkoOptimiser(bio, num)
        result = optimiser.solve_L1(control_cfg)

        ref = REFERENCE_VALUES["S1"][int(T)]
        err = relative_error(result.cost, ref)

        results[f"T={T}"] = {
            "J_this_work": result.cost,
            "J_reference": ref,
            "relative_error": err,
            "wall_time_s": result.wall_time,
            "converged": result.converged,
        }
        logger.info("T=%g: J=%.4e (ref=%.4e, err=%.2e)",
                    T, result.cost, ref, err)

    out_path = args.output / "verification.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()