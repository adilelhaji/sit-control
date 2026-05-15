"""Reproduce Table 2 of Almeida et al. (2022): solver cross-validation.

Compares J(u*) obtained by two independent solvers against the reference
values of Almeida et al. (2022), Table 2:

  - Bisection (Algorithm 2): exploits the bang-singular structure of
    Theorem 3.3 and finds the global optimum. Primary comparison with
    Almeida's reference. Only valid for T > T_opt ≈ 103 d (zero→singular
    structure); T=60 ≈ T_min uses bang-ON→singular and is skipped.
  - GEKKO/APOPT (NLP): local solver included for completeness and
    timing comparison (Almeida 2022, Table 4).

Parallelism: bisection and GEKKO for each T are independent subtasks,
giving 5 concurrent workers (bisection×2 + GEKKO×3). The pool uses
all available CPUs minus one.

Usage:
    python scripts/run_verification.py --config configs/almeida2022.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm

from sit_control.bisection import BisectionResult, singular_control, solve_by_bisection
from sit_control.controls import constant_control
from sit_control.metrics import relative_error
from sit_control.model import recruitment
from sit_control.optimizer import GekkoOptimiser
from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    NumericalConfig,
    load_config,
)
from sit_control.simulator import Simulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Reference values from Almeida et al. (2022), Table 2.
REFERENCE_VALUES = {
    "S1": {60: 1.46e5, 150: 1.46e5, 200: 1.46e5},
    "S2": {60: 1.47e5, 150: 1.47e5, 200: 1.47e5},
}

# T=60 ≈ T_min: bang-ON→singular structure; bisection (zero→singular) not valid.
BISECTION_HORIZONS = {150.0, 200.0}


# ---------------------------------------------------------------------------
# Cost helper
# ---------------------------------------------------------------------------

def _cost_from_bisection(
    params: BiologicalParameters,
    cfg: ControlConfig,
    bis_result: BisectionResult,
    num: NumericalConfig,
) -> float:
    """Compute J* = integral of u_sing on [t0, T] from a bisection result."""
    from scipy.integrate import solve_ivp

    T = cfg.T
    t0 = bis_result.t0
    initial = np.array([params.F_bar, 0.0], dtype=np.float64)

    state_t0 = initial.copy()
    if t0 > 1e-10:
        sim = Simulator(params, num)
        res1 = sim.simulate(T=t0, u_func=constant_control(0.0),
                            model="S1", initial_state=initial)
        state_t0 = res1.state[:, -1].copy()

    if bis_result.tau1 < 1e-10:
        return 0.0

    def _rhs(t: float, state: np.ndarray) -> np.ndarray:
        F_s = float(max(state[0], 0.0))
        Ms_s = float(max(state[1], 0.0))
        u_s = float(np.clip(singular_control(F_s, Ms_s, params), 0.0, cfg.U_max))
        dF = float(recruitment(F_s, Ms_s, params, singular_eps=num.singular_eps))
        dMs = u_s - params.delta_s * Ms_s
        return np.array([dF, dMs], dtype=np.float64)

    n_eval = max(500, int(bis_result.tau1 * 10))
    sol = solve_ivp(fun=_rhs, t_span=(t0, T),
                    y0=state_t0, method=num.solver_method,
                    t_eval=np.linspace(t0, T, n_eval),
                    rtol=num.rtol, atol=num.atol)

    u_grid = np.array([
        float(np.clip(singular_control(
            float(max(sol.y[0, i], 0.0)),
            float(max(sol.y[1, i], 0.0)), params),
            0.0, cfg.U_max))
        for i in range(sol.y.shape[1])
    ])
    return float(np.trapezoid(u_grid, sol.t))


# ---------------------------------------------------------------------------
# Worker functions  (one per subtask — serialisable for multiprocessing)
# ---------------------------------------------------------------------------

def _worker_bisection(T: float, cfg_dict: dict) -> dict:
    """Subtask: run bisection for horizon T."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [BIS T={int(T):3d}d] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    bio = BiologicalParameters(**cfg_dict["biological"])
    num = NumericalConfig(**cfg_dict["numerical"])
    cfg = ControlConfig(T=T, U_max=cfg_dict["control"]["U_max"])

    log.info("Starting bisection...")
    t0 = time.perf_counter()
    bis = solve_by_bisection(bio, cfg)
    J = _cost_from_bisection(bio, cfg, bis, num)
    wall = time.perf_counter() - t0
    err = relative_error(J, REFERENCE_VALUES["S1"][int(T)])

    log.info("Done — J=%.4e  err=%.2e  t=%.2fs  converged=%s  tau1=%.1fd",
             J, err, wall, bis.converged, bis.tau1)

    return {
        "solver": "bisection", "T": T, "J": J,
        "relative_error": err, "wall_time_s": wall,
        "converged": bis.converged, "tau1_days": bis.tau1,
        "t0_days": bis.t0, "F_min": bis.F_min,
    }


def _worker_gekko(T: float, cfg_dict: dict) -> dict:
    """Subtask: run GEKKO for horizon T."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GEK T={int(T):3d}d] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    bio = BiologicalParameters(**cfg_dict["biological"])
    num = NumericalConfig(**cfg_dict["numerical"])
    cfg = ControlConfig(T=T, U_max=cfg_dict["control"]["U_max"])

    log.info("Starting GEKKO (N=%d)...", num.n_collocation)
    result = GekkoOptimiser(bio, num).solve_L1(cfg)
    err = relative_error(result.cost, REFERENCE_VALUES["S1"][int(T)])

    log.info("Done — J=%.4e  err=%.2e  t=%.2fs  converged=%s",
             result.cost, err, result.wall_time, result.converged)

    return {
        "solver": "gekko", "T": T, "J": result.cost,
        "relative_error": err, "wall_time_s": result.wall_time,
        "converged": result.converged,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, metavar="YAML")
    parser.add_argument("--output", type=Path, default=Path("results"), metavar="DIR")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    bio = BiologicalParameters(**cfg_dict["biological"])

    # Build subtask list: bisection(150), bisection(200), gekko(60/150/200)
    subtasks: list[tuple[Literal["bisection", "gekko"], float]] = []
    for T in (60.0, 150.0, 200.0):
        if T in BISECTION_HORIZONS:
            subtasks.append(("bisection", T))
        subtasks.append(("gekko", T))
    # subtasks = [bisection(150), bisection(200), gekko(60), gekko(150), gekko(200)]

    n_cpus = max(1, (os.cpu_count() or 1) - 1)
    n_workers = min(n_cpus, len(subtasks))

    logger.info(
        "F_bar=%.2f | epsilon=%.2f | subtasks=%d | workers=%d / %d CPUs",
        bio.F_bar, bio.F_bar / 4, len(subtasks), n_workers, os.cpu_count() or 1,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # Collect raw results keyed by (solver, T)
    raw: dict[tuple[str, int], dict] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for solver, T in subtasks:
            fn = _worker_bisection if solver == "bisection" else _worker_gekko
            futures[pool.submit(fn, T, cfg_dict)] = (solver, T)

        with tqdm(total=len(subtasks), desc="Subtasks", unit="task",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                  ) as pbar:
            for fut in as_completed(futures):
                solver, T = futures[fut]
                label = f"{solver[:3].upper()} T={int(T)}d"
                try:
                    raw[(solver, int(T))] = fut.result()
                    pbar.set_postfix_str(f"{label} ✓")
                except Exception as exc:
                    logger.error("%s failed: %s", label, exc)
                    raw[(solver, int(T))] = {"error": str(exc)}
                    pbar.set_postfix_str(f"{label} ✗")
                pbar.update(1)

    # Assemble final results dict sorted by T
    results: dict[str, dict] = {}
    for T in (60, 150, 200):
        ref = REFERENCE_VALUES["S1"][T]
        entry: dict = {"T": float(T), "J_reference_S1": ref}
        if ("bisection", T) in raw:
            entry["bisection"] = raw[("bisection", T)]
        if ("gekko", T) in raw:
            entry["gekko"] = raw[("gekko", T)]
        results[f"T={T}"] = entry

    out_path = args.output / "verification.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
