"""CLI: evaluate a trained RL policy for Strategy 9.

Two modes
---------
1. Robustness evaluation (default): N=100 episodes with random parameters
   uniformly drawn from Theta = [0.7, 1.3] x theta_nominal.

2. K-Fold cross-validation (--kfold): partition Theta into K=5 folds,
   train K policies (one per fold), and aggregate metrics. Expensive
   (~K * 2-4 h of CPU); used to certify policy generalization.

Usage
-----
    python scripts/run_rl_evaluation.py \\
        --config configs/almeida2022.yaml \\
        --model results/rl/ppo_policy.zip \\
        --n-episodes 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    load_config,
)
from sit_control.rl_env import RLConfig
from sit_control.rl_evaluate import evaluate_policy

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to the trained policy .zip file")
    parser.add_argument("--output", type=Path, default=Path("results/rl"))
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algorithm", choices=["PPO", "SAC"], default="PPO",
        help="Algorithm used to train the model (needed to load it)",
    )
    parser.add_argument(
        "--randomize-low", type=float, default=0.7,
    )
    parser.add_argument(
        "--randomize-high", type=float, default=1.3,
    )
    parser.add_argument(
        "--no-randomize", action="store_true",
        help="Evaluate on nominal parameters only (no robustness test)",
    )
    parser.add_argument(
        "--kfold", action="store_true",
        help="Run K-Fold cross-validation (expensive, trains K models)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    cfg = load_config(args.config)
    bio_params = BiologicalParameters(**cfg.get("biological", {}))
    ctrl_cfg = ControlConfig(**cfg["control"])
    rl_cfg = RLConfig(
        randomize=not args.no_randomize,
        randomize_range=(args.randomize_low, args.randomize_high),
    )

    if args.kfold:
        logger.error(
            "K-Fold cross-validation requires retraining K=5 policies; "
            "use the kfold_evaluate() function directly from rl_evaluate.py "
            "and supply your own train_fn. Skipping."
        )
        return 1

    if args.algorithm == "PPO":
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
    else:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)

    logger.info("=" * 70)
    logger.info("Strategy 9: Deep RL evaluation")
    logger.info("=" * 70)
    logger.info("Model:               %s", args.model)
    logger.info("N episodes:          %d", args.n_episodes)
    logger.info("Randomize:           %s", "ON" if rl_cfg.randomize else "OFF")
    logger.info("")

    result = evaluate_policy(
        model=model,
        nominal_params=bio_params,
        control_config=ctrl_cfg,
        rl_config=rl_cfg,
        n_episodes=args.n_episodes,
        deterministic=True,
        seed=args.seed,
    )

    logger.info("Results")
    logger.info("-------")
    logger.info("Mean cost J:         %.4e  (std %.4e)",
                result.mean_cost, result.std_cost)
    logger.info("Success rate:        %.1f %%  (F(T) <= %.0f)",
                100.0 * result.success_rate, result.epsilon)
    logger.info("Mean F(T):           %.1f", result.mean_F_terminal)

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "rl_evaluation.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved evaluation to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
