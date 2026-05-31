"""CLI: train a PPO or SAC policy for Strategy 9 (Deep RL).

Usage
-----
    python scripts/run_rl_training.py \\
        --config configs/almeida2022.yaml \\
        --algorithm PPO \\
        --timesteps 1000000 \\
        --output results/rl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sit_control.parameters import (
    BiologicalParameters,
    ControlConfig,
    load_config,
)
from sit_control.rl_env import RLConfig
from sit_control.rl_train import TrainingConfig, train

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/rl"))
    parser.add_argument(
        "--algorithm", choices=["PPO", "SAC"], default="PPO",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Parallel environments (PPO only)")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="PPO minibatch size; a larger value (e.g. 256) lowers gradient "
             "variance and stabilises training across seeds.",
    )
    parser.add_argument(
        "--lr-schedule", choices=["constant", "linear"], default="constant",
        help="PPO learning-rate schedule; 'linear' decays to 0 over training "
             "(standard PPO stabiliser, reduces seed-to-seed variance).",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.0,
        help="PPO entropy coefficient; use >0 (e.g. 0.01) to encourage "
             "exploration and avoid the 'release nothing' collapse. Ignored "
             "by SAC (entropy auto-tuned).",
    )
    parser.add_argument(
        "--terminal-penalty", type=float, default=1.0,
        help="Weight of the squared terminal-constraint violation in the "
             "reward; larger values push the agent harder toward F(T)<=eps.",
    )
    parser.add_argument(
        "--randomize-low", type=float, default=0.7,
        help="Lower factor for domain randomization (default 0.7)",
    )
    parser.add_argument(
        "--randomize-high", type=float, default=1.3,
        help="Upper factor for domain randomization (default 1.3)",
    )
    parser.add_argument(
        "--no-randomize", action="store_true",
        help="Disable domain randomization (train on nominal parameters only)",
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
        terminal_penalty=args.terminal_penalty,
    )
    train_cfg = TrainingConfig(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        lr_schedule=args.lr_schedule,
        seed=args.seed,
    )

    logger.info("=" * 70)
    logger.info("Strategy 9: Deep RL training")
    logger.info("=" * 70)
    logger.info("Algorithm:           %s", args.algorithm)
    logger.info("Timesteps:           %d", args.timesteps)
    logger.info("Domain randomization: %s", "ON" if rl_cfg.randomize else "OFF")
    if rl_cfg.randomize:
        logger.info("  range:             [%.2f, %.2f] x nominal",
                    *rl_cfg.randomize_range)
        logger.info("  params:            %s", ", ".join(rl_cfg.randomize_params))
    logger.info("Output:              %s", args.output)
    logger.info("")

    model_path = train(
        nominal_params=bio_params,
        control_config=ctrl_cfg,
        rl_config=rl_cfg,
        training_config=train_cfg,
        output_dir=args.output,
    )
    logger.info("Training complete. Policy saved to %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
