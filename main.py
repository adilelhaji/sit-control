"""sit-control — Optimal control for the Sterile Insect Technique.

Unified entry point. Each subcommand delegates to the corresponding script.

Subcommands
-----------
verify      Reproduce the verification table of Almeida et al. (2022).
convergence Study J(u*) convergence as a function of GEKKO collocation size N.
strategies  Run and compare the five SIT control strategies.
sensitivity Parametric sensitivity analysis +/-20 % on delta_F, nu_E, K.
rl-train    Train the closed-loop RL policy. Requires the [rl] extras.
rl-eval     Evaluate a trained RL policy on random parameter samples.

Usage
-----
python main.py <subcommand> --config configs/almeida2022.yaml [options]
python main.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add --config and --output arguments shared by all subcommands."""
    parser.add_argument(
        "--config", type=Path, required=True,
        metavar="YAML",
        help="Path to YAML configuration file (e.g. configs/almeida2022.yaml)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        metavar="DIR",
        help="Output directory for figures and JSON (default: results/)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="sit-control",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="sit-control 0.1.0",
    )
    sub = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    sub.required = True

    # --- verify ------------------------------------------------------------
    p_verify = sub.add_parser(
        "verify",
        help="Reproduce Almeida et al. (2022) verification table",
        description="Solves L^1 optimal control for T in {60, 150, 200} days "
                    "and compares J(u*) against the reference values.",
    )
    _add_common(p_verify)

    # --- convergence -------------------------------------------------------
    p_conv = sub.add_parser(
        "convergence",
        help="Convergence study of J(u*) vs collocation size N",
        description="Reproduces Table 4 of Almeida et al. (2022).",
    )
    _add_common(p_conv)
    p_conv.add_argument(
        "--N-values", type=int, nargs="+", default=[50, 100, 200, 400],
        metavar="N",
        help="GEKKO collocation sizes to evaluate (default: 50 100 200 400)",
    )

    # --- strategies --------------------------------------------------------
    p_strat = sub.add_parser(
        "strategies",
        help="Run and compare the five SIT control strategies.",
        description="Executes strategies 1–5 and writes metrics + comparison figure.",
    )
    _add_common(p_strat)
    p_strat.add_argument(
        "--tau-values", type=float, nargs="+", default=[3.0, 7.0, 14.0],
        metavar="TAU",
        help="Release periods (days) for strategy 3 (default: 3 7 14)",
    )
    p_strat.add_argument(
        "--tau-optimal", type=float, default=7.0,
        metavar="TAU",
        help="Release period (days) for strategy 5 (default: 7)",
    )
    p_strat.add_argument(
        "--c-weight", type=float, default=1.0,
        metavar="C",
        help="Quadratic weight c for the L^2 cost functional (default: 1.0)",
    )

    # --- sensitivity -------------------------------------------------------
    p_sens = sub.add_parser(
        "sensitivity",
        help="Parametric sensitivity analysis +/-20 %% on delta_F, nu_E, K",
        description="Varies one parameter at a time and measures ΔJ/J₀.",
    )
    _add_common(p_sens)
    p_sens.add_argument(
        "--params", nargs="+",
        default=["delta_F", "nu_E", "K"],
        choices=["delta_F", "nu_E", "K"],
        help="Parameters to vary (default: delta_F nu_E K)",
    )
    p_sens.add_argument(
        "--levels", type=float, nargs="+", default=[-0.20, +0.20],
        metavar="LEVEL",
        help="Relative perturbation levels (default: -0.20 +0.20)",
    )

    # --- rl-train ----------------------------------------------------------
    p_rltrain = sub.add_parser(
        "rl-train",
        help="Train the closed-loop RL policy",
        description="Trains PPO or SAC on the SIT environment with optional "
                    "domain randomization. Requires the [rl] optional extras "
                    "(gymnasium, stable-baselines3, torch, scikit-learn).",
    )
    _add_common(p_rltrain)
    p_rltrain.add_argument(
        "--algorithm", choices=["PPO", "SAC"], default="PPO",
        help="RL algorithm (default: PPO)",
    )
    p_rltrain.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1_000_000)",
    )
    p_rltrain.add_argument(
        "--n-envs", type=int, default=16,
        help="Parallel environments for PPO (default: 16)",
    )
    p_rltrain.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Optimiser learning rate (default: 3e-4)",
    )
    p_rltrain.add_argument(
        "--seed", type=int, default=0, help="Random seed",
    )
    p_rltrain.add_argument(
        "--randomize-low", type=float, default=0.7,
        help="Lower factor for domain randomization (default: 0.7)",
    )
    p_rltrain.add_argument(
        "--randomize-high", type=float, default=1.3,
        help="Upper factor for domain randomization (default: 1.3)",
    )
    p_rltrain.add_argument(
        "--no-randomize", action="store_true",
        help="Disable domain randomization (train on nominal parameters only)",
    )

    # --- rl-eval -----------------------------------------------------------
    p_rleval = sub.add_parser(
        "rl-eval",
        help="Evaluate a trained RL policy",
        description="Runs N rollouts with random parameters from Theta and "
                    "reports mean cost, success rate, mean F(T).",
    )
    _add_common(p_rleval)
    p_rleval.add_argument(
        "--model", type=Path, required=True, metavar="ZIP",
        help="Path to the trained policy .zip file",
    )
    p_rleval.add_argument(
        "--algorithm", choices=["PPO", "SAC"], default="PPO",
        help="Algorithm used to train the model (needed to load it)",
    )
    p_rleval.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of evaluation rollouts (default: 100)",
    )
    p_rleval.add_argument(
        "--seed", type=int, default=0,
    )
    p_rleval.add_argument(
        "--randomize-low", type=float, default=0.7,
    )
    p_rleval.add_argument(
        "--randomize-high", type=float, default=1.3,
    )
    p_rleval.add_argument(
        "--no-randomize", action="store_true",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate script.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments; defaults to sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        from scripts.run_verification import main as _main
        sys.argv = [
            "run_verification",
            "--config", str(args.config),
            "--output", str(args.output),
        ]
        _main()

    elif args.command == "convergence":
        from scripts.run_convergence import main as _main
        sys.argv = [
            "run_convergence",
            "--config", str(args.config),
            "--output", str(args.output),
            "--N-values", *[str(n) for n in args.N_values],
        ]
        _main()

    elif args.command == "strategies":
        from scripts.run_strategies import main as _main
        sys.argv = [
            "run_strategies",
            "--config", str(args.config),
            "--output", str(args.output),
            "--tau-values", *[str(t) for t in args.tau_values],
            "--tau-optimal", str(args.tau_optimal),
            "--c-weight", str(args.c_weight),
        ]
        _main()

    elif args.command == "sensitivity":
        from scripts.run_sensitivity import main as _main
        sys.argv = [
            "run_sensitivity",
            "--config", str(args.config),
            "--output", str(args.output),
            "--params", *args.params,
            "--levels", *[str(lv) for lv in args.levels],
        ]
        _main()

    elif args.command == "rl-train":
        from scripts.run_rl_training import main as _main
        argv = [
            "--config", str(args.config),
            "--output", str(args.output),
            "--algorithm", args.algorithm,
            "--timesteps", str(args.timesteps),
            "--n-envs", str(args.n_envs),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
            "--randomize-low", str(args.randomize_low),
            "--randomize-high", str(args.randomize_high),
        ]
        if args.no_randomize:
            argv.append("--no-randomize")
        _main(argv)

    elif args.command == "rl-eval":
        from scripts.run_rl_evaluation import main as _main
        argv = [
            "--config", str(args.config),
            "--output", str(args.output),
            "--model", str(args.model),
            "--algorithm", args.algorithm,
            "--n-episodes", str(args.n_episodes),
            "--seed", str(args.seed),
            "--randomize-low", str(args.randomize_low),
            "--randomize-high", str(args.randomize_high),
        ]
        if args.no_randomize:
            argv.append("--no-randomize")
        _main(argv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
