"""Unit tests for the main.py entry point (no GEKKO required)."""

from __future__ import annotations

import pytest

from main import build_parser


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------

def test_parser_has_expected_subcommands() -> None:
    """The top-level parser must expose all expected subcommands."""
    parser = build_parser()
    subparsers_actions = [
        a for a in parser._actions
        if hasattr(a, "_name_parser_map")
    ]
    assert len(subparsers_actions) == 1
    sub_map = subparsers_actions[0]._name_parser_map
    expected = {
        "verify", "convergence", "strategies", "sensitivity",
        "rl-train", "rl-eval",
    }
    assert set(sub_map.keys()) == expected


def test_rl_train_defaults() -> None:
    """'rl-train' should default to PPO + 1e6 timesteps + randomization ON."""
    parser = build_parser()
    args = parser.parse_args([
        "rl-train", "--config", "configs/almeida2022.yaml",
    ])
    assert args.algorithm == "PPO"
    assert args.timesteps == 1_000_000
    assert args.n_envs == 16
    assert args.randomize_low == pytest.approx(0.7)
    assert args.randomize_high == pytest.approx(1.3)
    assert args.no_randomize is False


def test_rl_eval_requires_model() -> None:
    """'rl-eval' must fail without --model."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["rl-eval", "--config", "c.yaml"])
    assert exc_info.value.code != 0


def test_verify_subcommand_requires_config() -> None:
    """'verify' should fail if --config is not provided."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["verify"])
    assert exc_info.value.code != 0


def test_convergence_default_N_values() -> None:
    """'convergence' should default to N in {50, 100, 200, 400}."""
    parser = build_parser()
    args = parser.parse_args(["convergence", "--config", "configs/almeida2022.yaml"])
    assert args.N_values == [50, 100, 200, 400]


def test_strategies_default_tau_values() -> None:
    """'strategies' should default to tau in {3, 7, 14}."""
    parser = build_parser()
    args = parser.parse_args(["strategies", "--config", "configs/almeida2022.yaml"])
    assert args.tau_values == [3.0, 7.0, 14.0]
    assert args.tau_optimal == 7.0
    assert args.c_weight == pytest.approx(1.0)


def test_strategies_custom_tau_optimal() -> None:
    """'strategies --tau-optimal 14' should set tau_optimal = 14."""
    parser = build_parser()
    args = parser.parse_args([
        "strategies", "--config", "c.yaml", "--tau-optimal", "14",
    ])
    assert args.tau_optimal == pytest.approx(14.0)


def test_sensitivity_default_params() -> None:
    """'sensitivity' should default to all three parameters."""
    parser = build_parser()
    args = parser.parse_args(["sensitivity", "--config", "configs/almeida2022.yaml"])
    assert set(args.params) == {"delta_F", "nu_E", "K"}


def test_sensitivity_default_levels() -> None:
    """'sensitivity' should default to ±20 % perturbation levels."""
    parser = build_parser()
    args = parser.parse_args(["sensitivity", "--config", "c.yaml"])
    assert args.levels == pytest.approx([-0.20, +0.20])


def test_sensitivity_custom_params() -> None:
    """Passing '--params delta_F K' should restrict to those two parameters."""
    parser = build_parser()
    args = parser.parse_args([
        "sensitivity", "--config", "c.yaml", "--params", "delta_F", "K",
    ])
    assert set(args.params) == {"delta_F", "K"}
    assert "nu_E" not in args.params


def test_sensitivity_invalid_param_rejected() -> None:
    """An unrecognised parameter name should cause a non-zero exit."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([
            "sensitivity", "--config", "c.yaml", "--params", "gamma_s",
        ])
    assert exc_info.value.code != 0


def test_version_flag() -> None:
    """--version should exit with code 0."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
