"""Additional unit tests for rl_evaluate covering the K-Fold and sampling paths.

These tests use a lightweight DUMMY policy (a constant admissible action) so
NO training is required: they stay fast and UNMARKED. Only gymnasium is needed
(for SITEnv); Stable-Baselines3 / PyTorch are not imported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sklearn")

from sit_control.parameters import BiologicalParameters, ControlConfig  # noqa: E402
from sit_control.rl_env import RLConfig  # noqa: E402
from sit_control.rl_evaluate import (  # noqa: E402
    EvaluationResult,
    _evaluate_on_pool,
    evaluate_policy,
    kfold_evaluate,
    sample_theta_grid,
)


class _ConstantPolicy:
    """Mock SB3 model releasing a constant admissible action u = 0.5."""

    def predict(
        self,
        obs: Any,  # noqa: ARG002
        deterministic: bool = True,  # noqa: ARG002
    ) -> tuple[np.ndarray, None]:
        return np.array([0.5], dtype=np.float32), None


# ---------------------------------------------------------------------------
# EvaluationResult.to_dict
# ---------------------------------------------------------------------------


def test_evaluation_result_to_dict_roundtrip() -> None:
    """to_dict must serialise every dataclass field."""
    result = EvaluationResult(
        n_episodes=2,
        mean_cost=1.0,
        std_cost=0.1,
        success_rate=0.5,
        mean_F_terminal=3.0,
        epsilon=2.0,
        costs=[1.0, 1.0],
        F_terminals=[3.0, 3.0],
        sampled_params=[{"beta_E": 10.0}],
    )
    d = result.to_dict()
    assert d["n_episodes"] == 2
    assert d["success_rate"] == pytest.approx(0.5)
    assert d["costs"] == [1.0, 1.0]
    assert d["sampled_params"] == [{"beta_E": 10.0}]


# ---------------------------------------------------------------------------
# sample_theta_grid
# ---------------------------------------------------------------------------


def test_sample_theta_grid_shape_and_bounds() -> None:
    """Sampled grid must have shape (n_samples, n_params) within the range."""
    params = BiologicalParameters()
    randomize_params = ("beta_E", "delta_F", "nu_E", "K")
    lo, hi = 0.7, 1.3
    n_samples = 50

    grid = sample_theta_grid(
        params,
        randomize_params,
        (lo, hi),
        n_samples,
        seed=0,
    )

    assert grid.shape == (n_samples, len(randomize_params))
    for j, name in enumerate(randomize_params):
        nominal = getattr(params, name)
        assert np.all(grid[:, j] >= lo * nominal)
        assert np.all(grid[:, j] <= hi * nominal)


def test_sample_theta_grid_is_deterministic_for_fixed_seed() -> None:
    """The same seed must reproduce the same grid."""
    params = BiologicalParameters()
    args = (params, ("beta_E", "nu_E"), (0.8, 1.2), 10)
    grid_a = sample_theta_grid(*args, seed=42)
    grid_b = sample_theta_grid(*args, seed=42)
    assert np.array_equal(grid_a, grid_b)


# ---------------------------------------------------------------------------
# evaluate_policy with a constant policy
# ---------------------------------------------------------------------------


def test_evaluate_policy_constant_action_structure() -> None:
    """A constant-action policy must produce a coherent EvaluationResult."""
    params = BiologicalParameters()
    ctrl = ControlConfig(T=5.0, U_max=5000.0)
    rl_cfg = RLConfig(dt=1.0, randomize=True, fixed_epsilon=True)

    result = evaluate_policy(
        _ConstantPolicy(),
        params,
        ctrl,
        rl_cfg,
        n_episodes=5,
        seed=0,
    )

    assert result.n_episodes == 5
    assert len(result.costs) == 5
    assert len(result.F_terminals) == 5
    assert len(result.sampled_params) == 5
    assert 0.0 <= result.success_rate <= 1.0
    assert np.isfinite(result.mean_cost)
    assert np.isfinite(result.std_cost)
    assert all(np.isfinite(c) for c in result.costs)


# ---------------------------------------------------------------------------
# _evaluate_on_pool (both epsilon branches)
# ---------------------------------------------------------------------------


def test_evaluate_on_pool_explicit_epsilon() -> None:
    """With an explicit epsilon, that value must be reported unchanged."""
    params = BiologicalParameters()
    ctrl = ControlConfig(T=5.0, U_max=5000.0, epsilon=1234.0)
    rl_cfg = RLConfig(dt=1.0)
    pool = sample_theta_grid(
        params, rl_cfg.randomize_params, rl_cfg.randomize_range, 8, seed=0
    )

    result = _evaluate_on_pool(
        _ConstantPolicy(),
        params,
        ctrl,
        rl_cfg,
        pool,
        n_episodes=4,
        seed=0,
    )

    assert isinstance(result, EvaluationResult)
    assert result.epsilon == pytest.approx(1234.0)
    assert result.n_episodes == 4
    assert len(result.costs) == 4
    assert 0.0 <= result.success_rate <= 1.0
    assert result.sampled_params == []


def test_evaluate_on_pool_default_epsilon_from_F_bar() -> None:
    """With epsilon=None, the threshold must fall back to F_bar / 4."""
    params = BiologicalParameters()
    ctrl = ControlConfig(T=5.0, U_max=5000.0)  # epsilon None
    rl_cfg = RLConfig(dt=1.0)
    pool = sample_theta_grid(
        params, rl_cfg.randomize_params, rl_cfg.randomize_range, 8, seed=1
    )

    result = _evaluate_on_pool(
        _ConstantPolicy(),
        params,
        ctrl,
        rl_cfg,
        pool,
        n_episodes=3,
        seed=5,
    )

    assert result.epsilon == pytest.approx(params.F_bar / 4.0)


# ---------------------------------------------------------------------------
# kfold_evaluate end-to-end (dummy train_fn)
# ---------------------------------------------------------------------------


def test_kfold_evaluate_structure_and_persistence(tmp_path: Path) -> None:
    """kfold_evaluate must return per-fold + aggregate metrics and write JSON."""

    def train_fn(train_pool: Any) -> _ConstantPolicy:  # noqa: ARG001
        return _ConstantPolicy()

    params = BiologicalParameters()
    ctrl = ControlConfig(T=5.0, U_max=5000.0)
    rl_cfg = RLConfig(dt=1.0)

    summary = kfold_evaluate(
        train_fn,
        params,
        ctrl,
        rl_config=rl_cfg,
        n_folds=2,
        n_samples=20,
        n_eval_per_fold=3,
        seed=0,
        output_dir=tmp_path,
    )

    assert summary["n_folds"] == 2
    assert summary["n_samples"] == 20
    assert len(summary["per_fold"]) == 2
    for fold in summary["per_fold"]:
        assert 0.0 <= fold["success_rate"] <= 1.0
        assert np.isfinite(fold["mean_cost"])
        assert fold["n_episodes"] == 3

    agg = summary["aggregate"]
    assert np.isfinite(agg["mean_cost"])
    assert 0.0 <= agg["mean_success_rate"] <= 1.0

    out_path = tmp_path / "kfold_results.json"
    assert out_path.exists()


def test_kfold_evaluate_without_output_dir_skips_write() -> None:
    """When output_dir is None, kfold_evaluate must not raise and return a summary."""

    def train_fn(train_pool: Any) -> _ConstantPolicy:  # noqa: ARG001
        return _ConstantPolicy()

    summary = kfold_evaluate(
        train_fn,
        BiologicalParameters(),
        ControlConfig(T=5.0, U_max=5000.0),
        rl_config=RLConfig(dt=1.0),
        n_folds=2,
        n_samples=12,
        n_eval_per_fold=2,
        seed=1,
        output_dir=None,
    )

    assert summary["n_folds"] == 2
    assert "aggregate" in summary
