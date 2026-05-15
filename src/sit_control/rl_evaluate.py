"""Evaluation of trained RL policies for Strategy 9.

Provides two evaluation protocols:

1. **Robustness evaluation** (``evaluate_policy``): runs a single trained
   policy on ``n_episodes`` rollouts with biological parameters sampled from
   the full domain Theta. Reports mean cost, success rate (F(T) <= epsilon),
   and per-parameter conditional metrics.

2. **K-Fold cross-validation** (``kfold_evaluate``): partitions the
   parameter manifold into K folds via ``sklearn.model_selection.KFold``,
   trains a policy on K-1 folds, and evaluates it on the held-out fold.
   Quantifies generalization OUTSIDE the training distribution.

References
----------
Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
    J. Mach. Learn. Res. 12, 2825-2830.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .parameters import BiologicalParameters, ControlConfig
from .rl_env import RLConfig, SITEnv

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregated metrics of a policy evaluation."""

    n_episodes: int
    mean_cost: float
    std_cost: float
    success_rate: float
    mean_F_terminal: float
    epsilon: float
    costs: list[float] = field(default_factory=list)
    F_terminals: list[float] = field(default_factory=list)
    sampled_params: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_policy(
    model: Any,
    nominal_params: BiologicalParameters,
    control_config: ControlConfig,
    rl_config: RLConfig | None = None,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 0,
) -> EvaluationResult:
    """Evaluate a trained policy on n_episodes rollouts with random parameters.

    Parameters
    ----------
    model
        A trained SB3 model with a ``predict(obs, deterministic) -> (action, _)`` method.
    nominal_params, control_config, rl_config
        Standard environment configuration. ``rl_config.randomize`` controls
        whether each episode samples a new parameter set.
    n_episodes
        Number of evaluation rollouts.
    deterministic
        If True, uses the mean of the policy distribution (recommended for
        evaluation; stochastic exploration is only useful during training).
    seed
        Base seed; episode k uses seed ``seed + k`` for reproducibility.

    Returns
    -------
    EvaluationResult with per-episode and aggregate metrics.
    """
    env = SITEnv(nominal_params, control_config, rl_config)
    costs: list[float] = []
    F_terminals: list[float] = []
    sampled: list[dict[str, float]] = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        params = info["params"]
        sampled.append(
            {name: getattr(params, name) for name in (rl_config or RLConfig()).randomize_params}
        )

        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            if truncated:
                break

        costs.append(env.cumulative_cost)
        F_terminals.append(info["F"])

    epsilon = env.epsilon
    success_rate = float(np.mean([F <= epsilon for F in F_terminals]))

    return EvaluationResult(
        n_episodes=n_episodes,
        mean_cost=float(np.mean(costs)),
        std_cost=float(np.std(costs, ddof=1)) if n_episodes > 1 else 0.0,
        success_rate=success_rate,
        mean_F_terminal=float(np.mean(F_terminals)),
        epsilon=float(epsilon),
        costs=costs,
        F_terminals=F_terminals,
        sampled_params=sampled,
    )


def sample_theta_grid(
    nominal_params: BiologicalParameters,
    randomize_params: tuple[str, ...],
    randomize_range: tuple[float, float],
    n_samples: int,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Sample n_samples points from Theta uniformly (per-axis).

    Returns
    -------
    Array of shape (n_samples, len(randomize_params)).
    """
    rng = np.random.default_rng(seed)
    lo, hi = randomize_range
    nominal_values = np.array(
        [getattr(nominal_params, p) for p in randomize_params],
        dtype=np.float64,
    )
    return rng.uniform(
        low=lo * nominal_values,
        high=hi * nominal_values,
        size=(n_samples, len(randomize_params)),
    )


def kfold_evaluate(
    train_fn: Any,
    nominal_params: BiologicalParameters,
    control_config: ControlConfig,
    rl_config: RLConfig | None = None,
    n_folds: int = 5,
    n_samples: int = 1000,
    n_eval_per_fold: int = 100,
    seed: int = 0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """K-Fold cross-validation of an RL policy over the parameter manifold.

    For each of the K folds:
        1. Partition n_samples points in Theta into K disjoint groups.
        2. Train a policy on the union of K-1 groups (the train set).
        3. Evaluate on the held-out group (the test set).

    Parameters
    ----------
    train_fn
        Callable ``(train_params_pool) -> model`` returning a trained SB3 model.
        The pool restricts the env's randomization to the train fold.
    n_folds
        Number of folds (default 5, sklearn convention).
    n_samples
        Total number of parameter points to partition.
    n_eval_per_fold
        Number of evaluation episodes per fold (defaults to 100).

    Returns
    -------
    Dictionary with per-fold metrics and aggregate mean/std.

    Notes
    -----
    Training K policies is expensive (~K * 2-4 h on CPU). This
    function is provided as a reference implementation; the primary evaluation
    protocol is ``evaluate_policy`` over the full Theta.
    """
    try:
        from sklearn.model_selection import KFold
    except ImportError as e:
        raise ImportError(
            "K-Fold evaluation requires scikit-learn. "
            "Install with: pip install scikit-learn"
        ) from e

    rl_cfg = rl_config or RLConfig()
    theta = sample_theta_grid(
        nominal_params,
        rl_cfg.randomize_params,
        rl_cfg.randomize_range,
        n_samples,
        seed=seed,
    )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    per_fold: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(theta)):
        logger.info("Fold %d/%d: |train|=%d  |test|=%d",
                    fold_idx + 1, n_folds, len(train_idx), len(test_idx))
        train_pool = theta[train_idx]
        test_pool = theta[test_idx]

        model = train_fn(train_pool)
        fold_result = _evaluate_on_pool(
            model, nominal_params, control_config, rl_cfg, test_pool,
            n_episodes=n_eval_per_fold, seed=seed + 1000 * fold_idx,
        )
        per_fold.append(
            {
                "fold": fold_idx,
                "mean_cost": fold_result.mean_cost,
                "std_cost": fold_result.std_cost,
                "success_rate": fold_result.success_rate,
                "n_episodes": fold_result.n_episodes,
            }
        )

    mean_costs = np.array([f["mean_cost"] for f in per_fold])
    success_rates = np.array([f["success_rate"] for f in per_fold])

    summary = {
        "n_folds": n_folds,
        "n_samples": n_samples,
        "per_fold": per_fold,
        "aggregate": {
            "mean_cost": float(mean_costs.mean()),
            "std_cost_across_folds": float(mean_costs.std(ddof=1)),
            "mean_success_rate": float(success_rates.mean()),
            "std_success_rate": float(success_rates.std(ddof=1)),
        },
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "kfold_results.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Saved K-Fold results to %s", out_path)

    return summary


def _evaluate_on_pool(
    model: Any,
    nominal_params: BiologicalParameters,
    control_config: ControlConfig,
    rl_config: RLConfig,
    test_pool: NDArray[np.float64],
    n_episodes: int,
    seed: int,
) -> EvaluationResult:
    """Helper: evaluate on parameters drawn from a fixed test pool."""
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    costs: list[float] = []
    F_terminals: list[float] = []

    for episode in range(n_episodes):
        idx = int(rng.integers(0, len(test_pool)))
        params_values = test_pool[idx]
        updates = dict(zip(rl_config.randomize_params, params_values, strict=True))
        episode_params = replace(nominal_params, **updates)

        env = SITEnv(
            episode_params,
            control_config,
            replace(rl_config, randomize=False),
        )
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        info: dict[str, Any] = {"F": float(env.state[0])}
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if truncated:
                break
        costs.append(env.cumulative_cost)
        F_terminals.append(info["F"])

    epsilon = (
        float(control_config.epsilon)
        if control_config.epsilon is not None
        else float(nominal_params.F_bar / 4.0)
    )
    success_rate = float(np.mean([F <= epsilon for F in F_terminals]))

    return EvaluationResult(
        n_episodes=n_episodes,
        mean_cost=float(np.mean(costs)),
        std_cost=float(np.std(costs, ddof=1)) if n_episodes > 1 else 0.0,
        success_rate=success_rate,
        mean_F_terminal=float(np.mean(F_terminals)),
        epsilon=epsilon,
        costs=costs,
        F_terminals=F_terminals,
        sampled_params=[],
    )
