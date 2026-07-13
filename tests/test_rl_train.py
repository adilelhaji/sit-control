"""Unit tests for the RL training module (rl_train).

The pure helpers (``_linear_schedule``, ``TrainingConfig`` defaults) and the
error path of ``train`` run without any learning and stay UNMARKED so they
execute in CI's ``-m "not slow"`` selection. The two end-to-end ``train``
tests actually spin up Stable-Baselines3 + PyTorch and are ``@pytest.mark.slow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from sit_control.parameters import BiologicalParameters, ControlConfig  # noqa: E402
from sit_control.rl_env import RLConfig  # noqa: E402
from sit_control.rl_train import (  # noqa: E402
    TrainingConfig,
    _linear_schedule,
    train,
)

# ---------------------------------------------------------------------------
# Pure helpers (fast, unmarked)
# ---------------------------------------------------------------------------


def test_linear_schedule_returns_callable_decaying_to_zero() -> None:
    """_linear_schedule(initial) must decay linearly with progress_remaining."""
    initial = 3e-4
    schedule = _linear_schedule(initial)
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(initial)
    assert schedule(0.0) == pytest.approx(0.0)
    assert schedule(0.5) == pytest.approx(0.5 * initial)


def test_training_config_defaults() -> None:
    """Default TrainingConfig must match the documented PPO hyperparameters."""
    cfg = TrainingConfig()
    assert cfg.algorithm == "PPO"
    assert cfg.total_timesteps == 1_000_000
    assert cfg.n_envs == 16
    assert cfg.learning_rate == pytest.approx(3e-4)
    assert cfg.ent_coef == pytest.approx(0.0)
    assert cfg.batch_size == 64
    assert cfg.lr_schedule == "constant"
    assert cfg.seed == 0
    assert cfg.policy_kwargs is None


def test_training_config_is_frozen() -> None:
    """TrainingConfig is an immutable (frozen) dataclass."""
    cfg = TrainingConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.seed = 5  # type: ignore[misc]


def test_train_unknown_algorithm_raises(tmp_path: Path) -> None:
    """An unrecognised algorithm must fail loud with a ValueError.

    This exercises the error path of ``train`` without any learning: the
    env factory is built but neither the PPO nor the SAC branch is taken.
    """
    params = BiologicalParameters()
    ctrl = ControlConfig(T=5.0, U_max=5000.0)
    bad_cfg = TrainingConfig(algorithm="FOO")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown algorithm"):
        train(
            params,
            ctrl,
            rl_config=RLConfig(dt=1.0, randomize=False),
            training_config=bad_cfg,
            output_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# End-to-end training (slow: real SB3 + PyTorch learning)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_ppo_tiny_budget_saves_loadable_policy(tmp_path: Path) -> None:
    """train() with a tiny PPO budget must persist a loadable, predicting policy.

    Uses the linear LR schedule to also exercise that branch. The returned
    Path must point at an existing ``ppo_policy.zip`` whose loaded model
    exposes ``predict`` and yields an admissible action.
    """
    from stable_baselines3 import PPO

    params = BiologicalParameters()
    ctrl = ControlConfig(T=5.0, U_max=5000.0)
    train_cfg = TrainingConfig(
        algorithm="PPO",
        total_timesteps=64,
        n_envs=1,
        lr_schedule="linear",
        seed=7,
    )

    model_path = train(
        params,
        ctrl,
        rl_config=RLConfig(dt=1.0, randomize=False),
        training_config=train_cfg,
        output_dir=tmp_path,
    )

    assert isinstance(model_path, Path)
    assert model_path.exists()
    assert model_path.name == "ppo_policy.zip"

    model = PPO.load(model_path)
    assert hasattr(model, "predict")
    import numpy as np

    obs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,)
    assert 0.0 <= float(action[0]) <= 1.0


@pytest.mark.slow
def test_train_sac_tiny_budget_saves_loadable_policy(tmp_path: Path) -> None:
    """train() with a tiny SAC budget must persist a loadable, predicting policy.

    Exercises the off-policy (SAC) branch, which builds a single env rather
    than a vectorised one.
    """
    from stable_baselines3 import SAC

    params = BiologicalParameters()
    ctrl = ControlConfig(T=5.0, U_max=5000.0)
    train_cfg = TrainingConfig(
        algorithm="SAC",
        total_timesteps=32,
        seed=3,
    )

    model_path = train(
        params,
        ctrl,
        rl_config=RLConfig(dt=1.0, randomize=False),
        training_config=train_cfg,
        output_dir=tmp_path,
    )

    assert isinstance(model_path, Path)
    assert model_path.exists()
    assert model_path.name == "sac_policy.zip"

    model = SAC.load(model_path)
    assert hasattr(model, "predict")
