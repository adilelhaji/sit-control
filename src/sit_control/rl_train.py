"""Train PPO or SAC policies on the SIT environment (Strategy 9).

Hyperparameter defaults follow the original PPO (Schulman et al. 2017) and
SAC (Haarnoja et al. 2018) papers, adapted to the short-horizon SIT episode
(150 steps with dt=1 day). Uses Stable-Baselines3 + PyTorch under the hood.

References
----------
Schulman et al. (2017). Proximal Policy Optimization. arXiv:1707.06347.
Haarnoja et al. (2018). Soft Actor-Critic. ICML 2018.
Raffin et al. (2021). Stable-Baselines3. J. Mach. Learn. Res. 22, 1-8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from .parameters import BiologicalParameters, ControlConfig
from .rl_env import RLConfig, make_env

logger = logging.getLogger(__name__)

Algorithm = Literal["PPO", "SAC"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for policy training.

    Attributes
    ----------
    algorithm
        "PPO" (on-policy, stable) or "SAC" (off-policy, sample-efficient).
    total_timesteps
        Total environment steps for training.
    n_envs
        Number of parallel environments (for PPO). Ignored by SAC.
    learning_rate
        Optimiser learning rate. Defaults are algorithm-specific.
    seed
        Random seed for reproducibility.
    policy_kwargs
        Keyword arguments passed to the SB3 policy (e.g., network architecture).
    """

    algorithm: Algorithm = "PPO"
    total_timesteps: int = 1_000_000
    n_envs: int = 16
    learning_rate: float = 3e-4
    seed: int = 0
    policy_kwargs: dict[str, Any] | None = None


def train(
    nominal_params: BiologicalParameters,
    control_config: ControlConfig,
    rl_config: RLConfig | None = None,
    training_config: TrainingConfig | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Train a policy and persist the model to disk.

    Parameters
    ----------
    nominal_params
        Nominal biological parameters (centre of the randomization range).
    control_config
        Control problem configuration (T, U_max, epsilon).
    rl_config
        Environment configuration; default enables +/- 30 % domain randomization.
    training_config
        Training hyperparameters; default is PPO with 1e6 timesteps.
    output_dir
        Directory where the trained model is saved as `<algorithm>_policy.zip`.

    Returns
    -------
    Path to the saved model file.
    """
    train_cfg = training_config or TrainingConfig()
    output_dir = output_dir or Path("results/rl")
    output_dir.mkdir(parents=True, exist_ok=True)

    env_fn = partial(
        make_env,
        nominal_params=nominal_params,
        control_config=control_config,
        rl_config=rl_config,
    )

    if train_cfg.algorithm == "PPO":
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(env_fn, n_envs=train_cfg.n_envs, seed=train_cfg.seed)
        policy_kwargs = train_cfg.policy_kwargs or {"net_arch": [64, 64, 64]}
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=train_cfg.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=policy_kwargs,
            seed=train_cfg.seed,
            verbose=1,
        )
    elif train_cfg.algorithm == "SAC":
        from stable_baselines3 import SAC

        single_env = env_fn()
        policy_kwargs = train_cfg.policy_kwargs or {"net_arch": [64, 64, 64]}
        model = SAC(
            "MlpPolicy",
            single_env,
            learning_rate=train_cfg.learning_rate,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            seed=train_cfg.seed,
            verbose=1,
        )
    else:
        raise ValueError(
            f"Unknown algorithm '{train_cfg.algorithm}'; use 'PPO' or 'SAC'."
        )

    logger.info(
        "Training %s for %d timesteps on %d envs (seed=%d)",
        train_cfg.algorithm, train_cfg.total_timesteps,
        train_cfg.n_envs if train_cfg.algorithm == "PPO" else 1,
        train_cfg.seed,
    )
    model.learn(total_timesteps=train_cfg.total_timesteps, progress_bar=True)

    model_path = output_dir / f"{train_cfg.algorithm.lower()}_policy.zip"
    model.save(model_path)
    logger.info("Saved policy to %s", model_path)
    return model_path
