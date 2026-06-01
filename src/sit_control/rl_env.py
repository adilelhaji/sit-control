"""Gymnasium environment for the SIT reinforcement-learning strategy (Strategy 9).

Wraps the reduced model S1 (F, M_s) as a partially observable MDP suitable for
training PPO / SAC policies. Supports domain randomization on biological
parameters for robust generalization across the parameter manifold
Theta = [randomize_range_low, randomize_range_high] * theta_nominal.

The environment is independent of any deep-learning framework: it depends
only on numpy + gymnasium. The training and evaluation modules (rl_train,
rl_evaluate) provide the SB3 / PyTorch wrappers.

References
----------
Schulman et al. (2017). Proximal Policy Optimization. arXiv:1707.06347.
Haarnoja et al. (2018). Soft Actor-Critic. ICML 2018.
Tobin et al. (2017). Domain Randomization. IROS 2017, 23-30.
Bottcher (2026). Control of dynamical systems with neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .model import recruitment
from .parameters import BiologicalParameters, ControlConfig


@dataclass(frozen=True, slots=True)
class RLConfig:
    """RL-specific configuration of the SIT environment.

    Attributes
    ----------
    dt
        Integration step in days. Episode length is ceil(T / dt).
    randomize
        If True, sample new biological parameters on each reset().
    randomize_range
        Multiplicative range (low, high) applied to each randomized parameter.
        The default (0.7, 1.3) implements +/- 30 % domain randomization
        (Tobin et al. 2017).
    randomize_params
        Names of BiologicalParameters fields to randomize. Defaults to the
        four parameters identified as most influential in the sensitivity
        analysis (Strategy 8): beta_E, delta_F, nu_E, K.
    terminal_penalty
        Multiplier of the terminal hinge penalty (rel + rel^2), with
        rel = max(0, F(T) - epsilon) / F_bar_nom, scaled by U_max * T in the
        reward. Larger values push the agent harder towards feasibility.
    fixed_epsilon
        If True, the suppression threshold epsilon is computed from the
        NOMINAL F_bar (target invariant across perturbations). If False,
        epsilon scales with the perturbed F_bar. True recommended for
        K-Fold evaluation consistency.
    """

    dt: float = 1.0
    randomize: bool = True
    randomize_range: tuple[float, float] = (0.7, 1.3)
    randomize_params: tuple[str, ...] = ("beta_E", "delta_F", "nu_E", "K")
    terminal_penalty: float = 1.0
    fixed_epsilon: bool = True


class SITEnv(gym.Env):
    """Gymnasium environment wrapping the S1 SIT model for RL training.

    Observation
    -----------
    Vector (F / F_bar_nom, M_s / Ms_scale, t / T) of dtype float32. The
    normalization uses the NOMINAL parameters so the observation scale is
    independent of the per-episode perturbation.

    Action
    ------
    Continuous u in [0, 1] of dtype float32. Internally scaled to
    [0, U_max] mosquitoes per day.

    Reward
    ------
    - Step:     r_t = -u(t) * dt                            (linear cost)
    - Terminal: r_T -= terminal_penalty * (rel + rel^2) * (U_max * T),
                with rel = max(0, F(T) - epsilon) / F_bar_nom
                (hinge penalty: linear term keeps a non-zero gradient at the
                 boundary, quadratic term adds curvature)

    Episode end
    -----------
    Terminated when t >= T (truncated is always False).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        nominal_params: BiologicalParameters,
        control_config: ControlConfig,
        rl_config: RLConfig | None = None,
    ) -> None:
        super().__init__()
        self.nominal: BiologicalParameters = nominal_params
        self.ctrl_cfg: ControlConfig = control_config
        self.rl_cfg: RLConfig = rl_config or RLConfig()

        self._F_scale: float = float(nominal_params.F_bar)
        self._Ms_scale: float = float(
            control_config.U_max / nominal_params.delta_s
        )
        self._epsilon_nominal: float = (
            float(control_config.epsilon)
            if control_config.epsilon is not None
            else float(nominal_params.F_bar / 4.0)
        )

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.params: BiologicalParameters = nominal_params
        self.state: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self.t: float = 0.0
        self.cumulative_cost: float = 0.0

    def _sample_params(
        self, rng: np.random.Generator,
    ) -> BiologicalParameters:
        """Sample a perturbed parameter set for domain randomization."""
        if not self.rl_cfg.randomize:
            return self.nominal
        lo, hi = self.rl_cfg.randomize_range
        updates: dict[str, float] = {}
        for name in self.rl_cfg.randomize_params:
            nominal_value = getattr(self.nominal, name)
            updates[name] = float(rng.uniform(lo * nominal_value, hi * nominal_value))
        return replace(self.nominal, **updates)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.params = self._sample_params(self.np_random)
        self.state = np.array(
            [self.params.F_bar, 0.0], dtype=np.float64,
        )
        self.t = 0.0
        self.cumulative_cost = 0.0
        return self._observation(), {"params": self.params}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        u = float(np.clip(np.asarray(action).flatten()[0], 0.0, 1.0))
        u *= self.ctrl_cfg.U_max
        dt = self.rl_cfg.dt

        F, Ms = self.state
        dF = float(recruitment(F, Ms, self.params))
        dMs = u - self.params.delta_s * Ms
        F_new = max(F + dt * dF, 0.0)
        Ms_new = max(Ms + dt * dMs, 0.0)
        self.state = np.array([F_new, Ms_new], dtype=np.float64)
        self.t += dt
        self.cumulative_cost += u * dt

        reward = -u * dt

        terminated = self.t >= self.ctrl_cfg.T - 1e-9
        truncated = False
        if terminated:
            epsilon = (
                self._epsilon_nominal
                if self.rl_cfg.fixed_epsilon
                else float(self.params.F_bar / 4.0)
            )
            excess = max(0.0, F_new - epsilon)
            # Linear (hinge) penalty rather than quadratic: the quadratic term
            # has zero gradient as excess -> 0, so the agent always stops
            # slightly above epsilon (systematic under-suppression). A linear
            # penalty keeps a constant gradient at the boundary, driving F(T)
            # down to epsilon. A quadratic term is retained, with small weight,
            # to penalise large violations more strongly.
            rel = excess / self._F_scale
            reward -= (
                self.rl_cfg.terminal_penalty
                * (rel + rel * rel)
                * (self.ctrl_cfg.U_max * self.ctrl_cfg.T)
            )

        info = {
            "u": u,
            "F": F_new,
            "Ms": Ms_new,
            "t": self.t,
            "cumulative_cost": self.cumulative_cost,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> NDArray[np.float32]:
        F, Ms = self.state
        return np.array(
            [
                F / self._F_scale,
                Ms / self._Ms_scale,
                self.t / self.ctrl_cfg.T,
            ],
            dtype=np.float32,
        )

    @property
    def epsilon(self) -> float:
        """Suppression target used by the current episode."""
        if self.rl_cfg.fixed_epsilon:
            return self._epsilon_nominal
        return float(self.params.F_bar / 4.0)


def make_env(
    nominal_params: BiologicalParameters,
    control_config: ControlConfig,
    rl_config: RLConfig | None = None,
    seed: int | None = None,
) -> SITEnv:
    """Factory that returns an initialised SITEnv (Gymnasium-compatible).

    Useful for `stable_baselines3.common.env_util.make_vec_env`:

    >>> from functools import partial
    >>> env_fn = partial(make_env, nominal_params=p, control_config=c)
    >>> vec_env = make_vec_env(env_fn, n_envs=16)
    """
    env = SITEnv(nominal_params, control_config, rl_config)
    if seed is not None:
        env.reset(seed=seed)
    return env
