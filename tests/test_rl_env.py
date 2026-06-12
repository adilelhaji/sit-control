"""Unit tests for the SIT Gymnasium environment (closed-loop RL policy).

These tests verify the Gymnasium API contract and the dynamics of the
environment. They do NOT depend on stable-baselines3 or torch.
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from sit_control.parameters import BiologicalParameters, ControlConfig
from sit_control.rl_env import RLConfig, SITEnv


@pytest.fixture
def env() -> SITEnv:
    """Default env with domain randomization enabled."""
    return SITEnv(
        nominal_params=BiologicalParameters(),
        control_config=ControlConfig(T=150.0, U_max=5000.0),
        rl_config=RLConfig(dt=1.0, randomize=True),
    )


@pytest.fixture
def env_deterministic() -> SITEnv:
    """Env with randomization disabled (nominal params only)."""
    return SITEnv(
        nominal_params=BiologicalParameters(),
        control_config=ControlConfig(T=150.0, U_max=5000.0),
        rl_config=RLConfig(dt=1.0, randomize=False),
    )


# ---------------------------------------------------------------------------
# Gymnasium API contract
# ---------------------------------------------------------------------------

def test_action_space_box_unit_interval(env: SITEnv) -> None:
    """Action space must be Box([0, 1]) with shape (1,)."""
    assert env.action_space.shape == (1,)
    assert float(env.action_space.low[0]) == 0.0
    assert float(env.action_space.high[0]) == 1.0


def test_observation_space_three_dimensional(env: SITEnv) -> None:
    """Observation must be a 3-vector (F, Ms, t) normalized."""
    assert env.observation_space.shape == (3,)


def test_reset_returns_tuple(env: SITEnv) -> None:
    """reset() must return (obs, info)."""
    obs, info = env.reset(seed=42)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert "params" in info


def test_step_returns_five_tuple(env: SITEnv) -> None:
    """step() must return (obs, reward, terminated, truncated, info)."""
    env.reset(seed=0)
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info.keys()) >= {"u", "F", "Ms", "t", "cumulative_cost"}


def test_episode_terminates_at_T(env: SITEnv) -> None:
    """Episode must terminate when t >= T."""
    env.reset(seed=0)
    steps = 0
    terminated = False
    while not terminated and steps < 200:
        _, _, terminated, _, _ = env.step(np.array([0.5], dtype=np.float32))
        steps += 1
    assert terminated
    assert steps == int(env.ctrl_cfg.T / env.rl_cfg.dt)


# ---------------------------------------------------------------------------
# Dynamics correctness
# ---------------------------------------------------------------------------

def test_initial_state_is_persistence_equilibrium(env: SITEnv) -> None:
    """At reset, F equals F_bar of the (possibly perturbed) parameters."""
    env.reset(seed=0)
    F_obs = env.state[0]
    assert F_obs == pytest.approx(env.params.F_bar)
    assert env.state[1] == 0.0


def test_zero_action_grows_Ms_only_via_decay(env_deterministic: SITEnv) -> None:
    """With u=0 and Ms=0 at start, Ms stays 0; F evolves alone."""
    env_deterministic.reset(seed=0)
    _, _, _, _, info = env_deterministic.step(np.array([0.0], dtype=np.float32))
    assert info["Ms"] == 0.0
    assert info["u"] == 0.0


def test_max_action_releases_U_max(env_deterministic: SITEnv) -> None:
    """Action = 1 corresponds to releasing U_max per dt."""
    env_deterministic.reset(seed=0)
    _, _, _, _, info = env_deterministic.step(np.array([1.0], dtype=np.float32))
    assert info["u"] == pytest.approx(env_deterministic.ctrl_cfg.U_max)


def test_cumulative_cost_equals_sum_of_releases(env_deterministic: SITEnv) -> None:
    """cumulative_cost must equal sum_t u(t) * dt."""
    env_deterministic.reset(seed=0)
    expected_u = 2000.0
    expected_action = expected_u / env_deterministic.ctrl_cfg.U_max
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env_deterministic.step(
            np.array([expected_action], dtype=np.float32),
        )
    expected_cost = expected_u * env_deterministic.ctrl_cfg.T
    assert info["cumulative_cost"] == pytest.approx(expected_cost, rel=1e-6)


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------

def test_randomization_changes_params_across_resets(env: SITEnv) -> None:
    """With randomize=True, two resets must yield different parameter sets."""
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    p1, p2 = info1["params"], info2["params"]
    assert (p1.beta_E, p1.delta_F, p1.nu_E, p1.K) != (
        p2.beta_E, p2.delta_F, p2.nu_E, p2.K,
    )


def test_no_randomization_yields_nominal_params(env_deterministic: SITEnv) -> None:
    """With randomize=False, every reset returns the nominal parameter set."""
    _, info1 = env_deterministic.reset(seed=1)
    _, info2 = env_deterministic.reset(seed=2)
    assert info1["params"] == info2["params"]
    assert info1["params"] == env_deterministic.nominal


def test_randomization_within_range(env: SITEnv) -> None:
    """All randomized parameters must lie within [low, high] * nominal."""
    lo, hi = env.rl_cfg.randomize_range
    for seed in range(20):
        _, info = env.reset(seed=seed)
        p = info["params"]
        for name in env.rl_cfg.randomize_params:
            nominal = getattr(env.nominal, name)
            sampled = getattr(p, name)
            assert lo * nominal <= sampled <= hi * nominal


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

def test_reward_step_is_negative_cost(env_deterministic: SITEnv) -> None:
    """A single non-terminal step yields reward = -u * dt."""
    env_deterministic.reset(seed=0)
    action = np.array([0.4], dtype=np.float32)
    expected_u = 0.4 * env_deterministic.ctrl_cfg.U_max
    _, reward, terminated, _, _ = env_deterministic.step(action)
    assert not terminated
    assert reward == pytest.approx(-expected_u * env_deterministic.rl_cfg.dt)


def test_terminal_penalty_zero_when_F_below_epsilon() -> None:
    """If F(T) <= epsilon, no additional terminal penalty applies."""
    env = SITEnv(
        nominal_params=BiologicalParameters(),
        control_config=ControlConfig(T=2.0, U_max=5000.0, epsilon=1e6),
        rl_config=RLConfig(dt=1.0, randomize=False, terminal_penalty=100.0),
    )
    env.reset(seed=0)
    env.step(np.array([0.0], dtype=np.float32))
    _, terminal_reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert terminal_reward == 0.0
