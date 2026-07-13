"""Unit tests for evaluate_policy (per-episode suppression threshold)."""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.parameters import BiologicalParameters, ControlConfig
from sit_control.rl_env import RLConfig
from sit_control.rl_evaluate import evaluate_policy


class _ZeroPolicy:
    """Mock SB3 model that always releases nothing (u = 0)."""

    def predict(self, obs, deterministic=True):  # noqa: ANN001, ARG002
        return np.array([0.0], dtype=np.float32), None


def test_evaluate_policy_floating_epsilon_is_per_episode() -> None:
    """With fixed_epsilon=False each episode has its own epsilon (from its
    perturbed F_bar). evaluate_policy must judge every F(T) against the
    threshold it was actually run under, not a single last-episode value.

    A zero-release policy never suppresses, so success_rate must be 0 and the
    run must complete coherently for all episodes.
    """
    params = BiologicalParameters()
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    rl_cfg = RLConfig(randomize=True, fixed_epsilon=False)

    result = evaluate_policy(
        _ZeroPolicy(),
        params,
        cfg,
        rl_cfg,
        n_episodes=4,
        seed=0,
    )

    assert result.n_episodes == 4
    assert len(result.F_terminals) == 4
    assert 0.0 <= result.success_rate <= 1.0
    assert result.success_rate == 0.0  # no releases -> F stays ~F_bar > eps
    assert result.epsilon > 0.0


def test_evaluate_policy_fixed_epsilon_constant() -> None:
    """With fixed_epsilon=True the threshold is the nominal F_bar/4 for every
    episode, so the reported epsilon equals that constant."""
    params = BiologicalParameters()
    cfg = ControlConfig(T=150.0, U_max=5000.0)
    rl_cfg = RLConfig(randomize=True, fixed_epsilon=True)

    result = evaluate_policy(
        _ZeroPolicy(),
        params,
        cfg,
        rl_cfg,
        n_episodes=3,
        seed=0,
    )

    assert result.epsilon == pytest.approx(params.F_bar / 4.0, rel=1e-9)
    assert result.success_rate == 0.0
