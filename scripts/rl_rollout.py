"""Roll out a trained RL policy on the nominal model and dump the trajectory.

Produces a small JSON with the closed-loop trajectory (t, F, u) of the trained
policy under nominal parameters, for plotting against the open-loop optimum.

Usage
-----
    python scripts/rl_rollout.py --config configs/almeida2022.yaml \
        --model results/rl/tp3_seed0/ppo_policy.zip --algorithm PPO \
        --output results/rl/tp3_seed0/rollout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sit_control.parameters import BiologicalParameters, ControlConfig, load_config
from sit_control.rl_env import RLConfig, make_env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    p = BiologicalParameters(**cfg.get("biological", {}))
    cc = ControlConfig(**cfg["control"])

    if a.algorithm == "PPO":
        from stable_baselines3 import PPO

        model = PPO.load(str(a.model))
    else:
        from stable_baselines3 import SAC

        model = SAC.load(str(a.model))

    env = make_env(p, cc, RLConfig(randomize=False))
    obs, _ = env.reset(seed=a.seed)
    ts = [0.0]
    fs = [float(env.state[0])]
    us: list[float] = []
    done = False
    info: dict = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        ts.append(float(info["t"]))
        fs.append(float(info["F"]))
        us.append(float(info["u"]))
        done = bool(term or trunc)
    us.append(us[-1])  # repeat last u to match len(t) for step plotting

    out = {
        "t": ts,
        "F": fs,
        "u": us,
        "epsilon": float(env.epsilon),
        "F_bar": float(p.F_bar),
        "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
        "F_terminal": float(info.get("F", fs[-1])),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out), encoding="utf-8")
    print(
        f"Saved rollout to {a.output} | cost={out['cumulative_cost']:.3e} "
        f"F(T)={out['F_terminal']:.0f} eps={out['epsilon']:.0f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
