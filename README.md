# SIT-Control: Optimal Control for the Sterile Insect Technique

[![CI](https://github.com/adilelhaji/sit-control/actions/workflows/ci.yml/badge.svg)](https://github.com/adilelhaji/sit-control/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

Numerical implementation of the optimal control framework for the
Sterile Insect Technique (SIT) developed in Almeida, Duprez, Privat &
Vauchelet (2022), *Journal of Differential Equations* 311, 229–266.

This repository accompanies the Bachelor's thesis (TFG) *Optimal Control
Strategies for Mosquito Population Suppression via the Sterile Insect
Technique* — Adil El Haji, Universitat Autònoma de Barcelona, 2026.

## Features

- Numerical integration of the full (S2, 4-state) and reduced (S1, 2-state)
  mosquito population models via SciPy `solve_ivp`.
- L¹ and L² optimal control solvers via GEKKO + APOPT.
- Bisection algorithm reproducing Algorithm 2 of Almeida *et al.* (2022),
  exploiting the bang-singular-bang structure of the L¹ solution.
- Five release strategies: L¹ optimal, L² optimal, constant, periodic
  impulsive, and optimal impulsive (scipy SLSQP over batch amounts).
- Parametric sensitivity analysis (±20 % on δ_F, ν_E, K) with tornado chart.
- Unified CLI entry point (`main.py`) and four reproducibility scripts.
- Reproducibility-first design: typed code, YAML configuration, unit tests.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/adilelhaji/sit-control.git
cd sit-control
pip install -e ".[dev]"
```

## Quick start

### Unified CLI

```bash
# Verify J(u*) against Almeida et al. (2022) reference values
python main.py verify --config configs/almeida2022.yaml

# Run and compare the five SIT control strategies (TFG Chapter 4)
python main.py strategies --config configs/almeida2022.yaml --output results

# Convergence study: J(u*) vs GEKKO collocation size N
python main.py convergence --config configs/almeida2022.yaml

# Parametric sensitivity analysis ±20 % on delta_F, nu_E, K
python main.py sensitivity --config configs/almeida2022.yaml

# Train the closed-loop RL policy (requires the [rl] extras)
python main.py rl-train --config configs/almeida2022.yaml --algorithm PPO

# Evaluate a trained RL policy on 100 random parameter draws
python main.py rl-eval --config configs/almeida2022.yaml --model results/rl/ppo_policy.zip
```

To enable the RL features (closed-loop policy):

```bash
pip install -e ".[rl]"   # gymnasium, stable-baselines3, torch, scikit-learn
```

### Programmatic use

```python
from sit_control.parameters import BiologicalParameters, ControlConfig
from sit_control.optimizer import GekkoOptimiser

params = BiologicalParameters()           # Almeida et al. (2022), Table 1
cfg    = ControlConfig(T=150.0, U_max=5000.0)

# L1 optimal control (bang-singular-bang)
opt = GekkoOptimiser(params).solve_L1(cfg)
print(f"J_L1(u*) = {opt.cost:.4e}")

# L2 optimal control (smooth profile)
opt2 = GekkoOptimiser(params).solve_L2(cfg, c_weight=1.0)
print(f"J_L2(u*) = {opt2.cost:.4e}")
```

```python
from sit_control.simulator import Simulator
from sit_control.controls import constant_control, impulsive_control
import numpy as np

sim = Simulator(BiologicalParameters())

# Simulate the full (S2) model under constant release
result = sim.simulate(T=150.0, u_func=constant_control(1000.0), model="S2")
print(f"F(T) = {result.state[2, -1]:.1f} females")

# Simulate periodic impulsive releases every 7 days
times = np.arange(0.0, 150.0, 7.0)
result = sim.simulate(T=150.0, u_func=impulsive_control(times, 5000.0), model="S1")
```

## Project structure

```
sit-control/
├── main.py                   # Unified CLI entry point
├── configs/
│   ├── almeida2022.yaml      # Reference configuration (Almeida et al. 2022, K=18258)
│   └── join2026.yaml         # Alternative field calibration (Join et al. 2026, K=22200)
├── src/sit_control/
│   ├── parameters.py         # BiologicalParameters, ControlConfig, NumericalConfig
│   ├── model.py              # ODE right-hand sides for S1 and S2 (single source of f)
│   ├── simulator.py          # Simulator wrapping scipy solve_ivp
│   ├── controls.py           # Control law factories
│   ├── optimizer.py          # GekkoOptimiser: solve_L1, solve_L2
│   ├── bisection.py          # Algorithm 2: bang-singular-bang bisection
│   ├── pmp_sweep.py          # PMP design-space sweep (horizon vs capacity)
│   ├── metrics.py            # cost_L1, cost_L2, suppression_time
│   ├── plotting.py           # Publication-quality figures
│   ├── rl_env.py             # SITEnv: Gymnasium MDP over S1 with domain randomization
│   ├── rl_train.py           # PPO training (Stable-Baselines3)
│   └── rl_evaluate.py        # Robustness + K-fold evaluation of trained policies
├── scripts/
│   ├── run_verification.py   # Verification table vs Almeida (2022)
│   ├── run_convergence.py    # Convergence study J(u*) vs N
│   ├── run_strategies.py     # Five-strategy comparison (TFG Ch. 4)
│   ├── run_sensitivity.py    # Parametric sensitivity analysis (TFG Ch. 4)
│   ├── run_optimal_structure.py  # bang-singular-bang structure figure
│   ├── run_comparison_s1_s2.py   # S1 vs S2 trajectory comparison
│   ├── run_rl_training.py    # RL policy training entry point
│   ├── run_rl_evaluation.py  # RL policy evaluation entry point
│   └── rl_rollout.py, rl_figures.py, rl_summary.py  # RL post-processing helpers
└── tests/
    ├── test_parameters.py    test_model.py        test_simulator.py
    ├── test_optimizer.py     test_bisection.py    test_run_strategies.py
    └── test_run_sensitivity.py  test_rl_env.py  test_rl_evaluate.py  test_main.py
```

## Reproducing the thesis results

| Thesis element | Command |
|---|---|
| Table: verification of J(u*) for T ∈ {60, 150, 200} | `python main.py verify --config configs/almeida2022.yaml` |
| Table: convergence J(u*) vs N ∈ {50, 100, 200, 400} | `python main.py convergence --config configs/almeida2022.yaml` |
| Table + figures: five-strategy comparison (Ch. 4) | `python main.py strategies --config configs/almeida2022.yaml` |
| Table + tornado chart: sensitivity ±20 % (Ch. 4) | `python main.py sensitivity --config configs/almeida2022.yaml` |

All outputs are written to `results/` as JSON files and PDF figures.

## Mathematical model

The mosquito population is described by the four-state system S2 (Strugarek *et al.*, 2019):

$$\dot{E} = \beta_E F\!\left(1 - \tfrac{E}{K}\right) - (\nu_E + \delta_E)E, \quad
\dot{M} = (1-\nu)\nu_E E - \delta_M M$$

$$\dot{F} = \nu\nu_E E \cdot \frac{M}{M + \gamma_s M_s} - \delta_F F, \quad
\dot{M}_s = u(t) - \delta_s M_s$$

The quasi-stationary approximation (QSSA) eliminates E and M, yielding the
reduced two-state system S1 used for optimal control. The L¹ problem

$$\min_{u \in \mathcal{U}} \int_0^T u(t)\,dt \quad \text{s.t.} \quad F(T) \leq \varepsilon$$

is solved numerically via GEKKO + APOPT and analytically characterised
by the Pontryagin Maximum Principle (bang-singular-bang structure).

## Development

```bash
# Run the full test suite
pytest

# Type-check the source
mypy src/

# Lint and format
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/

# Install pre-commit hooks
pre-commit install
```

## Citing

If you use this code, please cite the original article and the thesis:

```bibtex
@article{Almeida2022optimal,
  author  = {Almeida, Lu{\'i}s and Duprez, Michel and Privat, Yannick
             and Vauchelet, Nicolas},
  title   = {Optimal control strategies for the sterile mosquitoes technique},
  journal = {Journal of Differential Equations},
  volume  = {311},
  pages   = {229--266},
  year    = {2022},
  doi     = {10.1016/j.jde.2021.12.002}
}

@thesis{ElHaji2026sit,
  author = {El Haji, Adil},
  title  = {Optimal Control Strategies for Mosquito Population
            Suppression via the Sterile Insect Technique},
  school = {Universitat Aut{\`o}noma de Barcelona},
  year   = {2026},
  type   = {Treball de Fi de Grau}
}
```

## License

The source code in this repository is released under the
[MIT License](LICENSE).

The accompanying thesis is deposited in the UAB *Dipòsit Digital de
Documents* (DDD) under Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International (CC BY-NC-ND 4.0), in accordance with UAB institutional policy.
