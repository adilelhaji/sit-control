---
paths:
  - "src/sit_control/**"
---
- Python 3.12, tipos completos: `mypy --strict` pasa sobre `src/` (`disallow_untyped_defs`). Sin `# type: ignore` nuevos sin motivo escrito.
- Las variables del modelo siguen la notación matemática de Almeida et al. (2022) a propósito (`E`, `M`, `F`, `Ms`, `T`, `K`, `beta_E`, `delta_F`…); no las renombres para contentar al linter (está configurado para eso en `pyproject.toml`).
- La dinámica del modelo (S1 y S2) vive en `model.py`; el resto (simulador, optimizador, bisección) la usa, no la copia.
- Los parámetros biológicos y de control van en `parameters.py` y se leen de `configs/*.yaml`; no pongas números del modelo sueltos en el código.
- Ruff con las reglas de `pyproject.toml` (línea de 88). Pasa `ruff check` y `ruff format --check`.
- `rl_env.py`, `rl_train.py` y `rl_evaluate.py` dependen de los extras `rl` (gymnasium, stable-baselines3, torch, scikit-learn); `main.py` los importa solo cuando se usa un subcomando de RL.
