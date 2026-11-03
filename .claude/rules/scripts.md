---
paths:
  - "scripts/**"
  - "main.py"
---
- `main.py` es la entrada de línea de órdenes y `scripts/run_*.py` reproducen los resultados de la tesis; la lógica va en `src/sit_control/`, no aquí.
- Las salidas (JSON, PDF) se escriben en `results/` (ignorado por git); no se confirman resultados generados.
- `ruff check` y `ruff format --check` también cubren `scripts/`.
- `scripts/comprobar.sh` es lo que ejecuta el hook de parada y debe seguir igual que CI (`.github/workflows/ci.yml`); si cambia una, cambia la otra.
