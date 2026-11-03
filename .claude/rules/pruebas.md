---
paths:
  - "tests/**"
---
- Pytest, un archivo por módulo (`tests/test_<modulo>.py`). La cobertura de `sit_control` se muestra siempre (`addopts` de `pyproject.toml`).
- Las pruebas lentas llevan `@pytest.mark.slow`; `scripts/comprobar.sh` y CI las saltan (`-m "not slow"`). No marques una prueba como lenta para esquivar un fallo.
- Las pruebas del optimizador necesitan GEKKO (`pytest.importorskip`); las de RL, los extras `rl`.
- Una prueba nueva prueba un criterio de la issue y se ha visto fallar antes de que exista el código. No se ajusta la prueba al código.
- Ni se baja la cobertura, ni se excluyen archivos, ni se fija un umbral (`fail_under`): eso lo decide Adil.
