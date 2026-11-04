# SIT-Control

Implementación numérica del control óptimo de la técnica del insecto estéril (SIT) de Almeida, Duprez, Privat y Vauchelet (2022, *J. Differential Equations* 311). Acompaña el TFG de Adil El Haji (UAB, 2026). Python 3.12+, poetry, paquete `sit_control` en `src/`. Detalle científico y estructura: `README.md`.

## Comandos

```bash
poetry install --all-extras      # dependencias (los extras `rl` traen torch: pesan)
scripts/comprobar.sh             # lo mismo que CI: ruff, formato, mypy estricto, pytest -m "not slow"
poetry run pytest                # pruebas con cobertura (incluye las lentas)
poetry run ruff check src/ tests/ scripts/
poetry run ruff format src/ tests/ scripts/
poetry run mypy --strict --ignore-missing-imports src/
python main.py verify --config configs/almeida2022.yaml   # también: strategies, convergence, sensitivity, rl-train, rl-eval
```

## Convenciones (las comprueba el código)

- Tipos completos, `mypy --strict` sobre `src/`. Ruff con `E F W I N UP B A C4 RET SIM`, línea de 88.
- Las variables del modelo siguen la notación del artículo (`E`, `M`, `F`, `Ms`, `beta_E`…) a propósito; el linter está configurado para permitirlo.
- Un archivo de pruebas por módulo en `tests/`; las lentas llevan `@pytest.mark.slow`.
- La lógica va en `src/sit_control/`; `main.py` y `scripts/run_*.py` solo la llaman. Los resultados van a `results/` (ignorado por git).
- Valores de referencia en `configs/*.yaml` (Almeida 2022, Join 2026).
- Reglas por carpeta en `.claude/rules/`.

## Cómo se trabaja

- Una issue = una rama = una PR. Flujo con las skills `/issue`, `/revisar` y `/pr` y los agentes de `.claude/agents/`. Explica el resultado a Adil en castellano y sin tecnicismos.
- Commits en castellano, con el porqué. Nunca push a `main`.
- **Aprobación de Adil** (decidido por Adil el 3-11-2026, confirmado el 4-11-2026): Adil solo aprueba una PR si cambia la arquitectura o se desvía de la propuesta original; entonces se para y se le pregunta. Todo lo demás lo fusiona la sesión (squash) cuando `/revisar` no deja nada que bloquee y las comprobaciones están en verde, en GitHub si las hay y con `scripts/comprobar.sh` si no. `CODEOWNERS` es aviso, no bloqueo: la PR lo dice en el cuerpo y no espera. Las rutas de `CODEOWNERS` son las delicadas (guía, flujo, CI, dependencias, `configs/`, `model.py`, `parameters.py`, `docs/`, README).
- Garantizado por los hooks de `.claude/hooks/`: no se hace push a `main` ni forzado ni con `--no-verify`; no se escriben archivos de secretos; no se pone ni se toca un umbral de cobertura (`fail_under`); al terminar, si hay código tocado, se ejecuta `scripts/comprobar.sh` y debe pasar. Para una excepción consciente: `SIT_PERMITIR_PROTEGIDOS=1` en la terminal de Adil.
- No leer `.env` ni otros secretos (están denegados en `.claude/settings.json`).

## Pendiente de decisión de Adil

- No hay umbral de cobertura: pytest la muestra pero no falla por ella. No se impone ninguno sin que Adil lo decida.
