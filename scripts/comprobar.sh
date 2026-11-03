#!/usr/bin/env bash
# Ejecuta las comprobaciones que ya existen en este repositorio (las mismas que CI,
# .github/workflows/ci.yml): lint, formato, tipos estrictos y pruebas.
# Es lo que ejecuta el hook de parada de Claude Code.
# No impone ningún umbral de cobertura: pytest la muestra (addopts de pyproject.toml)
# pero no falla por ella. Fijar un umbral es una decisión de Adil.
set -u
raiz="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$raiz" || exit 1

command -v poetry >/dev/null 2>&1 || { echo "comprobar.sh: falta poetry (https://python-poetry.org)"; exit 1; }
if [ ! -x .venv/bin/python ]; then
  echo "== instalando dependencias (poetry install --all-extras)"
  poetry install --all-extras --no-interaction >/dev/null || { echo "comprobar.sh: no se pudo instalar"; exit 1; }
fi

fallos=0
paso() { # nombre comando...
  local nombre="$1"; shift
  echo "== $nombre"
  "$@" || { echo "   FALLA: $nombre"; fallos=$((fallos + 1)); }
}

paso "ruff check"  poetry run ruff check src/ tests/ scripts/
paso "ruff format" poetry run ruff format --check src/ tests/ scripts/
paso "mypy"        poetry run mypy --strict --ignore-missing-imports src/
paso "pytest"      poetry run pytest -m "not slow"

if [ "$fallos" -ne 0 ]; then
  echo "comprobar.sh: $fallos comprobación(es) con fallos"
  exit 1
fi
echo "comprobar.sh: todo en verde"
