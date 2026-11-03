#!/usr/bin/env bash
# Stop. Antes de que Claude dé el turno por terminado, si ha tocado código, ejecuta
# scripts/comprobar.sh (lint, formato, tipos, pruebas). Si falla, devuelve código 2 y
# Claude recibe la salida y sigue trabajando. Si no hay código tocado, no hace nada.
raiz="${CLAUDE_PROJECT_DIR:-$PWD}"
entrada="$(cat)"
aqui="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Si este mismo hook ya ha devuelto al modelo al trabajo y estamos en el bucle, no insistir
# más de lo que el propio Claude Code permite; el harness corta a las ocho.
activo="$(printf '%s' "$entrada" | bash "$aqui/leer-campo.sh" '.stop_hook_active')"

[ -x "$raiz/scripts/comprobar.sh" ] || [ -f "$raiz/scripts/comprobar.sh" ] || exit 0
cd "$raiz" || exit 0

carpetas_codigo="src tests scripts main.py pyproject.toml poetry.lock"
[ -f pyproject.toml ] || exit 0

# ¿Hay cambios de código sin confirmar, o confirmados en esta rama y no en main?
tocado=0
if [ -n "$(git status --porcelain -- $carpetas_codigo 2>/dev/null)" ]; then tocado=1; fi
if git rev-parse --verify -q origin/main >/dev/null 2>&1; then
  if ! git diff --quiet origin/main...HEAD -- $carpetas_codigo 2>/dev/null; then tocado=1; fi
fi
[ "$tocado" = "1" ] || exit 0

salida="$(bash scripts/comprobar.sh 2>&1)"
estado=$?
if [ "$estado" -ne 0 ]; then
  {
    echo "scripts/comprobar.sh ha fallado (código $estado). No se puede dar la issue por terminada hasta que pase. Salida:"
    printf '%s\n' "$salida" | tail -n 120
  } >&2
  exit 2
fi
exit 0
