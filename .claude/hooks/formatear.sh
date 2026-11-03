#!/usr/bin/env bash
# PostToolUse (Edit|Write|MultiEdit). Formatea el archivo recién guardado con la herramienta
# del repositorio si existe; si no hay ninguna, no hace nada. Nunca falla la acción.
aqui="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
entrada="$(cat)"
ruta="$(printf '%s' "$entrada" | bash "$aqui/leer-campo.sh" '.tool_input.file_path')"
[ -z "$ruta" ] || [ ! -f "$ruta" ] && exit 0
raiz="${CLAUDE_PROJECT_DIR:-$PWD}"

case "$ruta" in
  *.py)
    # Solo formatea (ruff format). El orden de imports y el lint los revisa scripts/comprobar.sh.
    if [ -x "$raiz/.venv/bin/ruff" ]; then
      "$raiz/.venv/bin/ruff" format --quiet "$ruta" >/dev/null 2>&1 || true
    elif command -v ruff >/dev/null 2>&1; then
      ruff format --quiet "$ruta" >/dev/null 2>&1 || true
    fi
    ;;
esac
exit 0
