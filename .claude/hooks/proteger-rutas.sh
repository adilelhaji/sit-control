#!/usr/bin/env bash
# PreToolUse (Edit|Write|MultiEdit). Bloquea con código 2 lo que no admite excepción:
#  - los archivos de secretos no se escriben desde Claude;
#  - no se pone ni se toca un umbral de cobertura (fail_under / --cov-fail-under) en la
#    configuración: ese umbral lo decide Adil, no una issue.
# Para una excepción consciente, Adil exporta SIT_PERMITIR_PROTEGIDOS=1 en su terminal.
aqui="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
entrada="$(cat)"
ruta="$(printf '%s' "$entrada" | bash "$aqui/leer-campo.sh" '.tool_input.file_path')"
[ -z "$ruta" ] && exit 0
[ "${SIT_PERMITIR_PROTEGIDOS:-0}" = "1" ] && exit 0

raiz="${CLAUDE_PROJECT_DIR:-$PWD}"
relativa="${ruta#"$raiz"/}"

bloquear() {
  echo "Bloqueado por .claude/hooks/proteger-rutas.sh: $1 ($relativa). Si es una excepción consciente, Adil exporta SIT_PERMITIR_PROTEGIDOS=1." >&2
  exit 2
}

case "$relativa" in
  .env|.env.*|*/.env|*/.env.*|*/secrets/*|*.pem|*.key)
    bloquear "los secretos no se escriben desde Claude"
    ;;
  .coveragerc|*/.coveragerc)
    bloquear "la configuración de cobertura no se toca desde una issue"
    ;;
  pyproject.toml|setup.cfg|tox.ini|.github/workflows/*|scripts/comprobar.sh)
    texto="$(printf '%s' "$entrada" | bash "$aqui/leer-campo.sh" '.tool_input.new_string')
$(printf '%s' "$entrada" | bash "$aqui/leer-campo.sh" '.tool_input.content')"
    if printf '%s' "$texto" | grep -Eqi 'fail[_-]under'; then
      bloquear "los umbrales de cobertura los decide Adil; no se ponen ni se tocan desde una issue"
    fi
    ;;
esac
exit 0
