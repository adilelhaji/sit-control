#!/usr/bin/env bash
# PreToolUse (Bash). Bloquea con código 2 los comandos de git que no admiten excepción:
#  - push a main o master, push forzado, --no-verify, borrar main, reset duro contra origin/main.
# Lo demás (incluido push a la rama de la issue) pasa; git push sigue pidiendo permiso.
aqui="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
entrada="$(cat)"
cmd="$(printf '%s' "$entrada" | bash "$aqui/leer-campo.sh" '.tool_input.command')"
[ -z "$cmd" ] && exit 0

bloquear() {
  echo "Bloqueado por .claude/hooks/proteger-main.sh: $1. Comando: $cmd" >&2
  exit 2
}

# Normaliza espacios para que las expresiones sean sencillas.
plano="$(printf '%s' "$cmd" | tr -s '[:space:]' ' ')"

if printf '%s' "$plano" | grep -Eq 'git push[^;&|]*( -f| --force| --force-with-lease)'; then
  bloquear "no se fuerza un push"
fi
if printf '%s' "$plano" | grep -Eq 'git push[^;&|]*(origin|upstream) +(main|master)( |$|:)'; then
  bloquear "no se hace push a main: issue → rama → PR"
fi
if printf '%s' "$plano" | grep -Eq 'git push[^;&|]*:(main|master)( |$)'; then
  bloquear "no se hace push a main: issue → rama → PR"
fi
if printf '%s' "$plano" | grep -Eq 'git (commit|push)[^;&|]*--no-verify'; then
  bloquear "no se saltan las comprobaciones con --no-verify"
fi
if printf '%s' "$plano" | grep -Eq 'git branch[^;&|]* -[dD] +(main|master)( |$)'; then
  bloquear "no se borra main"
fi
if printf '%s' "$plano" | grep -Eq 'git reset --hard[^;&|]*origin/(main|master)'; then
  bloquear "no se tira el trabajo con reset --hard contra origin/main; usa una rama nueva"
fi
exit 0
