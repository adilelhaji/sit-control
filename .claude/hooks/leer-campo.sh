#!/usr/bin/env bash
# Lee un campo de la entrada JSON que Claude Code pasa a los hooks por stdin.
# Uso: valor=$(leer-campo.sh '.tool_input.file_path' <<< "$json")
# Prueba jq, luego python3, luego node. Si no hay ninguno, no devuelve nada.
campo="$1"
if command -v jq >/dev/null 2>&1; then
  jq -r "$campo // empty"
elif command -v python3 >/dev/null 2>&1; then
  python3 -c '
import sys, json
ruta = sys.argv[1].lstrip(".").split(".")
try:
    dato = json.load(sys.stdin)
    for parte in ruta:
        if parte:
            dato = dato.get(parte) if isinstance(dato, dict) else None
    if dato is not None and dato is not False:
        print(dato if not isinstance(dato, bool) else str(dato).lower())
except Exception:
    pass
' "$campo"
elif command -v node >/dev/null 2>&1; then
  node -e '
let entrada = "";
process.stdin.on("data", t => entrada += t).on("end", () => {
  try {
    const ruta = process.argv[1].replace(/^\./, "").split(".").filter(Boolean);
    let dato = JSON.parse(entrada);
    for (const p of ruta) dato = dato && typeof dato === "object" ? dato[p] : undefined;
    if (dato !== undefined && dato !== null) process.stdout.write(String(dato));
  } catch (e) {}
});
' "$campo"
fi
