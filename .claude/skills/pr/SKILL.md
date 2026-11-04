---
name: pr
description: Abre la PR de la rama actual con la plantilla del repositorio y la evidencia (comando y salida de las comprobaciones, figura o tabla si cambia un resultado), y se suscribe a la PR. Solo se usa al terminar una issue revisada.
disable-model-invocation: true
argument-hint: [número de issue]
---
Abre la PR de la rama actual para la issue `$ARGUMENTS` (si no se da, la de la rama `issue-N-...`).

Antes de abrirla, comprueba y no sigas si algo falla:

1. `git fetch origin` y que la rama esté al día con `origin/main` (si no, trae `main` con un merge, nunca con rebase sobre lo publicado, y vuelve a pasar las comprobaciones).
2. `scripts/comprobar.sh` en verde. Guarda el comando y las últimas 30 líneas de su salida: son la **evidencia**.
3. Si la issue cambia un resultado numérico, una tabla o una figura, el antes y el después (comando y valores) en la descripción.
4. Que `/revisar` se haya pasado y no quede nada que bloquee.
5. El diff tiene menos de unas 400 líneas; si no, di que la issue estaba mal partida y propón cómo partirla, en lugar de abrir una PR grande.
6. Que la descripción que vas a escribir **coincida con el diff**: ni promete lo que no hace ni oculta lo que hace.

Después:

7. Push de la rama (`git push -u origin <rama>`).
8. Abre la PR **lista para revisión (no borrador)** con el título en imperativo y el cuerpo según `.github/pull_request_template.md`: qué, `Closes #N`, por qué, evidencia, qué se probó, qué queda fuera, desvíos y revisión, lo que necesita a Adil (si toca una ruta de `CODEOWNERS`, cuál). Termina el cuerpo con las líneas de atribución que pide la sesión.
9. **Fusiona** (squash) si `/revisar` no dejó nada que bloquee y las comprobaciones están en verde, en GitHub si las hay y con `scripts/comprobar.sh` si no. Espera a que terminen: `--auto` fusiona al momento si el repositorio no exige ninguna. Adil decidió el 3-11-2026, y confirmó el 4-11-2026, que solo aprueba los cambios de arquitectura y los desvíos de la propuesta original: en esos casos paras y le preguntas. Si la PR toca una ruta de `CODEOWNERS`, lo dices en el cuerpo, pero no esperas.
10. Suscríbete a la actividad de la PR y responde a lo que llegue (CI en rojo, comentarios) según las reglas de esa suscripción.
11. Devuelve a la conversación el enlace y, en una línea, si ya está fusionada, si espera a las comprobaciones o qué desvío espera a Adil.
