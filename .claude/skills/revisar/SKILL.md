---
name: revisar
description: Lanza en paralelo los dos revisores (código y negocio) y las comprobaciones sobre el diff de la rama actual contra origin/main, consolida los hallazgos por gravedad y deja claro qué bloquea. Se usa después de implementar y antes de abrir la PR; también vale para revisar cualquier rama.
argument-hint: [rama o PR] [issue]
---
Revisa el trabajo de la rama actual (o de `$ARGUMENTS`) como lo haría un equipo que no lo ha escrito.

1. **Prepara el material**: la issue (número, objetivo, criterios de aceptación), el diff contra `origin/main` (`git diff origin/main...HEAD`), y la salida de `scripts/comprobar.sh` (ejecútalo ahora si no la tienes; guarda las últimas líneas).
2. **Lanza a la vez**, en segundo plano, y espera a los dos:
   - el agente `revisor-codigo` con la issue, el diff y la salida de las comprobaciones;
   - el agente `revisor-negocio` con la issue y el diff.
   Ninguno de los dos recibe esta conversación: solo lo anterior. No les resumas lo que crees que está bien.
3. **Consolida**: junta los hallazgos, quita duplicados, ordena por gravedad (**bloquea** / **conviene** / **opcional**). Un criterio de aceptación sin prueba bloquea. Un valor de referencia o una afirmación científica del proyecto que el diff contradice bloquea. Una descripción que no coincide con el diff bloquea. Lo de estilo se descarta aunque un revisor lo haya reportado.
4. **Decide y actúa**: arregla lo que bloquea (tú, en serie, para no pisar archivos), vuelve a pasar `scripts/comprobar.sh` y relanza **a los dos** revisores con el diff nuevo. Lo que conviene: si cabe en la issue, arréglalo; si no, abre una issue nueva con el hallazgo y enlázala. Lo opcional se apunta para la PR.
5. **Tres rondas como máximo**. Si a la cuarta sigue habiendo algo que bloquea, para y pregunta a Adil con dos o tres opciones.
6. **Devuelve** (a la conversación, no a GitHub): la tabla de criterios del revisor de negocio, los hallazgos que se arreglaron, los que se dejaron y por qué, y la salida final de las comprobaciones. Eso va después al cuerpo de la PR.
