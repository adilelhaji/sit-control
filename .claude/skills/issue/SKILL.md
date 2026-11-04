---
name: issue
description: Hace una issue de GitHub de este repositorio de principio a fin: elegir, sincronizar, explorar en paralelo, plan de la issue, implementar (contratos y pruebas primero), revisar con los dos revisores, abrir la PR y pasar a la siguiente. Uso: /issue 12, o /issue siguiente.
disable-model-invocation: true
argument-hint: <número de issue | siguiente>
---
Vas a hacer la issue `$ARGUMENTS` de principio a fin. Si el argumento es `siguiente`, elígela según el paso 1. Lee `CLAUDE.md` antes de nada (comandos, convenciones y lo que pasa por Adil).

## 1. Elegir

- Lista las issues abiertas (`gh issue list`). Si usan la etiqueta `listo`, elige entre esas la de número más bajo cuyas dependencias («depende de #N») estén fusionadas y que no tenga ya una PR abierta. Si no hay etiquetas, pregunta cuál.
- Si la issue es grande (toca varios módulos de `src/sit_control/` y sus pruebas, o cambia el modelo o los parámetros), para antes de tocar código y propón cómo partirla.

## 2. Sincronizar

`git fetch origin`; rama nueva desde `origin/main` llamada `issue-N-tres-palabras`. Si ya existe en `origin`, continúa sobre ella.

## 3. Explorar antes de tocar

Lanza a la vez, en segundo plano, y espera a los dos:

- el agente `explorador` con la issue (qué toca, dependencias, pruebas, patrón vecino, qué ya existe);
- un agente de coherencia (solo lectura) que compare la issue con el `README.md`, `docs/` y `configs/`, y diga si es coherente, qué contradice y qué le falta.

Con los informes decide:

- **Sin desvío**: sigue.
- **Desvío pequeño** (un nombre, un paso que falta, una prueba que no cubre el criterio): corrígelo y explícalo.
- **Desvío de rumbo o de arquitectura** (cambia el modelo matemático, valores de referencia, añade una dependencia o cambia una decisión de Adil): para y pregunta a Adil con dos o tres opciones (coste, riesgo, recomendación).

Deja en la issue un comentario **«Plan de la issue»** corto: qué se toca, qué no, qué pruebas lo demuestran y qué desvío se corrige. Termina el comentario con la línea de atribución de Claude Code.

## 4. Implementar

1. **Pruebas primero**: convierte cada criterio de aceptación en una prueba en `tests/`, ejecútala y confirma que falla. Commit.
2. **Código**: lo mínimo que las hace pasar, siguiendo el patrón vecino. Si la issue tiene dos o más partes independientes en archivos distintos, puedes lanzar un agente `implementador` por parte, cada uno en su propia copia de trabajo (worktree), con los contratos ya escritos. Si no, hazlo tú.
3. **Integra** y pasa `scripts/comprobar.sh` entero. Si una prueba estaba mal, se arregla la prueba con el criterio delante y se dice.

## 5. Revisar

Invoca `/revisar`. Arregla lo que **bloquea**; lo que **conviene**, si cabe en la issue, y si no, abre otra issue; lo **opcional** se anota en la PR. Tres rondas como máximo; a la cuarta, para y pregunta a Adil.

## 6. La PR

Invoca `/pr`. Mientras CI corre puedes explorar (solo lectura) la siguiente issue; si la PR vuelve con hallazgos o CI en rojo, la PR tiene prioridad.

## 7. Cerrar y seguir

La PR cierra la issue con `Closes #N`. Si la issue cambió una decisión o enseñó algo, déjalo escrito en `docs/` o el README en la misma PR. Si quedan issues y no has hecho cinco en esta sesión, di «siguiente: #N» y recomienda `/clear` antes de empezarla. Para si una issue necesita a Adil o si `main` está en rojo (arreglar `main` es lo primero).

## Lo que nunca haces

Bajar o excluir cobertura o fijar un umbral que Adil no ha decidido; saltar, apagar o marcar como lenta una prueba para que pase; commits vacíos para relanzar CI; push a `main`; fusionar en rojo.
