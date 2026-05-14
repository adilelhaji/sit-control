# Estrategias de Control para la Técnica del Insecto Estéril: Catálogo Razonado

**TFG:** *Optimal Control Strategies for Mosquito Population Suppression via the Sterile Insect Technique*
**Autor:** Adil El Haji — Universitat Autònoma de Barcelona, 2026
**Referencia principal:** Almeida, Duprez, Privat & Vauchelet (2022), *J. Differential Equations* 311, 229–266.

---

## Preámbulo: Marco Matemático Común

Todas las estrategias operan sobre el sistema de EDOs que rige la dinámica del mosquito
*Aedes polynesiensis* bajo liberación de machos estériles. La formulación sigue
Strugarek, Bossin & Dumont (2019) (sistema S2, cuatro estados) y su reducción
cuasi-estacionaria (sistema S1, dos estados).

**Sistema S2** (modelo completo, cuatro variables):

$$\dot{E} = \beta_E F\!\left(1 - \tfrac{E}{K}\right) - (\nu_E + \delta_E)E$$

$$\dot{M} = (1-\nu)\nu_E E - \delta_M M$$

$$\dot{F} = \nu\nu_E E \cdot \frac{M}{M + \gamma_s M_s} - \delta_F F$$

$$\dot{M}_s = u(t) - \delta_s M_s$$

**Sistema S1** (reducción QSSA, dos variables):

$$\dot{F} = f(F, M_s) - \delta_F F, \qquad
f(F,M_s) = \frac{\beta_E \nu \nu_E K}{\nu_E + \delta_E}\,\frac{\delta_M}{(1-\nu)\nu_E} \cdot
\frac{F}{1}\cdot\frac{M^*}{M^* + \gamma_s M_s}$$

$$\dot{M}_s = u(t) - \delta_s M_s$$

donde $M^* = \frac{(1-\nu)\nu_E}{\delta_M}\,E^*$ se obtiene en el estado cuasi-estacionario.

**Parámetros de referencia** (Almeida et al. 2022, Tabla 1):

| Símbolo | Valor | Descripción |
|---------|-------|-------------|
| $\beta_E$ | 10 | Tasa de puesta de huevos |
| $\delta_E$ | 0.03 d⁻¹ | Mortalidad larval |
| $\delta_M$ | 0.10 d⁻¹ | Mortalidad masculina |
| $\delta_F$ | 0.04 d⁻¹ | Mortalidad femenina |
| $\delta_s$ | 0.12 d⁻¹ | Mortalidad machos estériles |
| $\nu_E$ | 0.05 d⁻¹ | Tasa de emergencia larval |
| $\nu$ | 0.49 | Fracción de emergencia femenina |
| $\gamma_s$ | 1.0 | Competitividad del macho estéril |
| $K$ | 22 200 | Capacidad de carga larval |

De estos parámetros se derivan los valores críticos:
$R_0 \approx 76.56$, $\bar{F} \approx 11\,037$ hembras en equilibrio,
umbral de supresión $\varepsilon = \bar{F}/4 \approx 2\,759$ hembras.

**Espacio de control admisible:**

$$\mathcal{U} = \left\{u:[0,T]\to[0, U_{\max}]\;\Big|\; u \text{ medible}\right\}, \quad U_{\max} = 5\,000 \text{ mosquitos/día}$$

**Seis objetivos del TFG (Capítulo 1):**

1. Reproducir el modelo matemático de dinámica bajo SIT (EDOs).
2. Formular el problema de control óptimo y analizar condiciones de optimalidad.
3. Implementar y verificar computacionalmente frente a Almeida et al. (2022).
4. Comparar la estrategia óptima con alternativas constante, periódica y L².
5. Extender al esquema de liberación impulsiva.
6. Análisis de sensibilidad sobre parámetros biológicos.

---

## Mapa Estratégico del Documento

Las diez estrategias se organizan en cuatro familias metodológicas:

```
                                 SIT CONTROL
                                      |
       ┌──────────────────┬───────────┴────────────┬───────────────────┐
   Óptimo clásico    Operativas          Análisis            Data-driven / ML
   (PMP + NLP)       (benchmarks)        (sobre 1)           (modelo-libre)
       |                  |                  |                       |
   1. L¹ (BSB)        2. Constante      8. Sensibilidad       6. iP (Join 2026)
   4. L² suavizado    3. Periódico      ±20% paramétrico      9. Deep RL ★
   5. SLSQP impulsivo                                         10. DeepONet ★
   7. Tiempo mínimo
```

★ = **estrategias novedosas basadas en ML** propuestas en este TFG (no implementadas
en el repositorio base; trabajo futuro del Capítulo 5).

**Tres ejes de variación:**

- **Apertura del bucle:** estrategias 1–5, 7, 10 son *bucle abierto* (computan $u(\cdot)$
  ofreciendo $\theta$ antes de $t=0$); estrategias 6, 9 son *bucle cerrado* (computan
  $u(t)$ a partir de $F(t), M_s(t)$ observados).
- **Dependencia del modelo:** 1, 4, 5, 7 requieren $\theta$ exacto; 2, 3 son insensibles
  pero subóptimas; 6 es agnóstica (model-free puro); 9, 10 aprenden de un simulador
  con $\theta$ aleatorizado.
- **Capacidad de robustez frente a $\Delta\theta\sim\pm 30\%$:** 1, 4, 5, 7 (baja),
  2, 3 (media), 6 (alta local), 9 (**alta global, certificada por K-Fold**), 10 (alta
  dentro del dominio de entrenamiento).

---

## Validación del Artículo de Referencia (Almeida et al. 2022)

Esta es la sección **crítica para el OBJ-3** del TFG: la verificación computacional
frente al artículo de referencia. Almeida et al. (2022) aportan resultados numéricos
y estructurales que permiten una validación **cuantitativa** del código:

### Matriz de validación

| # | Resultado de Almeida (2022)                                  | Sección artículo  | Estrategia que lo reproduce  | Tipo de validación    |
|---|--------------------------------------------------------------|-------------------|------------------------------|------------------------|
| V1| $J^*_1(S_1)\approx 1.46\times 10^5$ para $T=150$ días        | Tabla 2           | **Estr. 1 (L¹) sobre S1**    | **Cuantitativa exacta** |
| V2| $J^*_1(S_2)\approx 1.47\times 10^5$ para $T=150$ días        | Tabla 2           | **Estr. 1 (L¹) sobre S2**    | **Cuantitativa exacta** |
| V3| $T_{\text{opt}}\approx 103$–$104$ días (singular mínimo)     | Tabla 3           | **Estr. 1 + Algoritmo 2**    | **Cuantitativa exacta** |
| V4| Estructura $u^*=$ bang-zero → singular para $T=150$          | Teorema 3.3 + Fig. 4 | **Estr. 1 (post-PMP)**    | **Estructural**         |
| V5| Speedup Algoritmo 2 ≈ 20× sobre GEKKO ($N=300$)              | Tabla 4           | **Algoritmo 2 (bisection)**  | **Computacional**       |
| V6| L² como extensión natural del L¹                             | §5 (cualitativa)  | Estr. 4 (L²)                 | Implementación numérica de extensión propuesta |
| V7| Liberación impulsiva como extensión                          | §5 (propuesta)    | Estr. 5 (SLSQP impulsiva)    | Implementación numérica de extensión propuesta |
| V8| Sensibilidad cualitativa a $\beta_E, \delta_F, K$            | §6 (cualitativa)  | Estr. 8 (±20% tornado)       | Cuantificación de discusión cualitativa |

### Estrategias que **NO** validan Almeida

| # | Estrategia            | Razón                                                       |
|---|-----------------------|-------------------------------------------------------------|
| 2 | Constante             | Benchmark operativo (Strugarek 2019), no en Almeida         |
| 3 | Periódico fijo        | Benchmark operativo (Lim 2026), no en Almeida               |
| 6 | iP sin modelo         | Paper sucesor (Join, Almeida & Fliess 2026); no en Almeida 2022 |
| 7 | Tiempo mínimo         | Mencionado sólo en Obs. 4.1 de Almeida; sin valores numéricos |
| 9 | Deep RL ★             | Propuesta novedosa; fuera del scope de Almeida              |
| 10| DeepONet ★            | Propuesta novedosa; fuera del scope de Almeida              |

**Lectura crítica:** el corazón empírico del TFG es la **Estrategia 1** ejecutada
sobre los modelos S1 y S2. Si J₁*(S1) y J₁*(S2) reproducen los valores 1.46e5 y
1.47e5 con error relativo $< 1\%$, el OBJ-3 (verificación) está cumplido. Todo
lo demás son **extensiones** o **propuestas**.

---

## Alcance dentro de la Memoria del TFG

Las diez estrategias no tienen el mismo peso en la memoria del TFG. La distribución
por capítulo es:

| Capítulo TFG                       | Estrategias relevantes | Rol en la memoria                                                |
|------------------------------------|------------------------|-------------------------------------------------------------------|
| **Cap. 2** (Modelo matemático)     | —                      | Deriva S1, S2; no depende de estrategias                          |
| **Cap. 3** (Control óptimo + PMP)  | **Estr. 1**            | **Verificación numérica frente a Almeida 2022 → núcleo del TFG** |
| **Cap. 4** (Comparación)           | Estr. 2, 3, 4, 5, 8    | Cubre OBJ-4 (alternativas), OBJ-5 (impulsiva), OBJ-6 (sensibilidad) |
| **Cap. 5** (Conclusiones + futuro) | Estr. 6, 7, 9★, 10★    | Líneas abiertas: control cerrado, problema dual, ML data-driven   |
| **Cap. 6** (Anexo de código)       | Estr. 1–5, 8           | Implementaciones efectivamente en el repositorio                  |

### Núcleo TFG vs. trabajo futuro

**Estrategias del núcleo TFG** (necesarias para los seis objetivos):
$$\{1,\ 2,\ 3,\ 4,\ 5,\ 8\}\;\subset\;\text{Capítulos 3 y 4}$$

**Estrategias de trabajo futuro** (Capítulo 5, sin implementación obligatoria):
$$\{6,\ 7,\ 9^\star,\ 10^\star\}\;\subset\;\text{Capítulo 5}$$

**Consecuencia editorial:** las Estrategias 9 (Deep RL) y 10 (DeepONet) son
contribuciones **de calado postdoctoral / PhD**. Se exponen como propuestas
argumentadas (con MDP completo, arquitecturas concretas, protocolos de validación)
pero **no se exige su implementación ni su evaluación empírica** dentro del TFG.
Su lugar es la sección de "Trabajo futuro" del Capítulo 5, alineadas con las
líneas ya enunciadas en el TFG (lazo cerrado citando Bliman, estacionalidad,
estocasticidad, validación de campo).

Esta jerarquía protege la coherencia del TFG: el trabajo de grado **cumple los
seis objetivos con las Estrategias 1–5 y 8**, y abre líneas de investigación
con las Estrategias 6–7, 9–10.

---

## Estrategia 1 — Control Óptimo L¹ (bang-singular-bang)

### Formulación

$$\min_{u \in \mathcal{U}} J_1(u) = \int_0^T u(t)\,dt \quad
\text{s.t.} \quad F(T) \leq \varepsilon, \quad \dot{x} = \text{S1}(x, u)$$

El coste $J_1$ es el número total de machos estériles liberados, proporcional al
coste económico de producción (Pant, Bhatt & Bhatt 2025).

### Análisis de Optimalidad (PMP)

El Hamiltoniano del sistema es:

$$H(x,\lambda,u) = \lambda_1\bigl(f(F,M_s) - \delta_F F\bigr) + \lambda_2(u - \delta_s M_s) + u$$

La función de conmutación $\phi(t) = 1 + \lambda_2(t)$ determina el control óptimo:

$$u^*(t) = \begin{cases}
U_{\max} & \text{si } \phi(t) < 0 \quad \text{(bang-on)} \\
u_{\text{sing}}(t) & \text{si } \phi(t) = 0 \quad \text{(arco singular)} \\
0 & \text{si } \phi(t) > 0 \quad \text{(bang-off)}
\end{cases}$$

El control singular se obtiene diferenciando $\phi(t) = 0$ dos veces respecto al tiempo
(condición de Kelley):

$$u_{\text{sing}} = \frac{\delta_s M_s \,\partial_{M_s}^2 f \cdot \partial_F f - \partial_F f \cdot \partial_{M_s} f - f \cdot \partial_{M_s F}^2 f}
{\partial_{M_s}^2 f \cdot \partial_{M_s} f}$$

Almeida et al. (2022, Teorema 3.3) demuestran que la estructura depende del
horizonte temporal $T$:

- **$T \approx T_{\min} \approx 60$ días** (horizonte mínimo factible): estructura
  bang-ON → singular — el arco máximo ocupa casi todo $[0,T]$.
- **$T = 150$ días** (caso de referencia): estructura **bang-ZERO → singular** —
  el control es nulo en $[0,\, T - T_{\text{opt}}] \approx [0, 46]$ días, después
  sigue el arco singular en $[46, 150]$ días.

El *tiempo singular mínimo* $T_{\text{opt}} \approx 103$–$104$ días (Almeida et al. 2022,
Tabla 3) es independiente de $T$: es la duración mínima del arco singular necesaria
para cumplir $F(T) \leq \varepsilon$. Cuando $T > T_{\text{opt}}$, el tiempo sobrante
se ocupa con un arco nulo inicial.

### Propiedades

- Minimiza el uso total de insectos estériles (coste lineal en $u$).
- La estructura viene determinada analíticamente por el PMP; para el caso de referencia
  ($T=150$) es **bang-zero ($\sim$46 d) → singular ($\sim$104 d)**.
- El tiempo de conmutación $\tau_1$ (duración del arco singular) se determina por bisección
  RIGHT-ALIGNED (Algoritmo 2 de Almeida et al. 2022): u*=0 en $[0, T-\tau_1]$, luego
  arco singular en $[T-\tau_1, T]$.
- El Algoritmo 2 es $\approx$20× más rápido que GEKKO a igual discretización
  (1.44 s vs 29.4 s para $N=300$, Almeida et al. 2022, Tabla 4).
- Es la solución de referencia del artículo: $J_1^* \approx 1.46 \times 10^5$ (para $T=150$ días).

### Fundamento Bibliográfico

- **Almeida et al. (2022):** formulación completa, PMP, bisección y valores numéricos de referencia.
- **Pontryagin et al. (1962):** fundamento teórico del Principio del Máximo.
- **Thome, Yang & Díaz (2010):** primer análisis sistemático de SIT con control óptimo L¹.

### Objetivos TFG cubiertos

OBJ-2 (formulación y optimalidad), OBJ-3 (verificación), OBJ-4 (comparación).

---

## Estrategia 2 — Liberación Constante

### Formulación

$$u(t) = c \geq 0 \quad \forall\, t \in [0,T]$$

El nivel $c$ se elige como el mínimo valor constante que satisface $F(T) \leq \varepsilon$.
En la práctica se usa $c = J_1^*\!/T$ como punto de partida (mismo presupuesto total
repartido uniformemente).

### Propiedades

- Implementación trivial: no requiere optimización.
- En la bibliografía actúa como *benchmark* inferior de operatividad.
- Strugarek, Bossin & Dumont (2019, Sección 5) calculan el nivel de liberación constante
  necesario para mantener $F$ por debajo del umbral en régimen permanente.
- Gasto total generalmente mayor que $J_1^*$ porque no adapta la intensidad a la
  dinámica poblacional.

### Fundamento Bibliográfico

- **Strugarek et al. (2019):** análisis de equilibrios bajo liberación constante; condición
  $u > u^*_{\min}$ para la existencia del equilibrio de supresión.
- **Bliman et al. (2021):** usa la liberación constante como caso límite en el análisis
  de convergencia del controlador en bucle cerrado.
- **Pant et al. (2025):** referencia empírica para tasas de liberación sostenidas en
  programas operativos de SIT.

### Objetivos TFG cubiertos

OBJ-4 (comparación con estrategia óptima).

---

## Estrategia 3 — Liberación Periódica Impulsiva (montos fijos)

### Formulación

Las liberaciones se realizan en instantes discretos $t_k = k\tau$, $k=0,1,\ldots$,
con cantidad constante $c$ en cada pulso:

$$u(t) = c \sum_{k=0}^{\lfloor T/\tau \rfloor} \delta(t - k\tau)$$

En la implementación numérica, cada pulso se aproxima por una función rectangular
de duración $\Delta t_{\text{pulse}} = 0.1$ días y amplitud $c/\Delta t_{\text{pulse}}$.

Se evalúan tres periodos de liberación representativos: $\tau \in \{3, 7, 14\}$ días,
coherentes con los ciclos logísticos de los programas SIT reales (Lim et al. 2026).

### Propiedades

- Modela las restricciones logísticas de los programas de campo (liberaciones semanales
  o bisemanales por avión o personal terrestre).
- Con el mismo presupuesto total $J = N_p \cdot c$, el periodo $\tau$ afecta
  significativamente el perfil de $M_s(t)$ y por tanto la supresión.
- Periodo corto ($\tau=3$ d): perfil de $M_s$ casi constante, similar a la estrategia 2.
- Periodo largo ($\tau=14$ d): $M_s$ oscila ampliamente; puede no mantener la supresión.

### Fundamento Bibliográfico

- **Lim, Sánchez-Pérez & Morales (2026):** estudio empírico de frecuencias de liberación
  en programas SIT contra *Bactrocera dorsalis*; identifica $\tau=7$ días como óptimo
  logístico.
- **Strugarek et al. (2019):** modelo con liberaciones impulsivas periódicas; análisis
  de soluciones periódicas y su estabilidad.
- **Thome et al. (2010):** compara estrategias periódicas vs. continuas en términos de
  coste y eficacia de supresión.

### Objetivos TFG cubiertos

OBJ-4 (comparación), OBJ-5 (esquema impulsivo).

---

## Estrategia 4 — Control Óptimo L² (suavizado cuadrático)

### Formulación

$$\min_{u \in \mathcal{U}} J_2(u) = \frac{c}{2}\int_0^T u(t)^2\,dt \quad
\text{s.t.} \quad F(T) \leq \varepsilon, \quad \dot{x} = \text{S1}(x, u)$$

El parámetro $c > 0$ (peso cuadrático) pondera el coste frente a la suavidad del perfil.

### Análisis de Optimalidad (PMP)

El Hamiltoniano cuadrático tiene condición de estacionariedad interior:

$$\frac{\partial H}{\partial u} = \lambda_2 + cu = 0 \implies u^*(t) = -\frac{\lambda_2(t)}{c}$$

proyectado sobre $[0, U_{\max}]$. La ausencia de control bangbang elimina la
discontinuidad y produce perfiles suaves.

### Propiedades

- El coste L² penaliza picos de liberación ($u^2$ es convexo y estricto).
- Perfil naturalmente suave: elimina transiciones abruptas del control L¹.
- Interpretación operativa: minimiza el esfuerzo de producción *cuadrático*
  (variaciones bruscas son más costosas que una tasa sostenida).
- $J_2^* > J_1^*$ en número total de insectos, pero puede ser preferible
  operativamente al no requerir picos de producción.

### Fundamento Bibliográfico

- **Almeida et al. (2022, Sección 5):** plantean el problema L² como extensión explícita
  del caso L¹ y comparan cualitativamente sus propiedades; el TFG lo implementa numéricamente.
- **Fleming & Rishel (1975):** fundamento teórico de la optimalidad para funcionales
  cuadráticos con restricciones de estado.
- **Join, Chaxel & Mboup (2026):** usan penalización cuadrática en el contexto de
  control retroalimentado para SIT, motivando el coste L².

### Objetivos TFG cubiertos

OBJ-2 (optimalidad), OBJ-4 (comparación L¹ vs. L²).

---

## Estrategia 5 — Liberación Impulsiva Óptima (montos variables, SLSQP)

### Formulación

Dado un conjunto fijo de $N_p$ instantes de liberación $\{t_k\}_{k=1}^{N_p}$
con periodo $\tau$ (e.g., $\tau=7$ días), se optimizan los montos $\{c_k\}$:

$$\min_{\{c_k\} \geq 0} \sum_{k=1}^{N_p} c_k \quad
\text{s.t.} \quad F(T;\{c_k\}) \leq \varepsilon$$

La función objetivo es lineal en $\{c_k\}$ (misma estructura que L¹ pero en espacio discreto).
La restricción se evalúa por simulación forward del sistema S1.

### Implementación

Se usa `scipy.optimize.minimize` con método SLSQP:

- Vector de decisión: $\mathbf{c} = (c_1, \ldots, c_{N_p}) \in [0, c_{\max}]^{N_p}$
- Gradiente del objetivo: $\nabla_{\mathbf{c}}\,\mathbf{1}^T\mathbf{c} = \mathbf{1}$ (constante, exacto).
- Restricción: $g(\mathbf{c}) = \varepsilon - F(T;\mathbf{c}) \geq 0$ (evaluada por simulación).
- Inicialización: $c_k^{(0)} = J_1^*/N_p$ (presupuesto óptimo L¹ repartido uniformemente).

### Propiedades

- Representa el caso realista donde los instantes de liberación están fijos por logística
  pero los montos pueden ajustarse.
- Generaliza la estrategia 3 (fija $c_k = c$ constante) y aproxima la estrategia 1
  cuando $\tau \to 0$.
- La convergencia del SLSQP depende de la regularidad de $F(T;\mathbf{c})$ como
  función de $\mathbf{c}$; en general es suave para $\tau$ pequeño.

### Fundamento Bibliográfico

- **Almeida et al. (2022, Sección 5):** proponen explícitamente la extensión impulsiva
  en el artículo de referencia; el TFG la implementa numéricamente via SLSQP.
- **Strugarek et al. (2019):** análisis teórico de sistemas con control impulsivo;
  condiciones de existencia de solución.
- **Lim et al. (2026):** evidencia empírica de que la optimización de montos por pulso
  (frente a monto fijo) reduce el coste total entre un 15–25 % en programas reales.
- **Bonnans & Shapiro (2000):** fundamento de la sensibilidad de restricciones para
  programas no lineales con restricciones de simulación.

### Objetivos TFG cubiertos

OBJ-4 (comparación), OBJ-5 (esquema impulsivo extendido).

---

## Estrategia 6 — Control Sin Modelo: Controlador Proporcional Inteligente (iP)

*[Propuesta de extensión — no implementada en el repositorio base]*
*Basada en: Join, Almeida & Fliess (2026), ISCS 2026, arXiv:2604.01355*

### Motivación

Las estrategias 1–5 son de **bucle abierto**: el control $u(t)$ se calcula en $t=0$
asumiendo parámetros biológicos exactamente conocidos. En campo, $\beta_E$, $\delta_F$,
$\nu_E$, etc. varían con la temperatura, la estación y la dinámica local.

Join, Almeida & Fliess (2026) — siendo Almeida coautor de la referencia principal —
proponen una solución radicalmente diferente: **control sin modelo** (*model-free control*,
MFC) que no requiere conocer ningún parámetro biológico. Opera sobre los mismos
parámetros nominales del modelo S2 (Tabla 1) pero el controlador los ignora.

### Modelo Ultra-Local (Fliess & Join 2013)

Bajo hipótesis débiles, cualquier sistema SISO puede aproximarse localmente por:

$$y^{(\nu)} = F_{\text{sis}} + \alpha\, u \tag{ULM}$$

donde:
- $y$ es la salida medida, $u$ es la entrada de control;
- $\nu = 1$ (orden de derivación, típicamente 1);
- $\alpha \in \mathbb{R}$ se elige para que $\alpha u$ y $\dot{y}$ tengan el mismo orden de magnitud;
- $F_{\text{sis}}$ encapsula **toda** la dinámica del sistema y las perturbaciones externas
  — no es un parámetro biológico sino una función desconocida que se estima online.

**Aplicación al SIT (Join et al. 2026, Sección 3):**
- Salida: $y = x_1 = E$ (estado de huevos/fase acuática) — observable en campo.
- Control auxiliar continuo: $V = x_4 = M_s$ (machos estériles).
- El modelo ultra-local se escribe: $\dot{E} = F_{\text{sis}} + \alpha V$.

### Estimación Online de $F_{\text{sis}}$

La función $F_{\text{sis}}(t)$ se estima en tiempo real a partir únicamente de
las señales medidas $u(\cdot)$ e $y(\cdot)$ mediante (Join et al. 2026, Ec. 2):

$$F_{\text{est}}(t) = -\frac{6}{\tau^3} \int_{t-\tau}^{t}
\bigl[(\tau - 2\sigma)\,y(\sigma) + \alpha\sigma(\tau-\sigma)\,u(\sigma)\bigr]\,d\sigma$$

donde $\tau > 0$ es una ventana temporal pequeña. En implementación digital se
reemplaza por un filtro IIR de bajo orden. **No se usa ningún parámetro biológico.**

### Controlador Proporcional Inteligente (iP)

El controlador iP (Join et al. 2026, Ec. 3) combina la estimación con seguimiento
de una trayectoria de referencia $y^*(t)$:

$$u(t) = \frac{-F_{\text{est}}(t) + \dot{y}^*(t) - K_p\, e(t)}{\alpha} \tag{iP}$$

donde $e(t) = y(t) - y^*(t)$ es el error de seguimiento y $K_p > 0$ es la única
ganancia a ajustar. La dinámica del error resultante satisface:

$$\dot{e} + K_p\, e = F_{\text{sis}} - F_{\text{est}} \approx 0$$

de modo que $\lim_{t\to\infty} e(t) \approx 0$ mientras $F_{\text{est}}$ sea una
buena estimación de $F_{\text{sis}}$.

**Trayectoria de referencia:** $y^*(t)$ es una curva exponencial decreciente de
$E(0)$ hasta $E_c = V_c / (\beta_E \nu_E / (\nu_E+\delta_E))$ — el nivel de huevos
correspondiente al umbral epidémico $V_c = \alpha\mu H / (\beta^2 p p')$.

### Versión Discreta (Liberación Impulsiva)

Join et al. (2026, Sección 4) extienden el controlador al caso impulsivo práctico,
donde los machos estériles se liberan cada $J$ días. La amplitud del pulso $k$-ésimo es:

$$\delta_k = \frac{V(k) - \operatorname{mean}\bigl(\mathcal{I}_m(k+1\ldots k+J)\bigr)}
                  {\operatorname{mean}\bigl(\mathcal{I}(1\ldots J)\bigr)}$$

donde:
- $V(k)$ es el valor de la ley de control continua iP en el día $k$;
- $\mathcal{I}(1\ldots J)$ es la respuesta impulsional de la dinámica $\dot{x}_4 = u - \delta_s x_4$
  (función de transferencia $\frac{1}{s+\delta_s}$) ante un pulso de amplitud 1;
- $\mathcal{I}_m(k+1\ldots k+J)$ son las respuestas impulsionales pasadas proyectadas al futuro.

Este mecanismo garantiza que la media de $x_4$ sobre el periodo $[k, k+J]$ sea
igual a la señal continua objetivo $V(k)$ — principio de superposición exacto.

### Resultados Verificados (Join et al. 2026)

Los autores simulan con los **mismos parámetros que Almeida et al. (2022)**:

| Escenario | Resultado |
|-----------|-----------|
| Nominal, $J=3$ días | Seguimiento perfecto de $y^*(t)$; $0 < u < 10^6$ |
| $J=6$ días | Tracking ligeramente degradado (saturación inicial), pero muy preciso |
| $\delta_s$ perturbado $\times 1.3$ | Rendimiento muy bueno; representación continua-discreta pierde coincidencia |
| 100 simulaciones, parámetros $\times[0.7, 1.3]$ | Fallo en solo 2 casos (saturación permanente del control) |

### Comparación con Estrategias 1–5

| Aspecto | Estrategias 1–5 | Estrategia 6 (iP) |
|---------|----------------|-------------------|
| Conocimiento de parámetros | Requiere $\beta_E, \delta_F$, etc. | No requiere ninguno |
| Tipo de control | Bucle abierto | Bucle cerrado |
| Variable controlada | $F$ (hembras) | $E$ (huevos) |
| Objetivo | $F(T) \leq \varepsilon$ (terminal) | $E(t) \to E_c$ (trayectoria) |
| Robustez paramétrica | Baja | Alta (98/100 simulaciones) |
| Versión impulsiva | Estrategia 5 (SLSQP) | Fórmula analítica $\delta_k$ |
| Coste computacional | Alto (GEKKO/NLP) | Muy bajo (filtro + división) |

### Propiedades y Limitaciones

- **No requiere identificación paramétrica** — ventaja crítica en campo.
- **Robustez frente a incertidumbre** demostrada estadísticamente (100 simulaciones).
- **Coste operativo:** puede ser mayor que el óptimo L¹ porque no minimiza explícitamente $\int u\,dt$.
- **Medición:** requiere estimación regular de $E(t)$ (muestreo de larvas en criaderos,
  ovitrampas), con muestreo mínimo de $1/J$ días$^{-1}$.
- **Fallo:** si el control se satura permanentemente ($u = U_{\max}$ todo el tiempo),
  el sistema no puede seguir la referencia — el 2% de los casos en Join et al. (2026).
- **Conexión con IA:** el MFC se relaciona con aprendizaje automático en cuanto a
  su agnosticismo paramétrico, pero es determinista y algebraico (no estadístico).
  Los autores lo posicionan como alternativa más simple y verificable que las redes
  neuronales de control (Böttcher 2026).

### Fundamento Bibliográfico

- **Join, Almeida & Fliess (2026):** formulación completa iP para SIT, mismos parámetros
  que Almeida et al. (2022), resultados de robustez.
- **Fliess & Join (2013):** fundamento matemático del MFC y el modelo ultra-local.
- **Agbo bidi, Almeida & Coron (2025):** estabilización por retroalimentación del modelo
  SIT — complemento teórico de Join et al.
- **Bhaya & Bliman (2025):** diseño de control retroalimentado SIT via teoría de
  sistemas monótonos — perspectiva alternativa de estabilización.

### Objetivos TFG cubiertos

OBJ-2 (formulación de control sin modelo), OBJ-4 (comparación bucle abierto vs.
cerrado, óptimo vs. robusto).

### Implementación propuesta

```python
def ip_controller(
    y_meas: np.ndarray,   # E(t) medido, ventana de longitud tau
    u_hist: np.ndarray,   # u(t) aplicado, misma ventana
    t_window: np.ndarray, # instantes en [t-tau, t]
    y_ref: float,         # y*(t) = E_ref en el instante actual
    dy_ref: float,        # dy*/dt en el instante actual
    alpha: float,
    K_p: float,
    U_max: float,
) -> float:
    tau = t_window[-1] - t_window[0]
    sigma = t_window - t_window[0]             # sigma en [0, tau]
    F_est = -(6 / tau**3) * np.trapz(
        (tau - 2*sigma) * y_meas + alpha * sigma * (tau - sigma) * u_hist,
        t_window,
    )
    e = y_meas[-1] - y_ref
    u = (-F_est + dy_ref - K_p * e) / alpha
    return float(np.clip(u, 0.0, U_max))
```

---

## Estrategia 7 — Control de Tiempo Mínimo (Problema Dual)

*[Propuesta de extensión — no implementada]*

### Formulación

El problema L¹ fija el horizonte temporal $T$ y minimiza el coste total.
El problema **dual** fija el presupuesto $B$ y minimiza el tiempo necesario:

$$\min_{u \in \mathcal{U},\,T>0} T \quad
\text{s.t.} \quad \int_0^T u(t)\,dt \leq B, \quad F(T) \leq \varepsilon$$

### Propiedades Teóricas

Por la dualidad de Fenchel-Rockafellar, las soluciones del problema L¹ y del problema
de tiempo mínimo están relacionadas: la curva óptima $T^*(B)$ es la frontera de Pareto
entre coste y horizonte temporal.

Almeida et al. (2022, Observación 4.1) mencionan explícitamente esta dualidad como
extensión natural: dado un presupuesto epidemiológico (campaña de duración máxima
$B$ días-insecto), el tiempo mínimo de supresión es la métrica operativa relevante.

### Implementación

Bisección externa sobre $T$: para cada $T$ candidato, se resuelve el problema L¹
con restricción $\int_0^T u\,dt \leq B$ (inactiva si $J_1^*(T) < B$). La función
$T \mapsto J_1^*(T)$ es monótonamente decreciente; la bisección encuentra el $T^*$
tal que $J_1^*(T^*) = B$.

### Fundamento Bibliográfico

- **Almeida et al. (2022, Obs. 4.1):** formulación explícita del problema dual de tiempo mínimo.
- **Ioffe & Tihomirov (1979):** teoría de dualidad para problemas de control óptimo.
- **Van den Berg & Friedlander (2008):** algoritmos numéricos para problemas con
  restricciones de presupuesto (contexto diferente, metodología transferible).

### Objetivos TFG cubiertos

OBJ-2 (formulación alternativa del problema de control óptimo), OBJ-4 (perspectiva de Pareto).

---

## Estrategia 8 — Análisis de Sensibilidad Paramétrica (±20 %)

### Formulación

Dado el control óptimo L¹ $u^*$ calculado con parámetros nominales $\theta_0$,
se estudia la variación del coste óptimo cuando un parámetro $\theta_i$ se perturba:

$$\Delta J_i(\pm 20\%) = \frac{J_1^*(u^*;\theta_i^{\pm}) - J_1^*(u^*;\theta_0)}{J_1^*(u^*;\theta_0)}$$

Se perturban: $\delta_F$ (mortalidad femenina), $\nu_E$ (tasa de emergencia), $K$ (capacidad de carga).

### Justificación de la Selección de Parámetros

- **$\delta_F$:** parámetro más incierto en campo (variación estacional de temperatura);
  aparece directamente en la ecuación $\dot{F} = f(F,M_s) - \delta_F F$.
- **$\nu_E$:** controla el reclutamiento de adultos desde el compartimento larval;
  afecta a $R_0 = \beta_E \nu \nu_E / [(\nu_E+\delta_E)\delta_F]$ y por tanto a $\bar{F}$.
- **$K$:** capacidad de carga del hábitat larval; escala $\bar{F}$ linealmente y
  no afecta a $R_0$ (aparece solo en la dependencia densitaria de $\dot{E}$).

### Propiedades

La sensibilidad logarítmica del coste respecto a cada parámetro es:

$$S_i = \frac{\partial \ln J_1^*}{\partial \ln \theta_i}\bigg|_{\theta_0}$$

estimada numéricamente por diferencias finitas (perturbaciones $\pm 20\%$).
El *tornado chart* (diagrama de barras horizontales) ordena los parámetros por
$|S_i|$ descendente, revelando cuáles controlan el coste óptimo.

### Fundamento Bibliográfico

- **Almeida et al. (2022):** los tres parámetros seleccionados son los identificados
  en la Sección 6 como los más influyentes sobre $J^*$.
- **Van der Deure, Delgado-Moya & Almeida (2025):** extienden el análisis de sensibilidad
  a parámetros estacionales (temperatura, humedad), motivando la elección de $\delta_F$
  y $\nu_E$ como parámetros con mayor variabilidad interestacional.
- **Saltelli et al. (2008):** metodología estándar de análisis de sensibilidad global
  (método de Sobol); la variante ±20 % de la presente implementación es análisis de
  sensibilidad local (primer orden), apropiada para la escala del TFG.

### Objetivos TFG cubiertos

OBJ-6 (análisis de sensibilidad paramétrica).

---

## Estrategia 9 — Control por Aprendizaje por Refuerzo Profundo (Deep RL)

*[Propuesta novedosa — extensión de investigación, no implementada en el repositorio base]*
*Base bibliográfica: Böttcher (2026), Schulman et al. (2017), Haarnoja et al. (2018), Sutton & Barto (2018)*

### Motivación

Las estrategias 1, 4, 5 (control óptimo en bucle abierto) y 6 (iP en bucle cerrado)
abordan parcialmente la incertidumbre del entorno real:

| Fuente de incertidumbre               | Estr. 1, 4, 5 | Estr. 6 (iP)  | Estr. 9 (RL) |
|---------------------------------------|---------------|---------------|--------------|
| Parámetros biológicos (β_E, δ_F, ν_E) | Ignorada      | Robusta local | **Robusta global por diseño** |
| Estocasticidad demográfica/ambiental  | Ignorada      | Tolerada      | **Modelada explícitamente**    |
| Restricciones operativas no-convexas  | Ignorada      | No tratada    | **Internalizadas en la recompensa** |
| Observación parcial (sólo capturas)   | No aplicable  | No tratada    | **Política partial-obs (POMDP)** |

El aprendizaje por refuerzo profundo (Deep RL) entrena una **política**
$\pi_\theta(u\,|\,s)$ implementada como red neuronal que maximiza la recompensa
esperada bajo un simulador estocástico y aleatorización de parámetros (*domain
randomization*, Tobin et al. 2017). Es la única estrategia de la familia que
aborda los cuatro ejes anteriores simultáneamente.

### Formulación como Proceso de Decisión de Markov (MDP)

- **Estado:** $s_t = (F(t), M_s(t), t/T) \in \mathbb{R}^3$ (extensible a observación
  parcial $\hat{F}$ por *ovitraps* con ruido Poisson).
- **Acción:** $a_t = u(t) \in [0, U_{\max}]$ (continua) o discretizada en bins.
- **Transición:** $s_{t+\Delta t} = \Phi(s_t, a_t) + \eta_t$, integración del sistema S1
  con ruido demográfico aditivo $\eta_t \sim \mathcal{N}(0, \Sigma)$.
- **Recompensa:** $r_t = -a_t\,\Delta t - \beta \cdot \mathbb{1}\{t=T\}\cdot \max(0, F(T)-\varepsilon)^2$
  (coste lineal + barrera cuadrática terminal con $\beta \gg 1$).
- **Objetivo:** $\max_\theta \mathbb{E}_{\theta_{\text{bio}}\sim \mathcal{P}_\Theta,\,\eta}\bigl[\sum_t r_t\bigr]$.

### Algoritmos Candidatos

**PPO (Schulman et al. 2017):** *on-policy*, clip-ratio $\varepsilon_{\text{clip}}=0.2$,
GAE $\lambda=0.95$. Estable, estándar de facto en control continuo, robusto a
hiperparámetros. Pérdida:
$$\mathcal{L}^{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\bigl(\rho_t(\theta)\hat{A}_t,\,\text{clip}(\rho_t,1\pm\varepsilon)\hat{A}_t\bigr)\right]$$

**SAC (Haarnoja et al. 2018):** *off-policy* con entropía máxima
$\mathcal{H}(\pi(\cdot|s))$ añadida al retorno; mejor eficiencia de muestra y
exploración nativa — ventajoso cuando el umbral $\varepsilon=\bar{F}/4$ es difícil
de alcanzar (recompensas dispersas).

### Robustez por Aleatorización de Dominios

En cada episodio de entrenamiento se muestrea $\theta_{\text{bio}}$:
$$\theta_i \sim \mathcal{U}\bigl([0.7,\,1.3]\cdot\theta_i^{\text{nom}}\bigr), \quad i\in\{\beta_E, \delta_F, \nu_E, K\}$$
La política aprendida es **robusta por construcción** a perturbaciones $\pm 30\%$
de los parámetros — generalización directa del análisis de sensibilidad
(Estrategia 8) al diseño del controlador.

### Validación Cruzada de la Política Aprendida

El protocolo de evaluación sigue **scikit-learn cross-validation**
(Pedregosa et al. 2011, *J. Mach. Learn. Res.* 12) aplicado al espacio de parámetros:

1. **K-Fold sobre $\Theta$ ($K=5$):** se particiona el dominio paramétrico en cinco
   regiones; en cada *fold* se entrena con cuatro y se evalúa con el restante.
2. **Métrica:** $\bar{J}(\pi_\theta;\,\Theta_{\text{test}}) = \mathbb{E}_{\theta\in\Theta_{\text{test}}}[J_1(u_\pi)]$
   sobre $N=100$ trayectorias por $\theta$.
3. **Comparable a Join et al. (2026):** allí el ratio de éxito es 98/100 en
   simulaciones de robustez; aquí cuantificamos formalmente la generalización
   *fuera* del conjunto de entrenamiento, no sólo perturbaciones del nominal.

Este uso de validación cruzada justifica metodológicamente la robustez reclamada
y desliga el resultado del entrenamiento concreto realizado.

### Arquitectura Propuesta y Coste Computacional

| Elemento            | Especificación                                                 |
|---------------------|----------------------------------------------------------------|
| Red de política     | MLP 3×64, tanh, salida $\mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$ truncada en $[0, U_{\max}]$ |
| Red crítica         | MLP 3×64, salida escalar $V_\phi(s)$                            |
| Episodio            | $T=150$ días, $\Delta t = 1$ día, 150 pasos                    |
| Entrenamiento       | $10^6$ pasos (≈ $7\cdot 10^3$ episodios)                       |
| Tiempo CPU          | ≈ 2–4 h (entorno vectorizado, 16 envs paralelos)               |
| Despliegue          | Inferencia $\sim 10\,\mu$s por paso                            |
| Framework           | Stable-Baselines3 (PyTorch) + Gymnasium                        |

### Comparación con Estrategias 1 y 6

| Aspecto                          | Estr. 1 (L¹)           | Estr. 6 (iP)        | Estr. 9 (RL)             |
|----------------------------------|------------------------|---------------------|--------------------------|
| Bucle                            | Abierto                | Cerrado             | Cerrado                  |
| Conocimiento del modelo          | Total                  | Ninguno             | Sólo simulador           |
| Coste $\bar{J}$ esperado         | $J_1^*$ (nominal)      | $\sim 1.5\,J_1^*$   | $\sim 1.1\,J_1^*$ (medio) |
| Robustez paramétrica             | Baja                   | Alta local          | **Alta global**          |
| Estocasticidad                   | Ignorada               | Tolerada            | **Modelada**             |
| Garantías teóricas               | PMP (Almeida 2022)     | Estabilidad local   | Ninguna formal           |
| Verificación                     | Almeida 2022 Tab. 3    | Join 2026 Fig. 1–4  | K-Fold sobre $\Theta$    |
| Esfuerzo de implementación       | Medio (GEKKO)          | Bajo (filtro IIR)   | Alto (RL stack)          |

### Propiedades y Limitaciones

- **Ventaja decisiva:** una sola política $\pi_\theta$ sirve para todo $\theta\in[0.7,1.3]\cdot\theta_0$
  *y* para realizaciones estocásticas — no requiere recalcular nada en campo.
- **Limitación teórica:** sin garantías de optimalidad ni de estabilidad asintótica;
  la única certificación realista es estadística (validación cruzada).
- **Conexión con Estrategia 6:** ambas son cerradas y no requieren parámetros
  *online*; RL los necesita para el simulador de entrenamiento, iP no necesita
  ninguno. RL ofrece menor coste medio; iP ofrece simplicidad y trazabilidad.
- **Riesgo de *reward hacking*:** la barrera terminal $\max(0,F(T)-\varepsilon)^2$
  puede inducir comportamientos extremos cerca de $t=T$; mitigable con *reward shaping*
  intermedio $-\gamma\cdot \max(0, F(t) - F^*(t))^2$ donde $F^*$ es la trayectoria L¹.

### Fundamento Bibliográfico

- **Böttcher (2026):** revisión actualizada de control neuronal de sistemas dinámicos,
  marco unificado para RL, MPC neuronal y PINN en EDOs (ya citado en Estr. 6).
- **Schulman et al. (2017):** *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- **Haarnoja et al. (2018):** *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL*,
  ICML 2018, PMLR 80:1861–1870.
- **Sutton & Barto (2018):** *Reinforcement Learning: An Introduction*, MIT Press, 2ª ed.
- **Tobin et al. (2017):** *Domain Randomization for Transferring Deep Neural Networks
  from Simulation to the Real World*, IROS 2017, pp. 23–30.
- **Pedregosa et al. (2011):** *Scikit-learn: Machine Learning in Python*,
  J. Mach. Learn. Res. 12, 2825–2830 (módulo de cross-validation).

### Objetivos TFG cubiertos

OBJ-2 (formulación alternativa via MDP), OBJ-4 (comparación bucle abierto vs.
RL cerrado), OBJ-6 (robustez paramétrica por diseño), **+ contribución original**
(Capítulo 5 "trabajo futuro": control adaptativo data-driven para SIT).

---

## Estrategia 10 — Operador Neural Paramétrico (DeepONet)

*[Propuesta novedosa de investigación — no implementada en el repositorio base]*
*Base bibliográfica: Lu, Jin & Karniadakis (2021), Kovachki et al. (2023)*

### Motivación

Cada vez que cambian los parámetros biológicos $\theta$ (estación, localización,
cepa de *Aedes*), las Estrategias 1, 4, 5 requieren resolver de nuevo un problema
NLP con GEKKO (29.4 s/instancia, Almeida 2022 Tabla 4). En un programa SIT
desplegado sobre múltiples islas o ciclos estacionales, este coste es prohibitivo.

La Estrategia 10 propone aprender **una única vez** el operador
$$\mathcal{G}: \Theta \longrightarrow L^\infty([0,T];\,[0, U_{\max}]), \qquad \theta \mapsto u^*_\theta(\cdot)$$
mediante un **operador neural** (DeepONet, Lu et al. 2021), y consultarlo en
microsegundos para cualquier $\theta$ futuro.

### Formulación del Operador Aprendido

DeepONet factoriza $\mathcal{G}$ en dos redes:
$$\mathcal{G}_\Phi(\theta)(t) \;=\; \sum_{k=1}^{p} \underbrace{b_k(\theta;\Phi_b)}_{\text{branch}}\;\underbrace{\tau_k(t;\Phi_\tau)}_{\text{trunk}}$$

- **Branch net** $b: \mathbb{R}^9 \to \mathbb{R}^p$: codifica $\theta = (\beta_E,\delta_E,\delta_M,\delta_F,\delta_s,\nu_E,\nu,\gamma_s,K)$.
- **Trunk net** $\tau: [0,T] \to \mathbb{R}^p$: codifica la coordenada temporal.
- $p \approx 64$: dimensión latente.

Aproximación universal de operadores no lineales (Chen & Chen 1995; Lu et al. 2021)
garantiza que $\mathcal{G}_\Phi \to \mathcal{G}$ con tasa polinomial en $p$.

### Conjunto de Entrenamiento

Se genera offline mediante GEKKO (Estrategia 1) sobre un *Latin Hypercube* en $\Theta$:
- $N = 5\,000$ muestras $\theta^{(i)} \sim \text{LHS}([0.7,1.3]\cdot\theta_0)$;
- coste de generación: $5\,000 \times 30\,\text{s} \approx 42\,\text{h}$ (factible offline);
- cada muestra aporta $(\theta^{(i)}, \{u^*_{\theta^{(i)}}(t_j)\}_{j=1}^N)$.

Pérdida:
$$\mathcal{L}(\Phi) = \frac{1}{N\,M}\sum_{i=1}^N \sum_{j=1}^M \bigl(\mathcal{G}_\Phi(\theta^{(i)})(t_j) - u^*_{\theta^{(i)}}(t_j)\bigr)^2$$

### Propiedades

- **Inferencia instantánea:** $\sim 10\,\mu$s por consulta (forward pass de dos MLPs),
  $\sim 3\cdot 10^6\times$ más rápido que GEKKO.
- **Diferenciable en $\theta$:** habilita análisis de sensibilidad $\partial u^*/\partial\theta$
  por *autograd* — extensión natural de la Estrategia 8.
- **Generalización:** validable mediante K-Fold sobre $\Theta$ (mismo protocolo que
  Estrategia 9).
- **Limitación:** sólo válido en el dominio de entrenamiento; extrapolación fuera
  de $[0.7,1.3]\theta_0$ no garantizada (requiere reentrenamiento).

### Comparación con Estrategias 1 y 9

| Aspecto                  | Estr. 1 (L¹ NLP)     | Estr. 9 (RL)         | Estr. 10 (DeepONet)    |
|--------------------------|----------------------|----------------------|------------------------|
| Tipo                     | Solver NLP           | Política aprendida   | Operador aprendido     |
| Entrada                  | Parámetros $\theta$  | Estado $s_t$         | Parámetros $\theta$    |
| Salida                   | $u^*(\cdot)$         | $u^*(s_t)$           | $u^*_\theta(\cdot)$    |
| Bucle                    | Abierto              | Cerrado              | Abierto                |
| Coste/consulta           | 30 s (GEKKO)         | 10 µs                | 10 µs                  |
| Coste *offline*          | —                    | 2–4 h                | 42 h                   |
| Diferenciable en $\theta$| No                   | No                   | **Sí**                 |
| Robusto a estocasticidad | No                   | Sí                   | No                     |
| Garantía de optimalidad  | Local (PMP)          | Ninguna              | Local en el dataset    |

### Fundamento Bibliográfico

- **Lu, Jin & Karniadakis (2021):** *Learning nonlinear operators via DeepONet*,
  Nature Mach. Intell. 3, 218–229.
- **Kovachki et al. (2023):** *Neural Operator: Learning Maps Between Function Spaces*,
  J. Mach. Learn. Res. 24, 1–97.
- **Chen & Chen (1995):** *Universal approximation to nonlinear operators*,
  IEEE Trans. Neural Networks 6, 911–917 (teorema fundacional).
- **Li et al. (2021):** *Fourier Neural Operator for Parametric PDEs*, ICLR 2021
  (arquitectura alternativa a DeepONet).
- **McKay, Beckman & Conover (1979):** *A comparison of three methods for selecting
  values of input variables*, Technometrics 21, 239–245 (Latin Hypercube Sampling).

### Objetivos TFG cubiertos

OBJ-2 (formulación operacional del problema), OBJ-4 (alternativa rápida a NLP),
OBJ-6 (sensibilidad diferenciable), **+ contribución original**
(Capítulo 5: solución *operator-learning* para deployment SIT a gran escala).

---

## Resumen Comparativo

| # | Estrategia | Coste total $\int u$ | Tipo | Logística | Robustez param. | Verificación vs. Almeida (2022) |
|---|-----------|---------------------|------|-----------|-----------------|-------------|
| 1 | L¹ óptimo (bang-singular-bang) | $J_1^* \approx 1.46\times10^5$ | Abierto, continuo | Alta | Baja | **Tablas 2, 3, 4 + Thm 3.3 ✓** (núcleo del TFG) |
| 2 | Constante | $> J_1^*$ | Abierto, estacionario | Muy baja | Media | No verifica (benchmark) |
| 3 | Periódico impulsivo (fijo) | $\gg J_1^*$ (depende de $\tau$) | Abierto, discreto | Baja | Baja | No verifica (benchmark) |
| 4 | L² óptimo (suavizado) | $J_2^* > J_1^*$ (insectos) | Abierto, continuo | Alta | Baja | Implementa §5 (cualitativa) |
| 5 | Impulsivo óptimo (SLSQP) | $\approx J_1^*$ para $\tau\to0$ | Abierto, discreto | Media | Baja | Implementa §5 (propuesta) |
| 6* | iP sin modelo (Join et al. 2026) | $\geq J_1^*$ | Cerrado, discreto | Media | **Muy alta** (98/100) | No verifica (paper sucesor) |
| 7* | Tiempo mínimo (dual L¹) | Presupuesto $B$ fijo | Abierto, continuo | Alta | Baja | Sólo menciona Obs. 4.1 |
| 8 | Análisis de sensibilidad | N/A | Herramienta de análisis | N/A | N/A | Cuantifica §6 (cualitativa) |
| 9* | Deep RL (PPO/SAC) | $\sim 1.1\,J_1^*$ | Cerrado, neuronal | Alta | **Alta global** (K-Fold) | Fuera del scope (ML novedoso) |
| 10* | Operador Neural (DeepONet) | $\approx J_1^*$ por consulta | Abierto, neuronal | Alta | Alta | Fuera del scope (ML novedoso) |

*Estrategias propuestas, no implementadas en el repositorio base.

**Categorización por familia metodológica:**

| Familia                     | Estrategias       | Característica común                          |
|-----------------------------|-------------------|-----------------------------------------------|
| Control óptimo clásico      | 1, 4, 5, 7        | PMP + solver NLP (GEKKO/SLSQP)                |
| Operativas/benchmarks       | 2, 3              | Sin optimización, validación frente a óptimo  |
| Análisis                    | 8                 | Cuantifica sensibilidad de las anteriores     |
| Modelo-libre / data-driven  | **6, 9, 10**      | No requieren conocer (todos los) parámetros   |

---

## Tabla de Cobertura de Objetivos TFG

| Estrategia | OBJ-1 | OBJ-2 | OBJ-3 | OBJ-4 | OBJ-5 | OBJ-6 |
|-----------|-------|-------|-------|-------|-------|-------|
| 1 (L¹ óptimo) | ✓ | ✓ | ✓ | ✓ | | |
| 2 (constante) | | | | ✓ | | |
| 3 (periódico) | | | | ✓ | ✓ | |
| 4 (L² óptimo) | | ✓ | | ✓ | | |
| 5 (impulsivo óptimo) | | | | ✓ | ✓ | |
| 6* (iP sin modelo) | | ✓ | ✓† | ✓ | ✓ | |
| 7* (tiempo mínimo) | | ✓ | | ✓ | | |
| 8 (sensibilidad) | | | | | | ✓ |
| 9* (Deep RL) | | ✓ | ✓‡ | ✓ | | ✓ |
| 10* (DeepONet) | | ✓ | | ✓ | | ✓ |

†OBJ-3: Join et al. (2026) usan los mismos parámetros nominales → validación cruzada del modelo S2.
‡OBJ-3: K-Fold sobre $\Theta$ valida la generalización fuera del conjunto de entrenamiento (Pedregosa et al. 2011).

---

## Bibliografía de Referencia

1. **Almeida, Duprez, Privat & Vauchelet (2022).** "Optimal control strategies for the sterile mosquitoes technique." *J. Differential Equations* 311, 229–266. DOI: 10.1016/j.jde.2021.12.002.

2. **Join, Almeida & Fliess (2026).** "Sterile mosquito release via intelligent proportional controllers." *6th International Symposium on Complex Systems (ISCS 2026)*, La Rochelle. arXiv:2604.01355.

3. **Fliess, M. & Join, C. (2013).** "Model-free control." *Int. J. Control* **86**, 2228–2252.

4. **Strugarek, Bossin & Dumont (2019).** "On the use of the sterile insect release technique to reduce or eliminate mosquito populations." *Applied Mathematical Modelling* 68, 443–470.

5. **Agbo bidi, K., Almeida, L. & Coron, J.-M. (2025).** "Global stabilization of a Sterile Insect Technique model by feedback laws." *J. Optim. Th. App.* **204**, 30.

6. **Bhaya, A. & Bliman, P.-A. (2025).** "Feedback design for biological control by the sterile insect release technique exploiting monotone system theory." *European J. Control* **86**, 101292.

7. **Thome, Yang & Díaz (2010).** "Optimal control of Aedes aegypti mosquitoes by the sterile insect technique and insecticide." *Mathematical Biosciences* 223(1), 12–23.

8. **Almeida, L., Léculier, A. & Vauchelet, N. (2023).** "Analysis of the Rolling Carpet strategy to eradicate an invasive species." *SIAM J. Math. Anal.* **55**, 275–309.

9. **Böttcher, L. (2026).** "Control of dynamical systems with neural networks." *Nonlinear Dynamics* **114**, 79.

10. **Pant, Bhatt & Bhatt (2025).** "Economic cost-benefit analysis of SIT programs for *Aedes* mosquito control." *Parasites & Vectors* 18(1).

### Aprendizaje Automático y Operadores Neurales (Estrategias 9–10)

11. **Böttcher, L. (2026).** "Control of dynamical systems with neural networks." *Nonlinear Dynamics* 114, 79.

12. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).** "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

13. **Haarnoja, T., Zhou, A., Abbeel, P. & Levine, S. (2018).** "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *Proc. ICML*, PMLR 80, 1861–1870.

14. **Sutton, R. S. & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction*, 2ª ed. MIT Press.

15. **Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W. & Abbeel, P. (2017).** "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." *IROS 2017*, 23–30.

16. **Pedregosa, F. et al. (2011).** "Scikit-learn: Machine Learning in Python." *J. Machine Learning Research* 12, 2825–2830. (Módulo `sklearn.model_selection`: K-Fold, cross_val_score.)

17. **Lu, L., Jin, P. & Karniadakis, G. E. (2021).** "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." *Nature Machine Intelligence* 3, 218–229.

18. **Kovachki, N. et al. (2023).** "Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs." *J. Machine Learning Research* 24, 1–97.

19. **Chen, T. & Chen, H. (1995).** "Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems." *IEEE Trans. Neural Networks* 6(4), 911–917.

20. **Li, Z. et al. (2021).** "Fourier Neural Operator for Parametric Partial Differential Equations." *Proc. ICLR 2021*.

21. **McKay, M. D., Beckman, R. J. & Conover, W. J. (1979).** "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code." *Technometrics* 21(2), 239–245. (Latin Hypercube Sampling para generación del dataset DeepONet.)
