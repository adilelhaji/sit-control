# Runbook — Estrategia 9 (Deep RL) en servidor

Entrena y evalúa una política de control en **lazo cerrado** (PPO/SAC) sobre el
entorno `SITEnv` (modelo S1, `T=150`, `U=5000`, `ε=F̄/4`), con **domain
randomization ±30 %** sobre `(beta_E, delta_F, nu_E, K)`, y la compara con la
solución óptima de Almeida (`J ≈ 1.328e5`, lazo abierto) en **coste** y, sobre
todo, en **robustez**.

> Pensado para el servidor AMD EPYC 7453 (28 núcleos). El entorno usa integración
> de Euler (barato): el cuello de botella es el número de pasos, que paraleliza
> bien con muchos `--n-envs` (PPO) y lanzando las semillas a la vez.

## 0. Aviso clave (por qué estos hiperparámetros)

En CPU local, PPO con `ent_coef=0` colapsó a la política trivial **"no liberar"**
(coste 0, éxito 0 %): sin exploración, el coste denso `-u·dt` empuja a `u=0`. Por
eso:
- **PPO** se entrena con `--ent-coef 0.01` (exploración).
- **SAC** autoajusta la entropía (`ent_coef="auto"`), no necesita el flag.
- `--terminal-penalty 3.0` refuerza la presión hacia `F(T) ≤ ε`.

## 1. Preparación

```bash
cd ~/sit-control                     # ruta del repo en el servidor
git pull                             # incluye ent_coef / terminal_penalty
poetry install -E rl                 # gymnasium + stable-baselines3 + torch
poetry run pip install rich tqdm     # barra de progreso de SB3
poetry run python -c "import torch, stable_baselines3, gymnasium as g; \
  print('torch', torch.__version__, '| sb3', stable_baselines3.__version__)"
```

## 2. Entrenamiento (3 semillas por algoritmo, en paralelo)

```bash
# PPO — on-policy, 24 envs, exploración
for S in 0 1 2; do
  poetry run python scripts/run_rl_training.py \
    --config configs/almeida2022.yaml \
    --algorithm PPO --timesteps 1000000 --n-envs 24 \
    --ent-coef 0.01 --terminal-penalty 3.0 \
    --seed $S --output results/rl/ppo_seed$S \
    > results/rl/ppo_seed$S.log 2>&1 &
done; wait

# SAC — off-policy, sample-efficient (ignora --n-envs y --ent-coef)
for S in 0 1 2; do
  poetry run python scripts/run_rl_training.py \
    --config configs/almeida2022.yaml \
    --algorithm SAC --timesteps 300000 \
    --terminal-penalty 3.0 \
    --seed $S --output results/rl/sac_seed$S \
    > results/rl/sac_seed$S.log 2>&1 &
done; wait
```

Cada run guarda `results/rl/<algo>_seed<S>/<algo>_policy.zip`.

## 3. Evaluación

Dos evaluaciones por modelo: **robustez** (±30 % DR, 500 episodios) y **nominal**
(sin DR, coste contra Almeida 1.328e5).

```bash
for S in 0 1 2; do
  for A in ppo sac; do
    M=results/rl/${A}_seed$S/${A}_policy.zip
    ALGO=$(echo $A | tr a-z A-Z)
    # robustez bajo domain randomization
    poetry run python scripts/run_rl_evaluation.py \
      --config configs/almeida2022.yaml --model $M --algorithm $ALGO \
      --n-episodes 500 --seed 999 \
      --output results/rl/${A}_seed$S/eval_dr
    # nominal (sin perturbación)
    poetry run python scripts/run_rl_evaluation.py \
      --config configs/almeida2022.yaml --model $M --algorithm $ALGO \
      --no-randomize --n-episodes 1 --seed 7 \
      --output results/rl/${A}_seed$S/eval_nom
  done
done
```

Cada evaluación escribe `rl_evaluation.json` con: `mean_cost`, `std_cost`,
`success_rate`, `epsilon`, `mean_F_terminal`.

**Métricas titulares a comparar:**
- *Nominal* `mean_cost` vs **1.328e5** (óptimo de Almeida): cuánto se acerca el RL.
- *DR* `success_rate` (frente a la fragilidad del lazo abierto y al 98/100 de
  Join 2026) y `mean_cost` bajo incertidumbre.

## 4. Qué traer de vuelta

- `results/rl/*/eval_*/rl_evaluation.json`  (métricas — imprescindible)
- `results/rl/*/*_policy.zip`  (modelos — para regenerar trayectorias y figuras)
- `results/rl/*.log`  (curvas de aprendizaje, opcional)

Con eso genero las figuras (trayectoria de la política RL vs óptimo; barras de
robustez) y redacto la sección del Capítulo 5.

## 5. Notas

- Si una semilla aún colapsa (éxito ≈0 %), subir `--ent-coef` a 0.02 (PPO) o
  `--terminal-penalty` a 5.0, o ampliar `--timesteps`.
- K-fold sobre el manifold (`kfold_evaluate`) es opcional y requiere reentrenar
  K modelos; la evaluación DR de 500 episodios ya da la métrica de robustez.
- `epsilon` se mantiene **fijo** (nominal) en entrenamiento y evaluación
  (`RLConfig.fixed_epsilon=True`), coherente con el criterio del §4.6.
