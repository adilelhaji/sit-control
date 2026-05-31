#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sweep EXHAUSTIVO y desatendido de la Estrategia 9 (Deep RL).
#
# Recorre el factorial completo de los hiperparametros clave de PPO, entrena
# cada combinacion, la evalua (robustez +-30% DR y nominal) y escribe un
# resumen agregado. Pensado para correr ~5-6 h con nohup y revisarse despues.
#
#   cd ~/sit && git pull && poetry install -E rl && poetry run pip install rich tqdm
#   nohup bash scripts/rl_sweep.sh > results/rl/sweep.out 2>&1 &
#   # ... volver en ~6 h ...
#   cat results/rl/sweep/SUMMARY.txt
#
# Rejilla por defecto (200 corridas):
#   terminal_penalty : 2 3 4 6 8
#   lr_schedule      : linear constant
#   timesteps        : 1M 2M
#   ent_coef         : 0.005 0.01
#   seeds            : 0 1 2 3 4
# Fijos: batch_size=256, n_envs=24, DR +-30%, epsilon fijo, net [64,64,64].
#
# Variables de entorno para acotar/ampliar (todas opcionales):
#   MAXJOBS (def 9)   TPS  LRS  TSS  ENTS  SEEDS
#   p.ej.  TSS="1000000" SEEDS="0 1 2"  bash scripts/rl_sweep.sh   (mas corto)
# Reanudable: salta cualquier corrida cuyo modelo/evaluacion ya exista.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=results/rl/sweep
mkdir -p "$ROOT"
CFG=configs/almeida2022.yaml
MAXJOBS=${MAXJOBS:-9}
TPS=${TPS:-"2 3 4 6 8"}
LRS=${LRS:-"linear constant"}
TSS=${TSS:-"1000000 2000000"}
ENTS=${ENTS:-"0.005 0.01"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

gate () { while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done; }

train_one () {
  local tag=$1 tp=$2 ts=$3 lr=$4 ent=$5 seed=$6
  local out=$ROOT/$tag
  if [ -f "$out/ppo_policy.zip" ]; then echo "[skip-train] $tag"; return; fi
  echo "[train] $tag"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 poetry run python scripts/run_rl_training.py \
    --config "$CFG" --algorithm PPO --timesteps "$ts" --n-envs 24 \
    --ent-coef "$ent" --terminal-penalty "$tp" --batch-size 256 --lr-schedule "$lr" \
    --seed "$seed" --output "$out" > "$out.log" 2>&1
}

eval_one () {
  local out=$1
  [ -f "$out/ppo_policy.zip" ] || return
  if [ -f "$out/eval_dr/rl_evaluation.json" ] && [ -f "$out/eval_nom/rl_evaluation.json" ]; then
    echo "[skip-eval] $out"; return; fi
  echo "[eval] $out"
  poetry run python scripts/run_rl_evaluation.py --config "$CFG" \
    --model "$out/ppo_policy.zip" --algorithm PPO --n-episodes 500 --seed 999 \
    --output "$out/eval_dr" > /dev/null 2>&1
  poetry run python scripts/run_rl_evaluation.py --config "$CFG" \
    --model "$out/ppo_policy.zip" --algorithm PPO --no-randomize --n-episodes 1 --seed 7 \
    --output "$out/eval_nom" > /dev/null 2>&1
}

# numero total de corridas
N=0
for tp in $TPS; do for ts in $TSS; do for lr in $LRS; do for ent in $ENTS; do for s in $SEEDS; do
  N=$((N+1)); done; done; done; done; done

echo "############ SWEEP RL EXHAUSTIVO — inicio ############"
echo "MAXJOBS=$MAXJOBS  total_corridas=$N"
echo "TPS=[$TPS] LRS=[$LRS] TSS=[$TSS] ENTS=[$ENTS] SEEDS=[$SEEDS]"
date

echo "==== FASE 1: ENTRENAMIENTO ($N corridas) ===="
for tp in $TPS; do for ts in $TSS; do M=$((ts/1000000)); for lr in $LRS; do for ent in $ENTS; do for s in $SEEDS; do
  tag="tp${tp}_ts${M}M_lr${lr}_ent${ent}_s${s}"
  gate; train_one "$tag" "$tp" "$ts" "$lr" "$ent" "$s" &
done; done; done; done; done
wait
echo "==== entrenamiento completo ===="; date

echo "==== FASE 2: EVALUACION ===="
for d in "$ROOT"/*/; do
  d=${d%/}
  gate; eval_one "$d" &
done
wait
echo "==== evaluacion completa ===="; date

echo "==== FASE 3: RESUMEN ===="
poetry run python scripts/rl_summary.py "$ROOT" | tee "$ROOT/SUMMARY.txt"
echo "############ SWEEP RL — fin. Resumen: $ROOT/SUMMARY.txt ############"
date
