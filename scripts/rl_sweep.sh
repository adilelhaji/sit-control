#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sweep desatendido de la Estrategia 9 (Deep RL).
#
# Entrena una rejilla de configuraciones PPO (penalizacion terminal x timesteps
# x schedule de LR x semillas), las evalua (robustez +-30% DR y nominal) y
# escribe un resumen agregado. Pensado para lanzarse con nohup y revisarse en
# unas horas.
#
#   cd ~/sit && git pull && poetry install -E rl && poetry run pip install rich tqdm
#   nohup bash scripts/rl_sweep.sh > results/rl/sweep.out 2>&1 &
#   # ... volver en ~1-2 h ...
#   cat results/rl/sweep/SUMMARY.txt
#
# Variables de entorno opcionales:
#   MAXJOBS  corridas en paralelo (def. 8; ~3 nucleos/corrida en 28 cores)
#   SEEDS    lista de semillas (def. "0 1 2 3 4")
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=results/rl/sweep
mkdir -p "$ROOT"
MAXJOBS=${MAXJOBS:-8}
SEEDS=${SEEDS:-"0 1 2 3 4"}
CFG=configs/almeida2022.yaml

gate () { while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done; }

train_one () {
  local tag=$1 tp=$2 ts=$3 lr=$4 seed=$5
  local out=$ROOT/$tag
  if [ -f "$out/ppo_policy.zip" ]; then echo "[skip-train] $tag"; return; fi
  echo "[train] $tag"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 poetry run python scripts/run_rl_training.py \
    --config "$CFG" --algorithm PPO --timesteps "$ts" --n-envs 24 \
    --ent-coef 0.01 --terminal-penalty "$tp" --batch-size 256 --lr-schedule "$lr" \
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

echo "############ SWEEP RL — inicio ############"
echo "MAXJOBS=$MAXJOBS  SEEDS=$SEEDS"
date

echo "==== FASE 1: ENTRENAMIENTO ===="
# Rejilla principal: tp in {3,4,6} x timesteps in {1M,2M} x LR lineal
for tp in 3 4 6; do
  for ts in 1000000 2000000; do
    M=$((ts/1000000))
    for s in $SEEDS; do
      gate; train_one "tp${tp}_ts${M}M_lrlinear_s${s}" "$tp" "$ts" linear "$s" &
    done
  done
done
# Ablacion del estabilizador: LR constante (tp=4, 1M) para comparar varianza
for s in $SEEDS; do
  gate; train_one "tp4_ts1M_lrconstant_s${s}" 4 1000000 constant "$s" &
done
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
