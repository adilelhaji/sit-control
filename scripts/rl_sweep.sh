#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# EXHAUSTIVE, unattended sweep of Strategy 9 (Deep RL).
#
# Iterates over the full factorial of the key PPO hyperparameters, trains
# each combination, evaluates it (robustness +-30% DR and nominal) and writes
# an aggregated summary. Designed to run ~5-6 h with nohup and be reviewed
# afterwards.
#
#   cd ~/sit && git pull && poetry install -E rl && poetry run pip install rich tqdm
#   nohup bash scripts/rl_sweep.sh > results/rl/sweep.out 2>&1 &
#   # ... come back in ~6 h ...
#   cat results/rl/sweep/SUMMARY.txt
#
# Default grid (200 runs):
#   terminal_penalty : 2 3 4 6 8
#   lr_schedule      : linear constant
#   timesteps        : 1M 2M
#   ent_coef         : 0.005 0.01
#   seeds            : 0 1 2 3 4
# Fixed: batch_size=256, n_envs=24, DR +-30%, fixed epsilon, net [64,64,64].
#
# Environment variables to narrow/widen the grid (all optional):
#   MAXJOBS (def 9)   TPS  LRS  TSS  ENTS  SEEDS
#   e.g.  TSS="1000000" SEEDS="0 1 2"  bash scripts/rl_sweep.sh   (shorter)
# Resumable: skips any run whose model/evaluation already exists.
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

# total number of runs
N=0
for tp in $TPS; do for ts in $TSS; do for lr in $LRS; do for ent in $ENTS; do for s in $SEEDS; do
  N=$((N+1)); done; done; done; done; done

echo "############ EXHAUSTIVE RL SWEEP — start ############"
echo "MAXJOBS=$MAXJOBS  total_runs=$N"
echo "TPS=[$TPS] LRS=[$LRS] TSS=[$TSS] ENTS=[$ENTS] SEEDS=[$SEEDS]"
date

echo "==== PHASE 1: TRAINING ($N runs) ===="
for tp in $TPS; do for ts in $TSS; do M=$((ts/1000000)); for lr in $LRS; do for ent in $ENTS; do for s in $SEEDS; do
  tag="tp${tp}_ts${M}M_lr${lr}_ent${ent}_s${s}"
  gate; train_one "$tag" "$tp" "$ts" "$lr" "$ent" "$s" &
done; done; done; done; done
wait
echo "==== training complete ===="; date

echo "==== PHASE 2: EVALUATION ===="
for d in "$ROOT"/*/; do
  d=${d%/}
  gate; eval_one "$d" &
done
wait
echo "==== evaluation complete ===="; date

echo "==== PHASE 3: SUMMARY ===="
poetry run python scripts/rl_summary.py "$ROOT" | tee "$ROOT/SUMMARY.txt"
echo "############ RL SWEEP — done. Summary: $ROOT/SUMMARY.txt ############"
date
