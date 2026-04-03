#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${1:-runs}"
OUT_ROOT="${2:-debug_infoI_full}"
LAYOUT="${LAYOUT:-layouteasy}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EPISODES="${EPISODES:-200}"
EVAL_SEED="${EVAL_SEED:-20260319}"

mkdir -p "$OUT_ROOT"

find_latest_run() {
  local pattern="$1"
  find "$RUNS_ROOT" -maxdepth 1 -type d -name "$pattern" | sort | tail -n 1
}

collect_one() {
  local run_dir="$1"
  local out_npz="$2"
  if [[ -z "$run_dir" ]]; then
    echo "[WARN] run not found -> $out_npz" >&2
    return 0
  fi
  echo "[COLLECT] $run_dir -> $out_npz"
  "$PYTHON_BIN" collect_debug_rollouts.py \
    --runs_dir "$run_dir" \
    --episodes "$EPISODES" \
    --deterministic \
    --eval_seed "$EVAL_SEED" \
    --out_npz "$out_npz"
}

for order in AB BA; do
  if [[ "$order" == "AB" ]]; then
    sw=25
  else
    sw=30
  fi

  for seed in $(seq 0 9); do
    # L
    run_dir="$(find_latest_run "*_modelL_d32_${LAYOUT}_order${order}_switch${sw}_seed${seed}_nenv*")"
    out_npz="${OUT_ROOT}/L/order${order}_switch${sw}/seed${seed}.npz"
    mkdir -p "$(dirname "$out_npz")"
    collect_one "$run_dir" "$out_npz"

    # I
    run_dir="$(find_latest_run "*_modelI_d32_a0.1_${LAYOUT}_order${order}_switch${sw}_seed${seed}_nenv*")"
    out_npz="${OUT_ROOT}/I/order${order}_switch${sw}/seed${seed}.npz"
    mkdir -p "$(dirname "$out_npz")"
    collect_one "$run_dir" "$out_npz"

    # M sweep
    for mem in 8 16 32 64; do
      run_dir="$(find_latest_run "*_modelM_d32_mem${mem}_${LAYOUT}_order${order}_switch${sw}_seed${seed}_nenv*")"
      out_npz="${OUT_ROOT}/M${mem}/order${order}_switch${sw}/seed${seed}.npz"
      mkdir -p "$(dirname "$out_npz")"
      collect_one "$run_dir" "$out_npz"
    done
  done
done

echo "[DONE] full debug rollout collection finished."
