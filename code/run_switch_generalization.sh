#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Appendix table collection for unseen switch generalization
#
# Evaluates existing trained runs at unseen switch times:
#   order AB: use AB25-trained runs
#   order BA: use BA30-trained runs
#
# Models:
#   L, I, M16
#
# Output:
#   switch_generalization_eval/
#     raw_csvs/*.csv
#     combined_switch_generalization_raw.csv
#     combined_summary_switch_generalization.csv
# ============================================================

RUNS_ROOT="${1:-runs}"
OUT_ROOT="${2:-switch_generalization_eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval.py}"
LAYOUT="${LAYOUT:-layouteasy}"

EPISODES="${EPISODES:-200}"
EVAL_SEED="${EVAL_SEED:-20260403}"

mkdir -p "${OUT_ROOT}/raw_csvs"

find_latest_run() {
  local pattern="$1"
  find "$RUNS_ROOT" -maxdepth 1 -type d -name "$pattern" | sort | tail -n 1
}

run_eval_one() {
  local run_dir="$1"
  local order="$2"
  local eval_switch="$3"
  local out_csv="$4"

  if [[ -z "$run_dir" ]]; then
    echo "[WARN] missing run for order=${order} eval_switch=${eval_switch} -> ${out_csv}" >&2
    return 0
  fi

  echo "[EVAL] ${run_dir} | eval ${order}${eval_switch} -> ${out_csv}"

  "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --runs_dir "$run_dir" \
    --episodes "$EPISODES" \
    --deterministic \
    --eval_seed "$EVAL_SEED" \
    --eval_fixed_order "$order" \
    --eval_fixed_switch "$eval_switch" \
    --out_csv "$out_csv"
}

# ------------------------------------------------------------
# 1) Run eval.py on every needed run
# ------------------------------------------------------------
for order in AB BA; do
  if [[ "$order" == "AB" ]]; then
    train_switch=25
  else
    train_switch=30
  fi

  for seed in $(seq 0 9); do
    # L
    run_dir="$(find_latest_run "*_modelL_d32_${LAYOUT}_order${order}_switch${train_switch}_seed${seed}_nenv*")"
    for eval_switch in 25 30 35 40; do
      out_csv="${OUT_ROOT}/raw_csvs/L_train${order}${train_switch}_seed${seed}_eval${order}${eval_switch}.csv"
      run_eval_one "$run_dir" "$order" "$eval_switch" "$out_csv"
    done

    # I
    run_dir="$(find_latest_run "*_modelI_d32_a0.1_${LAYOUT}_order${order}_switch${train_switch}_seed${seed}_nenv*")"
    for eval_switch in 25 30 35 40; do
      out_csv="${OUT_ROOT}/raw_csvs/I_train${order}${train_switch}_seed${seed}_eval${order}${eval_switch}.csv"
      run_eval_one "$run_dir" "$order" "$eval_switch" "$out_csv"
    done

    # M16
    run_dir="$(find_latest_run "*_modelM_d32_mem16_${LAYOUT}_order${order}_switch${train_switch}_seed${seed}_nenv*")"
    for eval_switch in 25 30 35 40; do
      out_csv="${OUT_ROOT}/raw_csvs/M16_train${order}${train_switch}_seed${seed}_eval${order}${eval_switch}.csv"
      run_eval_one "$run_dir" "$order" "$eval_switch" "$out_csv"
    done
  done
done

# ------------------------------------------------------------
# 2) Aggregate raw CSVs into Appendix-ready summaries
# ------------------------------------------------------------
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

out_root = Path("switch_generalization_eval")
raw_dir = out_root / "raw_csvs"

# allow custom OUT_ROOT when passed as second shell arg
import os
out_root_env = os.environ.get("OUT_ROOT_OVERRIDE")
if out_root_env:
    out_root = Path(out_root_env)
    raw_dir = out_root / "raw_csvs"

rows = []
for p in sorted(raw_dir.glob("*.csv")):
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

raw_out = out_root / "combined_switch_generalization_raw.csv"
if rows:
    header = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                header.append(k)
    with open(raw_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
else:
    with open(raw_out, "w", encoding="utf-8", newline="") as f:
        f.write("")

groups = defaultdict(list)
for r in rows:
    model = r.get("model", "")
    order = r.get("fixed_order", "")
    try:
        sw = int(float(r.get("fixed_switch", "")))
    except Exception:
        continue
    groups[(model, order, sw)].append(r)

summary_rows = []
for (model, order, sw), items in sorted(groups.items()):
    success_col = "success_ab" if order == "AB" else "success_ba"

    success_vals = []
    phase1_vals = []
    return_vals = []

    for r in items:
        try:
            success_vals.append(float(r.get(success_col, 0.0)))
        except Exception:
            pass
        try:
            phase1_vals.append(float(r.get("phase1_rate", 0.0)))
        except Exception:
            pass
        try:
            return_vals.append(float(r.get("avg_return", 0.0)))
        except Exception:
            pass

    n = len(items)
    success_rate = sum(success_vals) / len(success_vals) if success_vals else ""
    success_count = int(round(sum(success_vals))) if success_vals else ""
    phase1_mean = sum(phase1_vals) / len(phase1_vals) if phase1_vals else ""
    return_mean = sum(return_vals) / len(return_vals) if return_vals else ""

    summary_rows.append({
        "model": model,
        "order": order,
        "switch": sw,
        "n_seeds": n,
        "success_count": success_count,
        "success_rate": success_rate,
        "phase1_rate_mean": phase1_mean,
        "avg_return_mean": return_mean,
    })

summary_out = out_root / "combined_summary_switch_generalization.csv"
header = [
    "model", "order", "switch", "n_seeds",
    "success_count", "success_rate", "phase1_rate_mean", "avg_return_mean"
]
with open(summary_out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(summary_rows)

print(f"[DONE] raw combined CSV: {raw_out}")
print(f"[DONE] appendix summary CSV: {summary_out}")
PY
