#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paper runs: M with memory_dim = 16, 32
# Conditions: AB25 and BA30
# Seeds: 0..9
# Same reward/training setting as current paper runs
# ============================================================

MEM_DIMS=(16 32)

D=32
ALPHA=0.1
LAYOUT="easy"
EP_LEN=70
SHAPING=0.02
SHAPING_GATE=20
BOTH_BONUS=6.0
WRONG_GOAL_PENALTY=0.01
STEP_COST=-0.005
BLOCKED_PENALTY=0.02
BLOCKED_STREAK=3
MISS0_PENALTY=0.0
LR=3e-4
GAMMA=0.995
GAE_LAMBDA=0.95
ENT=0.01
CLIP=0.2
N_STEPS=512
BATCH=256
VF_COEF=0.5
MAX_GRAD_NORM=0.5
FEAT_HID=128
TRAIN_STEPS=300000
N_ENVS=4
START_METHOD="spawn"
DEVICE="cpu"

EVAL_EPISODES=200

mkdir -p paper_summaries

for MEM in "${MEM_DIMS[@]}"; do
  echo "============================================================"
  echo "TRAIN: model=M mem=${MEM} AB25 seeds 0..9"
  echo "============================================================"
  for SEED in 0 1 2 3 4 5 6 7 8 9; do
    python train.py --model M --d "${D}" --alpha "${ALPHA}" \
      --memory_dim "${MEM}" \
      --layout "${LAYOUT}" --episode_len "${EP_LEN}" \
      --fixed_order AB --fixed_switch 25 \
      --shaping "${SHAPING}" --shaping_gate "${SHAPING_GATE}" \
      --both_bonus "${BOTH_BONUS}" \
      --wrong_goal_penalty "${WRONG_GOAL_PENALTY}" \
      --step_cost "${STEP_COST}" \
      --blocked_penalty "${BLOCKED_PENALTY}" \
      --blocked_streak "${BLOCKED_STREAK}" \
      --miss0_penalty "${MISS0_PENALTY}" \
      --lr "${LR}" --gamma "${GAMMA}" --gae_lambda "${GAE_LAMBDA}" \
      --ent_coef "${ENT}" --clip_range "${CLIP}" \
      --n_steps "${N_STEPS}" --batch_size "${BATCH}" \
      --vf_coef "${VF_COEF}" --max_grad_norm "${MAX_GRAD_NORM}" \
      --features_hidden "${FEAT_HID}" \
      --steps "${TRAIN_STEPS}" --n_envs "${N_ENVS}" \
      --start_method "${START_METHOD}" --device "${DEVICE}" \
      --seed "${SEED}"
  done

  echo "============================================================"
  echo "TRAIN: model=M mem=${MEM} BA30 seeds 0..9"
  echo "============================================================"
  for SEED in 0 1 2 3 4 5 6 7 8 9; do
    python train.py --model M --d "${D}" --alpha "${ALPHA}" \
      --memory_dim "${MEM}" \
      --layout "${LAYOUT}" --episode_len "${EP_LEN}" \
      --fixed_order BA --fixed_switch 30 \
      --shaping "${SHAPING}" --shaping_gate "${SHAPING_GATE}" \
      --both_bonus "${BOTH_BONUS}" \
      --wrong_goal_penalty "${WRONG_GOAL_PENALTY}" \
      --step_cost "${STEP_COST}" \
      --blocked_penalty "${BLOCKED_PENALTY}" \
      --blocked_streak "${BLOCKED_STREAK}" \
      --miss0_penalty "${MISS0_PENALTY}" \
      --lr "${LR}" --gamma "${GAMMA}" --gae_lambda "${GAE_LAMBDA}" \
      --ent_coef "${ENT}" --clip_range "${CLIP}" \
      --n_steps "${N_STEPS}" --batch_size "${BATCH}" \
      --vf_coef "${VF_COEF}" --max_grad_norm "${MAX_GRAD_NORM}" \
      --features_hidden "${FEAT_HID}" \
      --steps "${TRAIN_STEPS}" --n_envs "${N_ENVS}" \
      --start_method "${START_METHOD}" --device "${DEVICE}" \
      --seed "${SEED}"
  done

  RUN_GLOB_AB="runs/*_modelM_d32_mem${MEM}_*_orderAB_switch25_*_nenv4*"
  RUN_GLOB_BA="runs/*_modelM_d32_mem${MEM}_*_orderBA_switch30_*_nenv4*"

  echo "============================================================"
  echo "EVAL: model=M mem=${MEM} AB25 latest 10 runs"
  echo "============================================================"
  for RUN in $(ls -dt ${RUN_GLOB_AB} | head -n 10); do
    python eval.py --runs_dir "$RUN" --episodes "${EVAL_EPISODES}" --deterministic \
      --eval_layout "${LAYOUT}" --eval_episode_len "${EP_LEN}" \
      --eval_fixed_order AB --eval_fixed_switch 25 \
      --out_csv "$RUN/summary.csv"
  done

  echo "============================================================"
  echo "EVAL: model=M mem=${MEM} BA30 latest 10 runs"
  echo "============================================================"
  for RUN in $(ls -dt ${RUN_GLOB_BA} | head -n 10); do
    python eval.py --runs_dir "$RUN" --episodes "${EVAL_EPISODES}" --deterministic \
      --eval_layout "${LAYOUT}" --eval_episode_len "${EP_LEN}" \
      --eval_fixed_order BA --eval_fixed_switch 30 \
      --out_csv "$RUN/summary.csv"
  done

  echo "============================================================"
  echo "AGGREGATE: model=M mem=${MEM} AB25"
  echo "============================================================"
  OUT_AB="paper_summaries/combined_summary_AB25_M_mem${MEM}_phase1only001_10seeds.csv"
  FIRST_AB=$(ls -dt ${RUN_GLOB_AB} | head -n 1)
  echo "run,$(head -n 1 "${FIRST_AB}/summary.csv")" > "${OUT_AB}"
  for RUN in $(ls -dt ${RUN_GLOB_AB} | head -n 10); do
    tail -n +2 "$RUN/summary.csv" | sed "s|^|$(basename "$RUN"),|" >> "${OUT_AB}"
  done
  echo "Saved: ${OUT_AB}"

  echo "============================================================"
  echo "AGGREGATE: model=M mem=${MEM} BA30"
  echo "============================================================"
  OUT_BA="paper_summaries/combined_summary_BA30_M_mem${MEM}_phase1only001_10seeds.csv"
  FIRST_BA=$(ls -dt ${RUN_GLOB_BA} | head -n 1)
  echo "run,$(head -n 1 "${FIRST_BA}/summary.csv")" > "${OUT_BA}"
  for RUN in $(ls -dt ${RUN_GLOB_BA} | head -n 10); do
    tail -n +2 "$RUN/summary.csv" | sed "s|^|$(basename "$RUN"),|" >> "${OUT_BA}"
  done
  echo "Saved: ${OUT_BA}"

  echo "============================================================"
  echo "AGGREGATE: model=M mem=${MEM} AB25 + BA30"
  echo "============================================================"
  OUT_ALL="paper_summaries/combined_summary_AB25_BA30_M_mem${MEM}_phase1only001_20runs.csv"
  head -n 1 "${OUT_AB}" > "${OUT_ALL}"
  tail -n +2 "${OUT_AB}" >> "${OUT_ALL}"
  tail -n +2 "${OUT_BA}" >> "${OUT_ALL}"
  echo "Saved: ${OUT_ALL}"
done

echo "============================================================"
echo "DONE"
echo "============================================================"
