#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ============================================================
# Constants from env_qtow.py
# ============================================================
EMPTY = 0
WALL = 1
G1 = 2
G2 = 3

CTX_A = 0
CTX_B = 1

# obs9 flattened indices for each action
# [[0,1,2],
#  [3,4,5],
#  [6,7,8]]
ACTION_TO_OBS9_IDX = {
    0: 1,  # up
    1: 7,  # down
    2: 3,  # left
    3: 5,  # right
}


# ============================================================
# Generic utilities
# ============================================================

def _read_meta_json(npz: np.lib.npyio.NpzFile) -> Dict:
    if "meta_json" not in npz:
        return {}
    raw = npz["meta_json"]
    try:
        if getattr(raw, "shape", None) == ():
            s = raw.item()
        else:
            s = raw.reshape(-1)[0]
        return json.loads(str(s))
    except Exception:
        return {}


def _infer_from_path(path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "model": None,
        "memory_dim": None,
        "order": None,
        "switch": None,
        "seed": None,
    }

    for p in path.parts:
        if p in ("L", "I"):
            out["model"] = p

        m = re.fullmatch(r"M(\d+)", p)
        if m:
            out["model"] = "M"
            out["memory_dim"] = int(m.group(1))

        m = re.fullmatch(r"order(AB|BA)_switch(\d+)", p)
        if m:
            out["order"] = m.group(1)
            out["switch"] = int(m.group(2))

        m = re.fullmatch(r"seed(\d+)\.npz", p)
        if m:
            out["seed"] = int(m.group(1))

    return out


def _extract_run_info(path: Path, meta: Dict) -> Dict[str, object]:
    info = _infer_from_path(path)

    env_kwargs = meta.get("env_kwargs", {}) if isinstance(meta, dict) else {}
    hparams = meta.get("hparams", {}) if isinstance(meta, dict) else {}

    model = hparams.get("model", info.get("model"))
    memory_dim = hparams.get("memory_dim", info.get("memory_dim"))
    order = env_kwargs.get("fixed_order", hparams.get("fixed_order", info.get("order")))
    switch = env_kwargs.get("fixed_switch", hparams.get("fixed_switch", info.get("switch")))
    seed = hparams.get("seed", info.get("seed"))

    return {
        "model": model,
        "memory_dim": None if memory_dim in (None, -1, "") else int(memory_dim),
        "order": order,
        "switch": None if switch in (None, -1, "") else int(switch),
        "seed": None if seed in (None, -1, "") else int(seed),
    }


def _iter_npz_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.npz")))
        elif p.suffix == ".npz":
            files.append(p)

    out: List[Path] = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _subset_mask(npz: np.lib.npyio.NpzFile, phase_mode: str) -> np.ndarray:
    n = len(npz["context"])
    mask = np.ones(n, dtype=bool)
    if phase_mode != "all":
        if "phase" not in npz:
            raise KeyError("phase filter requested but `phase` missing from npz")
        mask &= (npz["phase"].astype(int) == int(phase_mode))
    return mask


def _bits(x_nats: float) -> float:
    return float(x_nats / math.log(2.0))


def _kl_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def _compute_cf_info(q0: np.ndarray, q1: np.ndarray, w0: float, w1: float) -> Dict[str, float]:
    m = w0 * q0 + w1 * q1
    kl0m = _kl_rows(q0, m)
    kl1m = _kl_rows(q1, m)
    kl01 = _kl_rows(q0, q1)
    kl10 = _kl_rows(q1, q0)
    info = w0 * kl0m + w1 * kl1m
    js = 0.5 * (kl0m + kl1m)
    tv = 0.5 * np.sum(np.abs(q0 - q1), axis=1)

    return {
        "info_mean_nats": float(np.mean(info)),
        "info_std_nats": float(np.std(info, ddof=1)) if info.size >= 2 else 0.0,
        "js_mean_nats": float(np.mean(js)),
        "kl01_mean_nats": float(np.mean(kl01)),
        "kl10_mean_nats": float(np.mean(kl10)),
        "tv_mean": float(np.mean(tv)),
    }


# ============================================================
# Outcome construction from existing .npz
# ============================================================

def _adjacent_cell_codes(obs: np.ndarray) -> np.ndarray:
    """
    obs: (N, 10) or (N, >=9)
    returns codes for [up, down, left, right], shape (N, 4)
    """
    obs9 = np.asarray(obs, dtype=np.float64)[:, :9]
    idx = np.array([ACTION_TO_OBS9_IDX[a] for a in range(4)], dtype=int)
    return obs9[:, idx].astype(int)


def _binary_dist_from_action_probs(flag_per_action: np.ndarray, action_probs: np.ndarray) -> np.ndarray:
    """
    flag_per_action: (N,4) bool/int indicating whether each action causes event
    action_probs:    (N,4)
    return q: (N,2) for {no, yes}
    """
    p_yes = np.sum(action_probs * flag_per_action, axis=1)
    p_yes = np.clip(p_yes, 0.0, 1.0)
    p_no = 1.0 - p_yes
    return np.stack([p_no, p_yes], axis=1)


def _categorical_dist_from_action_probs(classes_per_action: np.ndarray, action_probs: np.ndarray, n_classes: int) -> np.ndarray:
    """
    classes_per_action: (N,4) int in [0, n_classes-1]
    action_probs: (N,4)
    return q: (N,n_classes)
    """
    n = action_probs.shape[0]
    q = np.zeros((n, n_classes), dtype=np.float64)
    for k in range(n_classes):
        mask = (classes_per_action == k).astype(np.float64)
        q[:, k] = np.sum(action_probs * mask, axis=1)
    q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
    return q


def _make_outcome_distributions(
    obs: np.ndarray,
    p_ctx0: np.ndarray,
    p_ctx1: np.ndarray,
    outcome: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Construct q0(o|s), q1(o|s) from existing action-level counterfactuals.

    outcome choices:
      - action       : original 4-way action
      - target_hit   : Bernoulli {no, yes} for hitting target goal in next step
      - wrong_hit    : Bernoulli {no, yes} for hitting wrong goal in next step
      - goal3        : 3-way {other, target, wrong}
      - g1g2other    : 3-way {other, G1, G2}
    """
    p_ctx0 = np.asarray(p_ctx0, dtype=np.float64)
    p_ctx1 = np.asarray(p_ctx1, dtype=np.float64)
    p_ctx0 = p_ctx0 / np.clip(p_ctx0.sum(axis=1, keepdims=True), 1e-12, None)
    p_ctx1 = p_ctx1 / np.clip(p_ctx1.sum(axis=1, keepdims=True), 1e-12, None)

    if outcome == "action":
        return p_ctx0, p_ctx1, {
            "support_any_goal_frac": np.nan,
            "support_ctx0_target_frac": np.nan,
            "support_ctx1_target_frac": np.nan,
        }

    cell_codes = _adjacent_cell_codes(obs)  # (N,4)
    is_g1 = (cell_codes == G1)
    is_g2 = (cell_codes == G2)
    is_any_goal = is_g1 | is_g2

    # In QTOW, context A(0)->G1 and context B(1)->G2
    target0 = is_g1
    wrong0 = is_g2
    target1 = is_g2
    wrong1 = is_g1

    support = {
        "support_any_goal_frac": float(np.mean(np.any(is_any_goal, axis=1))),
        "support_ctx0_target_frac": float(np.mean(np.any(target0, axis=1))),
        "support_ctx1_target_frac": float(np.mean(np.any(target1, axis=1))),
    }

    if outcome == "target_hit":
        q0 = _binary_dist_from_action_probs(target0.astype(float), p_ctx0)
        q1 = _binary_dist_from_action_probs(target1.astype(float), p_ctx1)
        return q0, q1, support

    if outcome == "wrong_hit":
        q0 = _binary_dist_from_action_probs(wrong0.astype(float), p_ctx0)
        q1 = _binary_dist_from_action_probs(wrong1.astype(float), p_ctx1)
        return q0, q1, support

    if outcome == "goal3":
        # 0=other, 1=target, 2=wrong
        cls0 = np.zeros_like(cell_codes, dtype=int)
        cls1 = np.zeros_like(cell_codes, dtype=int)

        cls0[target0] = 1
        cls0[wrong0] = 2

        cls1[target1] = 1
        cls1[wrong1] = 2

        q0 = _categorical_dist_from_action_probs(cls0, p_ctx0, n_classes=3)
        q1 = _categorical_dist_from_action_probs(cls1, p_ctx1, n_classes=3)
        return q0, q1, support

    if outcome == "g1g2other":
        # 0=other, 1=G1, 2=G2
        cls = np.zeros_like(cell_codes, dtype=int)
        cls[is_g1] = 1
        cls[is_g2] = 2

        q0 = _categorical_dist_from_action_probs(cls, p_ctx0, n_classes=3)
        q1 = _categorical_dist_from_action_probs(cls, p_ctx1, n_classes=3)
        return q0, q1, support

    raise ValueError(f"Unknown outcome={outcome}")


# ============================================================
# Per-file analysis
# ============================================================

def analyze_one_file(path: Path, phase_mode: str, outcome: str) -> Optional[Dict[str, object]]:
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] failed to load {path}: {e}")
        return None

    required = ["context", "obs", "action_probs_ctx0", "action_probs_ctx1"]
    missing = [k for k in required if k not in npz]
    if missing:
        print(f"[WARN] skip {path}: missing keys {missing}")
        return None

    meta = _read_meta_json(npz)
    run_info = _extract_run_info(path, meta)

    mask = _subset_mask(npz, phase_mode)
    n_total = int(len(npz["context"]))
    n_used = int(mask.sum())
    if n_used == 0:
        print(f"[WARN] skip {path}: no samples after filter phase={phase_mode}")
        return None

    context = np.asarray(npz["context"], dtype=int)[mask]
    obs = np.asarray(npz["obs"], dtype=np.float64)[mask]
    p0 = np.asarray(npz["action_probs_ctx0"], dtype=np.float64)[mask]
    p1 = np.asarray(npz["action_probs_ctx1"], dtype=np.float64)[mask]

    q0, q1, support = _make_outcome_distributions(obs, p0, p1, outcome=outcome)

    p_ctx1_emp = float(np.mean(context == 1))
    p_ctx0_emp = 1.0 - p_ctx1_emp

    uni = _compute_cf_info(q0, q1, 0.5, 0.5)
    emp = _compute_cf_info(q0, q1, p_ctx0_emp, p_ctx1_emp)

    row: Dict[str, object] = {
        "file": str(path),
        "model": run_info["model"],
        "memory_dim": run_info["memory_dim"],
        "order": run_info["order"],
        "switch": run_info["switch"],
        "seed": run_info["seed"],
        "phase_filter": phase_mode,
        "outcome": outcome,
        "n_total": n_total,
        "n_used": n_used,
        "p_ctx1_empirical": p_ctx1_emp,

        "info_uniform_nats": uni["info_mean_nats"],
        "info_uniform_bits": _bits(uni["info_mean_nats"]),
        "info_uniform_std_nats": uni["info_std_nats"],
        "js_uniform_nats": uni["js_mean_nats"],
        "js_uniform_bits": _bits(uni["js_mean_nats"]),

        "info_empirical_nats": emp["info_mean_nats"],
        "info_empirical_bits": _bits(emp["info_mean_nats"]),

        "tv_mean": uni["tv_mean"],
        "kl01_mean_nats": uni["kl01_mean_nats"],
        "kl10_mean_nats": uni["kl10_mean_nats"],

        "support_any_goal_frac": support["support_any_goal_frac"],
        "support_ctx0_target_frac": support["support_ctx0_target_frac"],
        "support_ctx1_target_frac": support["support_ctx1_target_frac"],
    }

    return row


def summarize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple, List[Dict[str, object]]] = {}
    for row in rows:
        for prior in ("uniform", "empirical"):
            rr = dict(row)
            rr["prior"] = prior
            rr["info_bits"] = row[f"info_{prior}_bits"]
            rr["info_nats"] = row[f"info_{prior}_nats"]
            key = (
                rr["model"],
                rr["memory_dim"],
                rr["order"],
                rr["switch"],
                rr["phase_filter"],
                rr["outcome"],
                rr["prior"],
            )
            groups.setdefault(key, []).append(rr)

    out: List[Dict[str, object]] = []
    for key, items in groups.items():
        model, memory_dim, order, switch, phase_filter, outcome, prior = key

        x_bits = np.asarray([float(r["info_bits"]) for r in items], dtype=np.float64)
        x_nats = np.asarray([float(r["info_nats"]) for r in items], dtype=np.float64)
        tv = np.asarray([float(r["tv_mean"]) for r in items], dtype=np.float64)
        n_used = np.asarray([float(r["n_used"]) for r in items], dtype=np.float64)
        sup_goal = np.asarray([float(r["support_any_goal_frac"]) for r in items], dtype=np.float64)

        summary = {
            "model": model,
            "memory_dim": memory_dim,
            "order": order,
            "switch": switch,
            "phase_filter": phase_filter,
            "outcome": outcome,
            "prior": prior,
            "n_files": len(items),
            "mean_n_used": float(np.mean(n_used)),
            "info_mean_bits": float(np.mean(x_bits)),
            "info_std_bits": float(np.std(x_bits, ddof=1)) if len(x_bits) >= 2 else 0.0,
            "info_se_bits": float(np.std(x_bits, ddof=1) / np.sqrt(len(x_bits))) if len(x_bits) >= 2 else 0.0,
            "info_mean_nats": float(np.mean(x_nats)),
            "info_std_nats": float(np.std(x_nats, ddof=1)) if len(x_nats) >= 2 else 0.0,
            "tv_mean": float(np.mean(tv)),
            "support_any_goal_frac": float(np.mean(sup_goal)),
        }
        out.append(summary)

    def sort_key(r: Dict[str, object]):
        model_rank = {"L": 0, "I": 1, "M": 2}.get(r["model"], 9)
        mem = -1 if r["memory_dim"] is None else int(r["memory_dim"])
        order_rank = {"AB": 0, "BA": 1}.get(r["order"], 9)
        return (
            model_rank, mem, order_rank,
            int(r["switch"] or -1),
            str(r["phase_filter"]), str(r["outcome"]), str(r["prior"])
        )

    out.sort(key=sort_key)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate counterfactual estimates of I(C;O|S) from debug_rollouts .npz using alternative 1-step outcomes."
    )
    p.add_argument("inputs", nargs="+", help="One or more .npz files or directories containing them recursively.")
    p.add_argument("--phase", choices=["all", "0", "1"], default="all")
    p.add_argument(
        "--outcome",
        choices=["action", "target_hit", "wrong_hit", "goal3", "g1g2other"],
        default="goal3",
        help=(
            "action: original action-level outcome; "
            "target_hit: Bernoulli next-step target hit; "
            "wrong_hit: Bernoulli next-step wrong-goal hit; "
            "goal3: {other,target,wrong}; "
            "g1g2other: {other,G1,G2}"
        ),
    )
    p.add_argument("--out_prefix", type=str, default="infoI_outcome")
    args = p.parse_args()

    files = _iter_npz_files(args.inputs)
    if not files:
        raise SystemExit("No .npz files found.")

    rows: List[Dict[str, object]] = []
    for path in files:
        row = analyze_one_file(path, phase_mode=args.phase, outcome=args.outcome)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No analyzable .npz files found.")

    summary = summarize_rows(rows)

    out_prefix = Path(args.out_prefix)
    per_file_csv = out_prefix.parent / f"{out_prefix.name}_per_file.csv"
    summary_csv = out_prefix.parent / f"{out_prefix.name}_summary.csv"

    write_csv(per_file_csv, rows)
    write_csv(summary_csv, summary)

    print(f"[DONE] per-file : {per_file_csv}")
    print(f"[DONE] summary  : {summary_csv}")
    print("")
    print("Top-level summary:")
    for r in summary:
        mem_txt = "" if r["memory_dim"] in (None, "") else f"(mem={r['memory_dim']})"
        print(
            f"  model={r['model']}{mem_txt} "
            f"order={r['order']} switch={r['switch']} "
            f"phase={r['phase_filter']} outcome={r['outcome']} prior={r['prior']} "
            f"n={r['n_files']} "
            f"I_mean_bits={r['info_mean_bits']:.6f} "
            f"+/- {r['info_se_bits']:.6f} (SE) "
            f"support_any_goal={r['support_any_goal_frac']:.4f}"
        )


if __name__ == "__main__":
    main()
    
