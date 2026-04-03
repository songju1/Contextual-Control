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
# Helpers
# ============================================================

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default


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
    """
    Fallback parser from path like:
      debug_infoI_full/M16/orderBA_switch30/seed3.npz
    """
    out: Dict[str, object] = {
        "model": None,
        "memory_dim": None,
        "order": None,
        "switch": None,
        "seed": None,
    }

    parts = path.parts
    for p in parts:
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

    if model == "M" and memory_dim in (None, -1):
        memory_dim = info.get("memory_dim")

    return {
        "model": model,
        "memory_dim": None if memory_dim in (None, -1) else int(memory_dim),
        "order": order,
        "switch": None if switch in (None, -1) else int(switch),
        "seed": None if seed in (None, -1) else int(seed),
    }


def _iter_npz_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.npz")))
        elif p.suffix == ".npz":
            files.append(p)
    # remove duplicates while preserving order
    out = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _kl_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise KL(p || q), returns shape (N,)
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def _compute_info_from_counterfactual(
    p0: np.ndarray,
    p1: np.ndarray,
    w0: float,
    w1: float,
) -> Dict[str, float]:
    """
    Estimate:
      E_s [ w0 KL(p0(.|s)||m(.|s)) + w1 KL(p1(.|s)||m(.|s)) ]
    where m = w0 p0 + w1 p1
    """
    m = w0 * p0 + w1 * p1

    kl0m = _kl_rows(p0, m)
    kl1m = _kl_rows(p1, m)
    kl01 = _kl_rows(p0, p1)
    kl10 = _kl_rows(p1, p0)

    info = w0 * kl0m + w1 * kl1m
    js = 0.5 * (kl0m + kl1m)

    # total variation between the two action distributions
    tv = 0.5 * np.sum(np.abs(p0 - p1), axis=1)

    return {
        "info_mean_nats": float(np.mean(info)),
        "info_std_nats": float(np.std(info, ddof=1)) if info.size >= 2 else 0.0,
        "js_mean_nats": float(np.mean(js)),
        "kl01_mean_nats": float(np.mean(kl01)),
        "kl10_mean_nats": float(np.mean(kl10)),
        "tv_mean": float(np.mean(tv)),
    }


def _subset_mask(npz: np.lib.npyio.NpzFile, phase_mode: str) -> np.ndarray:
    n = len(npz["context"])
    mask = np.ones(n, dtype=bool)

    if phase_mode != "all":
        want = int(phase_mode)
        if "phase" not in npz:
            raise KeyError("phase filter requested, but `phase` not found in npz")
        mask &= (npz["phase"].astype(int) == want)

    return mask


def _mean_l1_reconstruction_error(
    context: np.ndarray,
    action_probs: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> float:
    p_sel = np.where(context[:, None] == 0, p0, p1)
    return float(np.mean(np.sum(np.abs(action_probs - p_sel), axis=1)))


def _bits(x_nats: float) -> float:
    return float(x_nats / math.log(2.0))


def _group_key(row: Dict[str, object]) -> Tuple:
    return (
        row["model"],
        row["memory_dim"],
        row["order"],
        row["switch"],
        row["phase_filter"],
        row["prior"],
    )


# ============================================================
# Main analysis
# ============================================================

def analyze_one_file(path: Path, phase_mode: str) -> Optional[Dict[str, object]]:
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] failed to load {path}: {e}")
        return None

    required = ["context", "action_probs_ctx0", "action_probs_ctx1"]
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

    context = npz["context"].astype(int)[mask]
    p0 = np.asarray(npz["action_probs_ctx0"], dtype=np.float64)[mask]
    p1 = np.asarray(npz["action_probs_ctx1"], dtype=np.float64)[mask]

    # normalize for safety
    p0 = p0 / p0.sum(axis=1, keepdims=True)
    p1 = p1 / p1.sum(axis=1, keepdims=True)

    # priors over context
    p_ctx1_emp = float(np.mean(context == 1))
    p_ctx0_emp = 1.0 - p_ctx1_emp

    uni = _compute_info_from_counterfactual(p0, p1, 0.5, 0.5)
    emp = _compute_info_from_counterfactual(p0, p1, p_ctx0_emp, p_ctx1_emp)

    row: Dict[str, object] = {
        "file": str(path),
        "model": run_info["model"],
        "memory_dim": run_info["memory_dim"],
        "order": run_info["order"],
        "switch": run_info["switch"],
        "seed": run_info["seed"],
        "phase_filter": phase_mode,
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
    }

    if "action_probs" in npz:
        action_probs = np.asarray(npz["action_probs"], dtype=np.float64)[mask]
        action_probs = action_probs / action_probs.sum(axis=1, keepdims=True)
        row["recon_l1_mean"] = _mean_l1_reconstruction_error(context, action_probs, p0, p1)

    return row


def summarize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple, List[Dict[str, object]]] = {}
    for row in rows:
        for prior in ("uniform", "empirical"):
            rr = dict(row)
            rr["prior"] = prior
            rr["info_bits"] = row[f"info_{prior}_bits"]
            rr["info_nats"] = row[f"info_{prior}_nats"]
            groups.setdefault(_group_key(rr), []).append(rr)

    out: List[Dict[str, object]] = []
    for key, items in groups.items():
        model, memory_dim, order, switch, phase_filter, prior = key
        x_bits = np.asarray([float(r["info_bits"]) for r in items], dtype=np.float64)
        x_nats = np.asarray([float(r["info_nats"]) for r in items], dtype=np.float64)
        tv = np.asarray([float(r["tv_mean"]) for r in items], dtype=np.float64)
        n_used = np.asarray([float(r["n_used"]) for r in items], dtype=np.float64)

        summary = {
            "model": model,
            "memory_dim": memory_dim,
            "order": order,
            "switch": switch,
            "phase_filter": phase_filter,
            "prior": prior,
            "n_files": len(items),
            "mean_n_used": float(np.mean(n_used)),
            "info_mean_bits": float(np.mean(x_bits)),
            "info_std_bits": float(np.std(x_bits, ddof=1)) if len(x_bits) >= 2 else 0.0,
            "info_se_bits": float(np.std(x_bits, ddof=1) / np.sqrt(len(x_bits))) if len(x_bits) >= 2 else 0.0,
            "info_mean_nats": float(np.mean(x_nats)),
            "info_std_nats": float(np.std(x_nats, ddof=1)) if len(x_nats) >= 2 else 0.0,
            "tv_mean": float(np.mean(tv)),
        }
        out.append(summary)

    def sort_key(r: Dict[str, object]):
        model_rank = {"L": 0, "I": 1, "M": 2}.get(r["model"], 9)
        mem = -1 if r["memory_dim"] is None else int(r["memory_dim"])
        order_rank = {"AB": 0, "BA": 1}.get(r["order"], 9)
        return (model_rank, mem, order_rank, int(r["switch"] or -1), str(r["phase_filter"]), str(r["prior"]))

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
        description="Aggregate counterfactual estimates of I(C;A|S) from debug_rollouts .npz files."
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="One or more .npz files or directories containing them recursively.",
    )
    p.add_argument(
        "--phase",
        choices=["all", "0", "1"],
        default="all",
        help="Restrict analysis to phase 0 / phase 1 / all timesteps.",
    )
    p.add_argument(
        "--out_prefix",
        type=str,
        default="infoI",
        help="Prefix for CSV outputs: <prefix>_per_file.csv and <prefix>_summary.csv",
    )
    args = p.parse_args()

    files = _iter_npz_files(args.inputs)
    if not files:
        raise SystemExit("No .npz files found.")

    rows: List[Dict[str, object]] = []
    for path in files:
        row = analyze_one_file(path, phase_mode=args.phase)
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
            f"phase={r['phase_filter']} prior={r['prior']} "
            f"n={r['n_files']} "
            f"I_mean_bits={r['info_mean_bits']:.6f} "
            f"+/- {r['info_se_bits']:.6f} (SE)"
        )


if __name__ == "__main__":
    main()
    
