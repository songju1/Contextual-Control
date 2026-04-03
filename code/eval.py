# eval.py
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

# Import custom policies so SB3 can deserialize custom policy classes.
from models import LabelLstmPolicy, InterventionLstmPolicy, MemoryLstmPolicy, PlainLstmPolicy  # noqa: F401

# env_qtow may expose either QTOWEnv or QTOWGridEnv depending on version.
try:
    from env_qtow import QTOWEnv  # type: ignore
except ImportError:  # pragma: no cover
    from env_qtow import QTOWGridEnv as QTOWEnv  # type: ignore


# -----------------------------
# Helpers
# -----------------------------
def _detect_model_type(run_dir_name: str) -> str:
    """
    Heuristic only for labeling in CSV when hparams.json is missing.
    Supports L / I / M.
    """
    name = run_dir_name.lower()
    if "modelm" in name or "_modelm_" in name:
        return "M"
    if "modeli" in name or "_modeli_" in name:
        return "I"
    if "modell" in name or "_modell_" in name:
        return "L"
    return ""


def _find_model_path(run_dir: Path) -> Optional[Path]:
    for cand in ["best_model.zip", "model.zip", "final_model.zip"]:
        p = run_dir / cand
        if p.exists():
            return p
    zips = sorted(run_dir.glob("*.zip"))
    return zips[0] if zips else None


def _parse_from_run_name(run_name: str) -> Tuple[str, str, str]:
    d = ""
    seed = ""
    memory_dim = ""

    m = re.search(r"_d(\d+)", run_name)
    if m:
        d = m.group(1)
    m = re.search(r"_seed(\d+)", run_name)
    if m:
        seed = m.group(1)
    m = re.search(r"_mem(\d+)", run_name)
    if m:
        memory_dim = m.group(1)
    return d, seed, memory_dim


def _safe_read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_env_kwargs(run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load env_kwargs.json from the run directory, and optionally override via eval flags.
    If env_kwargs.json is missing/broken, falls back to {} (env defaults).
    """
    env_kwargs: Dict[str, Any] = _safe_read_json(run_dir / "env_kwargs.json")

    # eval-only overrides (None => do not override)
    if args.eval_shaping is not None:
        env_kwargs["shaping"] = float(args.eval_shaping)
    if args.eval_shaping_gate is not None:
        env_kwargs["shaping_gate"] = float(args.eval_shaping_gate)
    if args.eval_wrong_goal_penalty is not None:
        env_kwargs["wrong_goal_penalty"] = float(args.eval_wrong_goal_penalty)
    if args.eval_both_bonus is not None:
        env_kwargs["both_success_bonus"] = float(args.eval_both_bonus)
    if args.eval_step_cost is not None:
        env_kwargs["step_penalty"] = float(args.eval_step_cost)

    # optional additional overrides
    if args.eval_episode_len is not None:
        env_kwargs["episode_len"] = int(args.eval_episode_len)
    if args.eval_switch_low is not None:
        env_kwargs["switch_low"] = int(args.eval_switch_low)
    if args.eval_switch_high is not None:
        env_kwargs["switch_high"] = int(args.eval_switch_high)
    if args.eval_fixed_order is not None:
        env_kwargs["fixed_order"] = args.eval_fixed_order
    if args.eval_fixed_switch is not None:
        env_kwargs["fixed_switch"] = int(args.eval_fixed_switch)
    if args.eval_layout is not None:
        env_kwargs["layout_name"] = args.eval_layout

    return env_kwargs


def _load_hparams(run_dir: Path) -> Dict[str, Any]:
    """Load hparams.json saved by train.py. Missing => {}."""
    return _safe_read_json(run_dir / "hparams.json")


def _make_env_for_eval(run_dir: Path, args: argparse.Namespace) -> QTOWEnv:
    env_kwargs = _load_env_kwargs(run_dir, args)
    env = QTOWEnv(**env_kwargs)
    return env


def _load_model_with_fallback(model_path: Path) -> object:
    """
    Current train.py uses RecurrentPPO for L / I / M.
    We keep PPO fallback for older checkpoints.
    Importing custom policy classes above is usually enough for deserialization.
    """
    try:
        return RecurrentPPO.load(str(model_path), env=None, device="cpu")
    except Exception:
        return PPO.load(str(model_path), env=None, device="cpu")


def _eval_one_episode(env: QTOWEnv, model, deterministic: bool, rng: np.random.Generator) -> Dict[str, Any]:
    try:
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    except TypeError:
        obs, info = env.reset()

    done = False
    ep_return = 0.0

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    last_info = info if isinstance(info, dict) else {}

    phase0_time: Optional[int] = None
    phase1_time: Optional[int] = None
    wrong_hits = 0

    while not done:
        if isinstance(model, RecurrentPPO):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)

        done = bool(terminated) or bool(truncated)
        episode_starts = np.array([done], dtype=bool)

        if isinstance(info, dict):
            last_info = info
            t = int(info.get("t", 0))

            if phase0_time is None and bool(info.get("success_phase0", False)):
                phase0_time = t
            if phase1_time is None and bool(info.get("success_phase1", False)):
                phase1_time = t

            # wrong-goal hits based on hit flags and inferred target
            hit_g1 = bool(info.get("hit_g1", False))
            hit_g2 = bool(info.get("hit_g2", False))
            if hit_g1 or hit_g2:
                order = info.get("order", "AB")
                phase = int(info.get("phase", 0))
                target_is_g1 = (order == "AB" and phase == 0) or (order == "BA" and phase == 1)
                hit_target = (hit_g1 and target_is_g1) or (hit_g2 and (not target_is_g1))
                if not hit_target:
                    wrong_hits += 1

    return {
        "order": last_info.get("order", "AB"),
        "success_phase0": bool(last_info.get("success_phase0", False)),
        "success_phase1": bool(last_info.get("success_phase1", False)),
        "success_both": int(bool(last_info.get("success_both", 0))),
        "return": float(ep_return),
        "phase0_time": phase0_time,
        "phase1_time": phase1_time,
        "wrong_hits": int(wrong_hits),
    }


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    model_path = _find_model_path(run_dir)
    if model_path is None:
        print(f"[SKIP] {run_dir} no model zip found")
        return None

    env_kwargs = _load_env_kwargs(run_dir, args)
    hparams = _load_hparams(run_dir)

    if args.print_env_kwargs:
        print(f"[ENV_KWARGS] {run_dir.name}: {env_kwargs}")
    if args.print_hparams:
        print(f"[HPARAMS] {run_dir.name}: {hparams}")

    env = _make_env_for_eval(run_dir, args)

    try:
        model = _load_model_with_fallback(model_path)
    except Exception as e:
        print(f"[SKIP] {run_dir} load failed: {e}")
        env.close()
        return None

    rng = np.random.default_rng(args.eval_seed) if args.eval_seed is not None else np.random.default_rng()

    returns: List[float] = []
    ab_both = 0
    ba_both = 0
    phase0_cnt = 0
    phase1_cnt = 0

    phase0_times: List[int] = []
    phase1_times: List[int] = []
    wrong_hits_total = 0

    for _ in range(args.episodes):
        ep = _eval_one_episode(env, model, deterministic=args.deterministic, rng=rng)
        returns.append(ep["return"])
        phase0_cnt += int(ep["success_phase0"])
        phase1_cnt += int(ep["success_phase1"])
        wrong_hits_total += int(ep["wrong_hits"])

        if ep["phase0_time"] is not None:
            phase0_times.append(int(ep["phase0_time"]))
        if ep["phase1_time"] is not None:
            phase1_times.append(int(ep["phase1_time"]))

        if ep["success_both"] == 1:
            if ep["order"] == "AB":
                ab_both += 1
            else:
                ba_both += 1

    env.close()

    run_name = run_dir.name

    # prefer hparams values when available
    parsed_d, parsed_seed, parsed_mem = _parse_from_run_name(run_name)
    model_type = (str(hparams.get("model", "")) or _detect_model_type(run_name)).upper()
    d = str(hparams.get("d", "")) if "d" in hparams else parsed_d
    seed_str = str(hparams.get("seed", "")) if "seed" in hparams else parsed_seed
    alpha = hparams.get("alpha", "")
    memory_dim = hparams.get("memory_dim", parsed_mem)
    features_hidden = hparams.get("features_hidden", "")
    policy_class = hparams.get("policy_class", "")

    # core metrics
    avg_return = float(np.mean(returns)) if returns else 0.0
    phase0_rate = phase0_cnt / args.episodes
    phase1_rate = phase1_cnt / args.episodes
    success_ab = ab_both / args.episodes
    success_ba = ba_both / args.episodes

    avg_phase0_time = float(np.mean(phase0_times)) if phase0_times else ""
    avg_phase1_time = float(np.mean(phase1_times)) if phase1_times else ""
    wrong_hits_per_ep = wrong_hits_total / args.episodes

    # env fields
    switch_low = env_kwargs.get("switch_low", "")
    switch_high = env_kwargs.get("switch_high", "")
    fixed_switch = env_kwargs.get("fixed_switch", "")
    fixed_order = env_kwargs.get("fixed_order", "")
    episode_len = env_kwargs.get("episode_len", "")
    shaping = env_kwargs.get("shaping", "")
    shaping_gate = env_kwargs.get("shaping_gate", "")
    both_bonus = env_kwargs.get("both_success_bonus", "")
    wrong_goal_penalty = env_kwargs.get("wrong_goal_penalty", "")
    step_penalty = env_kwargs.get("step_penalty", "")
    layout_name = env_kwargs.get("layout_name", env_kwargs.get("layout", ""))
    blocked_penalty = env_kwargs.get("blocked_penalty", "")
    blocked_streak = env_kwargs.get("blocked_streak", "")
    miss0_penalty = env_kwargs.get("miss0_penalty", "")

    # training hparams fields (if missing => blank)
    steps = hparams.get("steps", "")
    n_envs = hparams.get("n_envs", "")
    lr = hparams.get("lr", "")
    gamma = hparams.get("gamma", "")
    gae_lambda = hparams.get("gae_lambda", "")
    ent_coef = hparams.get("ent_coef", "")
    clip_range = hparams.get("clip_range", "")
    n_steps = hparams.get("n_steps", "")
    batch_size = hparams.get("batch_size", "")
    vf_coef = hparams.get("vf_coef", "")
    max_grad_norm = hparams.get("max_grad_norm", "")

    return {
        # identifiers
        "run": run_name,
        "model": model_type,
        "policy_class": policy_class,
        "d": d,
        "seed": seed_str,

        # model params
        "alpha": alpha,
        "memory_dim": memory_dim,
        "features_hidden": features_hidden,

        # training hparams
        "train_steps": steps,
        "n_envs": n_envs,
        "lr": lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "rollout_n_steps": n_steps,
        "batch_size": batch_size,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,

        # env params
        "layout_name": layout_name,
        "fixed_order": fixed_order,
        "fixed_switch": fixed_switch,
        "switch_low": switch_low,
        "switch_high": switch_high,
        "episode_len": episode_len,
        "shaping": shaping,
        "shaping_gate": shaping_gate,
        "both_success_bonus": both_bonus,
        "wrong_goal_penalty": wrong_goal_penalty,
        "step_penalty": step_penalty,
        "blocked_penalty": blocked_penalty,
        "blocked_streak": blocked_streak,
        "miss0_penalty": miss0_penalty,

        # eval metrics
        "success_ab": success_ab,
        "success_ba": success_ba,
        "phase0_rate": phase0_rate,
        "phase1_rate": phase1_rate,
        "avg_return": avg_return,
        "avg_phase0_time": avg_phase0_time,
        "avg_phase1_time": avg_phase1_time,
        "wrong_hits_per_ep": wrong_hits_per_ep,
    }


def _list_run_dirs(runs_dir: Path) -> List[Path]:
    """
    Accept either:
      - a parent directory containing multiple run directories, or
      - a single run directory (containing model.zip).
    """
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    if (runs_dir / "model.zip").exists() or (runs_dir / "best_model.zip").exists() or (runs_dir / "final_model.zip").exists():
        return [runs_dir]

    return sorted([p for p in runs_dir.iterdir() if p.is_dir()])


def _match_run(run_name: str, args: argparse.Namespace) -> bool:
    if args.only_model:
        mt = _detect_model_type(run_name).upper()
        # Prefer name-based filter only when hparams are unavailable.
        # This keeps backward compatibility with old runs.
        if mt and mt != args.only_model:
            return False
        if not mt:
            name = run_name.lower()
            if f"model{args.only_model.lower()}" not in name:
                return False
    if args.match and args.match not in run_name:
        return False
    if args.regex and re.search(args.regex, run_name) is None:
        return False
    return True


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs", help="Parent runs dir OR a single run dir")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--deterministic", action="store_true")

    # filtering
    ap.add_argument("--only_model", type=str, default="", help="L / I / M (optional)")
    ap.add_argument("--match", type=str, default="", help="substring match on run directory name")
    ap.add_argument("--regex", type=str, default="", help="regex match on run directory name")

    # debug prints
    ap.add_argument("--print_env_kwargs", action="store_true", help="print env_kwargs used for each run")
    ap.add_argument("--print_hparams", action="store_true", help="print hparams used for each run")

    # eval random seed control
    ap.add_argument("--eval_seed", type=int, default=None, help="fix eval randomness; None=random")

    # eval-only env overrides (None = do not override)
    ap.add_argument("--eval_shaping", type=float, default=None)
    ap.add_argument("--eval_shaping_gate", type=float, default=None)
    ap.add_argument("--eval_wrong_goal_penalty", type=float, default=None)
    ap.add_argument("--eval_both_bonus", type=float, default=None)
    ap.add_argument("--eval_step_cost", type=float, default=None)

    # optional additional overrides
    ap.add_argument("--eval_episode_len", type=int, default=None)
    ap.add_argument("--eval_switch_low", type=int, default=None)
    ap.add_argument("--eval_switch_high", type=int, default=None)
    ap.add_argument("--eval_fixed_order", type=str, default=None, choices=[None, "AB", "BA"])
    ap.add_argument("--eval_fixed_switch", type=int, default=None)
    ap.add_argument("--eval_layout", type=str, default=None, choices=[None, "hard", "easy"])

    # output
    ap.add_argument("--out_csv", type=str, default="", help="output csv path. default: <runs_dir>/summary.csv")

    args = ap.parse_args()
    args.only_model = args.only_model.strip().upper()
    args.match = args.match.strip()
    args.regex = args.regex.strip()

    runs_dir = Path(args.runs_dir)
    run_dirs = _list_run_dirs(runs_dir)

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        if not _match_run(run_dir.name, args):
            continue

        r = evaluate_run(run_dir, args)
        if r is None:
            continue
        rows.append(r)

        print(
            f"[OK] {r['run']}  model={r['model']} d={r['d']} mem={r['memory_dim']} "
            f"AB={r['success_ab']:.3f} BA={r['success_ba']:.3f} "
            f"p0={r['phase0_rate']:.3f} p1={r['phase1_rate']:.3f} "
            f"avgR={r['avg_return']:.6f} wrong/ep={float(r['wrong_hits_per_ep']):.3f} "
            f"(gamma={r['gamma']}, ent={r['ent_coef']}, n_steps={r['rollout_n_steps']})"
        )

    out_csv = Path(args.out_csv) if args.out_csv else (runs_dir / "summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        # identifiers
        "run", "model", "policy_class", "d", "seed",

        # model params
        "alpha", "memory_dim", "features_hidden",

        # training hparams
        "train_steps", "n_envs", "lr", "gamma", "gae_lambda", "ent_coef", "clip_range",
        "rollout_n_steps", "batch_size", "vf_coef", "max_grad_norm",

        # env params
        "layout_name", "fixed_order", "fixed_switch", "switch_low", "switch_high", "episode_len",
        "shaping", "shaping_gate", "both_success_bonus", "wrong_goal_penalty", "step_penalty",
        "blocked_penalty", "blocked_streak", "miss0_penalty",

        # eval metrics
        "success_ab", "success_ba", "phase0_rate", "phase1_rate",
        "avg_return", "avg_phase0_time", "avg_phase1_time", "wrong_hits_per_ep",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved summary: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
    
