from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates

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
def _safe_read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_model_path(run_dir: Path) -> Optional[Path]:
    for cand in ["best_model.zip", "model.zip", "final_model.zip"]:
        p = run_dir / cand
        if p.exists():
            return p
    zips = sorted(run_dir.glob("*.zip"))
    return zips[0] if zips else None


def _load_env_kwargs(run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    env_kwargs: Dict[str, Any] = _safe_read_json(run_dir / "env_kwargs.json")

    # eval/debug overrides
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
    return _safe_read_json(run_dir / "hparams.json")


def _make_env_for_eval(run_dir: Path, args: argparse.Namespace) -> QTOWEnv:
    env_kwargs = _load_env_kwargs(run_dir, args)
    env = QTOWEnv(**env_kwargs)
    return env


def _load_model_with_fallback(model_path: Path):
    try:
        return RecurrentPPO.load(str(model_path), env=None, device="cpu")
    except Exception:
        return PPO.load(str(model_path), env=None, device="cpu")


def _initial_lstm_states(policy, device: th.device, n_envs: int = 1) -> RNNStates:
    shape = getattr(policy, "lstm_hidden_state_shape", None)
    if shape is None:
        num_layers = int(getattr(policy.lstm_actor, "num_layers", 1))
        hidden_size = int(getattr(policy.lstm_actor, "hidden_size"))
        shape = (num_layers, n_envs, hidden_size)
    else:
        shape = tuple(shape)
        if len(shape) != 3:
            raise ValueError(f"Unexpected lstm_hidden_state_shape={shape}")
        shape = (shape[0], n_envs, shape[2])

    def zeros():
        return th.zeros(shape, dtype=th.float32, device=device)

    return RNNStates(pi=(zeros(), zeros()), vf=(zeros(), zeros()))


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_batch(x: Any) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _parse_target_goal(info: Dict[str, Any]) -> int:
    tg = info.get("target_goal", "")
    if tg == "G1":
        return 2
    if tg == "G2":
        return 3
    return -1


# -----------------------------
# Rollout collection
# -----------------------------
def collect_debug_rollouts(run_dir: Path, args: argparse.Namespace) -> Path:
    model_path = _find_model_path(run_dir)
    if model_path is None:
        raise FileNotFoundError(f"No model zip found in {run_dir}")

    env = _make_env_for_eval(run_dir, args)
    hparams = _load_hparams(run_dir)
    env_kwargs = _load_env_kwargs(run_dir, args)

    model = _load_model_with_fallback(model_path)
    if not isinstance(model, RecurrentPPO):
        raise TypeError(
            "collect_debug_rollouts.py currently supports RecurrentPPO checkpoints only."
        )

    policy = model.policy
    if not hasattr(policy, "forward_debug"):
        raise AttributeError(
            "Loaded policy does not implement forward_debug(...). "
            "Please make sure you are using the latest models.py."
        )

    device = next(policy.parameters()).device
    policy.set_training_mode(False)

    rng = np.random.default_rng(args.eval_seed) if args.eval_seed is not None else np.random.default_rng()

    logs: Dict[str, List[Any]] = {
        "episode_id": [],
        "t": [],
        "global_step": [],
        "order_is_ba": [],
        "phase": [],
        "context": [],
        "target_goal": [],
        "action": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "success_phase0": [],
        "success_phase1": [],
        "success_both": [],
        "hit_g1": [],
        "hit_g2": [],
        "agent_r": [],
        "agent_c": [],
        "obs": [],
        "ctx_token": [],
        "pre_latent_pi": [],
        "pre_latent_vf": [],
        "post_latent_pi": [],
        "post_latent_vf": [],
        "action_logits": [],
        "action_probs": [],
        "values": [],
        "rnn_actor_h": [],
        "rnn_actor_c": [],
        "rnn_critic_h": [],
        "rnn_critic_c": [],
    }

    if args.include_counterfactual:
        logs.update(
            {
                "action_logits_ctx0": [],
                "action_probs_ctx0": [],
                "action_logits_ctx1": [],
                "action_probs_ctx1": [],
            }
        )

    global_step = 0
    for episode_id in range(args.episodes):
        try:
            obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        except TypeError:
            obs, info = env.reset()

        lstm_states = _initial_lstm_states(policy, device=device, n_envs=1)
        episode_starts = th.ones((1,), dtype=th.float32, device=device)
        done = False
        t = 0

        while not done:
            obs_tensor, _ = policy.obs_to_tensor(obs)
            with th.no_grad():
                actions_t, values_t, log_prob_t, new_states, debug = policy.forward_debug(
                    obs_tensor,
                    lstm_states,
                    episode_starts,
                    deterministic=args.deterministic,
                    include_counterfactual=args.include_counterfactual,
                )

            action_arr = _to_numpy(actions_t)
            action_scalar = int(np.asarray(action_arr).reshape(-1)[0])

            next_obs, reward, terminated, truncated, next_info = env.step(action_scalar)
            done = bool(terminated) or bool(truncated)

            logs["episode_id"].append(int(episode_id))
            logs["t"].append(int(t))
            logs["global_step"].append(int(global_step))
            logs["order_is_ba"].append(1 if str(next_info.get("order", info.get("order", "AB"))) == "BA" else 0)
            logs["phase"].append(int(next_info.get("phase", info.get("phase", 0))))
            logs["context"].append(int(next_info.get("context", info.get("context", 0))))
            logs["target_goal"].append(int(_parse_target_goal(next_info if isinstance(next_info, dict) else {})))
            logs["action"].append(action_scalar)
            logs["reward"].append(float(reward))
            logs["terminated"].append(int(bool(terminated)))
            logs["truncated"].append(int(bool(truncated)))
            logs["success_phase0"].append(int(bool(next_info.get("success_phase0", False))))
            logs["success_phase1"].append(int(bool(next_info.get("success_phase1", False))))
            logs["success_both"].append(int(bool(next_info.get("success_both", 0))))
            logs["hit_g1"].append(int(bool(next_info.get("hit_g1", False))))
            logs["hit_g2"].append(int(bool(next_info.get("hit_g2", False))))

            agent_pos = next_info.get("agent_pos", (-1, -1))
            if isinstance(agent_pos, (tuple, list)) and len(agent_pos) == 2:
                logs["agent_r"].append(int(agent_pos[0]))
                logs["agent_c"].append(int(agent_pos[1]))
            else:
                logs["agent_r"].append(-1)
                logs["agent_c"].append(-1)

            logs["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            logs["ctx_token"].append(float(np.asarray(obs, dtype=np.float32)[-1]))
            logs["pre_latent_pi"].append(_squeeze_batch(debug["pre_latent_pi"]).astype(np.float32))
            logs["pre_latent_vf"].append(_squeeze_batch(debug["pre_latent_vf"]).astype(np.float32))
            logs["post_latent_pi"].append(_squeeze_batch(debug["post_latent_pi"]).astype(np.float32))
            logs["post_latent_vf"].append(_squeeze_batch(debug["post_latent_vf"]).astype(np.float32))
            logs["action_logits"].append(_squeeze_batch(debug["action_logits"]).astype(np.float32))
            logs["action_probs"].append(_squeeze_batch(debug["action_probs"]).astype(np.float32))
            logs["values"].append(_squeeze_batch(debug["values"]).astype(np.float32))
            logs["rnn_actor_h"].append(_squeeze_batch(debug["rnn_actor_h"]).astype(np.float32))
            logs["rnn_actor_c"].append(_squeeze_batch(debug["rnn_actor_c"]).astype(np.float32))
            logs["rnn_critic_h"].append(_squeeze_batch(debug["rnn_critic_h"]).astype(np.float32))
            logs["rnn_critic_c"].append(_squeeze_batch(debug["rnn_critic_c"]).astype(np.float32))

            if args.include_counterfactual:
                logs["action_logits_ctx0"].append(_squeeze_batch(debug["action_logits_ctx0"]).astype(np.float32))
                logs["action_probs_ctx0"].append(_squeeze_batch(debug["action_probs_ctx0"]).astype(np.float32))
                logs["action_logits_ctx1"].append(_squeeze_batch(debug["action_logits_ctx1"]).astype(np.float32))
                logs["action_probs_ctx1"].append(_squeeze_batch(debug["action_probs_ctx1"]).astype(np.float32))

            lstm_states = new_states
            episode_starts = th.tensor([1.0 if done else 0.0], dtype=th.float32, device=device)
            obs = next_obs
            info = next_info if isinstance(next_info, dict) else {}
            t += 1
            global_step += 1

            if args.max_steps is not None and global_step >= args.max_steps:
                done = True
                break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    env.close()

    arrays: Dict[str, np.ndarray] = {}
    for key, values in logs.items():
        first = values[0] if len(values) > 0 else None
        if isinstance(first, np.ndarray):
            arrays[key] = np.stack(values, axis=0)
        else:
            arrays[key] = np.asarray(values)

    metadata = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "episodes": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "include_counterfactual": bool(args.include_counterfactual),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "env_kwargs": env_kwargs,
        "hparams": hparams,
    }
    arrays["meta_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object)

    out_npz = Path(args.out_npz) if args.out_npz else (run_dir / "debug_rollouts.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    return out_npz


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect debug rollouts with pre-state and counterfactual action distributions."
    )
    p.add_argument("--runs_dir", type=str, required=True,
                   help="Run directory containing model.zip / best_model.zip / final_model.zip")
    p.add_argument("--out_npz", type=str, default=None,
                   help="Output .npz path (default: <runs_dir>/debug_rollouts.npz)")
    p.add_argument("--episodes", type=int, default=200,
                   help="Number of evaluation episodes to collect")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional cap on total collected timesteps")
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic action selection")
    p.add_argument("--eval_seed", type=int, default=None,
                   help="Seed for episode reset sampling")
    p.add_argument("--no_counterfactual", action="store_true",
                   help="Disable ctx=0/1 counterfactual action distributions")

    # Optional env overrides
    p.add_argument("--eval_layout", type=str, default=None)
    p.add_argument("--eval_episode_len", type=int, default=None)
    p.add_argument("--eval_switch_low", type=int, default=None)
    p.add_argument("--eval_switch_high", type=int, default=None)
    p.add_argument("--eval_fixed_order", type=str, default=None, choices=[None, "AB", "BA"])
    p.add_argument("--eval_fixed_switch", type=int, default=None)
    p.add_argument("--eval_shaping", type=float, default=None)
    p.add_argument("--eval_shaping_gate", type=float, default=None)
    p.add_argument("--eval_wrong_goal_penalty", type=float, default=None)
    p.add_argument("--eval_both_bonus", type=float, default=None)
    p.add_argument("--eval_step_cost", type=float, default=None)

    args = p.parse_args()
    args.include_counterfactual = not args.no_counterfactual
    return args


def main() -> None:
    args = parse_args()
    run_dir = Path(args.runs_dir)
    out_npz = collect_debug_rollouts(run_dir, args)
    print(f"Saved debug rollouts: {out_npz}")


if __name__ == "__main__":
    main()
