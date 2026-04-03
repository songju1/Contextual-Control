# train.py
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Callable, Dict, Any

import numpy as np
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO

from env_qtow import QTOWGridEnv
from models import LabelLstmPolicy, InterventionLstmPolicy, MemoryLstmPolicy


# -----------------------------
# Utils
# -----------------------------
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    # deterministic-ish
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only kwargs accepted by callable `fn` (by name)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)

    params = sig.parameters
    # If callable accepts **kwargs, pass everything.
    for p in params.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return dict(kwargs)

    allowed = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


#def build_run_dir(args: argparse.Namespace) -> str:
#    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
#    if args.model == "M":
#        model_suffix = f"model{args.model}_d{args.d}_mem{args.memory_dim}"
#    elif args.model == "I":
#        model_suffix = f"model{args.model}_d{args.d}_a{args.alpha:g}"
#    else:
#        model_suffix = f"model{args.model}_d{args.d}"
#    suffix = f"{ts}_{model_suffix}_seed{args.seed}_nenv{args.n_envs}"
#    run_dir = os.path.join("runs", suffix)
#    os.makedirs(run_dir, exist_ok=True)
#    return run_dir
#

def build_run_dir(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Model identity
    if args.model == "M":
        model_suffix = f"model{args.model}_d{args.d}_mem{args.memory_dim}"
    elif args.model == "I":
        model_suffix = f"model{args.model}_d{args.d}_a{args.alpha:g}"
    else:
        model_suffix = f"model{args.model}_d{args.d}"

    # Task identity (so eval/collection won't mix runs)
    layout_suffix = f"layout{args.layout}"
    order_suffix = f"order{args.fixed_order}" if args.fixed_order is not None else "orderrand"
    switch_suffix = f"switch{args.fixed_switch}" if args.fixed_switch is not None else "switchrand"

    suffix = (
        f"{ts}_{model_suffix}_{layout_suffix}_{order_suffix}_{switch_suffix}"
        f"_seed{args.seed}_nenv{args.n_envs}"
    )

    run_dir = os.path.join("runs", suffix)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir



def make_env(env_kwargs: Dict[str, Any], base_seed: int, rank: int) -> Callable[[], QTOWGridEnv]:
    """Factory for SubprocVecEnv (each worker creates its own env)."""
    def _init() -> QTOWGridEnv:
        env = QTOWGridEnv(**env_kwargs)
        # per-worker seed
        try:
            env.reset(seed=base_seed + rank)
        except TypeError:
            pass
        return env
    return _init


class SimpleLoggerCallback(BaseCallback):
    """Writes some training scalars to a file."""
    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self._fp = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._fp = open(self.log_path, "w", encoding="utf-8")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self._fp is None:
            return
        try:
            ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
            ep_len_mean = self.logger.name_to_value.get("rollout/ep_len_mean", None)
            fps = self.logger.name_to_value.get("time/fps", None)
            timesteps = self.num_timesteps
            self._fp.write(
                f"t={timesteps} ep_rew_mean={ep_rew_mean} ep_len_mean={ep_len_mean} fps={fps}\n"
            )
            self._fp.flush()
        except Exception:
            pass

    def _on_training_end(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # model
    p.add_argument(
        "--model",
        type=str,
        default="I",
        choices=["L", "I", "M"],
        help=(
            "L: label model (ctx token is ordinary input), "
            "I: intervention model (ctx only acts as operator), "
            "M: memory model (no ctx input, extra hidden memory)"
        ),
    )
    p.add_argument("--d", type=int, default=32, help="base latent dimension")
    p.add_argument("--alpha", type=float, default=0.1, help="intervention strength (model=I only)")
    p.add_argument("--memory_dim", type=int, default=8, help="extra recurrent memory for model=M only")
    p.add_argument("--features_hidden", type=int, default=128, help="hidden size of observation encoder MLP")

    # run control
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=350_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--start_method", type=str, default="forkserver", choices=["forkserver", "spawn", "fork"])

    # RecurrentPPO / optimizer hparams
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # env params
    p.add_argument("--episode_len", type=int, default=None)
    p.add_argument("--switch_low", type=int, default=None)
    p.add_argument("--switch_high", type=int, default=None)
    p.add_argument("--goal_reward", type=float, default=None)

    # reward knobs
    p.add_argument("--shaping", type=float, default=0.03)
    p.add_argument("--shaping_gate", type=float, default=6.0)
    p.add_argument("--wrong_goal_penalty", type=float, default=0.0)
    p.add_argument("--both_bonus", type=float, default=2.0)
    p.add_argument("--step_cost", type=float, default=-0.01)
    p.add_argument("--blocked_penalty", type=float, default=0.0)
    p.add_argument("--blocked_streak", type=int, default=3)
    p.add_argument("--miss0_penalty", type=float, default=0.0)

    p.add_argument("--layout", type=str, default="hard", choices=["hard", "easy"], help="maze layout type")
    p.add_argument("--fixed_order", type=str, default=None, choices=[None, "AB", "BA"])
    p.add_argument("--fixed_switch", type=int, default=None)

    # debug
    p.add_argument("--print_env_kwargs", action="store_true")
    p.add_argument("--print_hparams", action="store_true")
    p.add_argument("--print_policy_kwargs", action="store_true")

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    run_dir = build_run_dir(args)
    sb3_log_path = os.path.join(run_dir, "sb3_stdout.log")

    # ---- build env kwargs ----
    raw_env_kwargs: Dict[str, Any] = dict(
        shaping=args.shaping,
        shaping_gate=args.shaping_gate,
        wrong_goal_penalty=args.wrong_goal_penalty,
        both_success_bonus=args.both_bonus,
        step_penalty=args.step_cost,
        layout_name=args.layout,
        blocked_penalty=args.blocked_penalty,
        blocked_streak=args.blocked_streak,
        miss0_penalty=args.miss0_penalty,
    )
    if args.episode_len is not None:
        raw_env_kwargs["episode_len"] = int(args.episode_len)
    if args.switch_low is not None:
        raw_env_kwargs["switch_low"] = int(args.switch_low)
    if args.switch_high is not None:
        raw_env_kwargs["switch_high"] = int(args.switch_high)
    if args.goal_reward is not None:
        raw_env_kwargs["goal_reward"] = float(args.goal_reward)
    if args.fixed_order is not None:
        raw_env_kwargs["fixed_order"] = args.fixed_order
    if args.fixed_switch is not None:
        raw_env_kwargs["fixed_switch"] = int(args.fixed_switch)
    env_kwargs = _filter_kwargs_for_callable(QTOWGridEnv.__init__, raw_env_kwargs)

    with open(os.path.join(run_dir, "env_kwargs.json"), "w", encoding="utf-8") as f:
        json.dump(env_kwargs, f, ensure_ascii=False, indent=2)

    if args.print_env_kwargs:
        print("ENV_KWARGS:", env_kwargs)

    # ---- policy selection ----
    common_policy_kwargs = dict(
        d=int(args.d),
        features_hidden=int(args.features_hidden),
        net_arch=dict(pi=[], vf=[]),
    )

    if args.model == "L":
        policy = LabelLstmPolicy
        policy_kwargs = dict(common_policy_kwargs)
    elif args.model == "I":
        policy = InterventionLstmPolicy
        policy_kwargs = dict(common_policy_kwargs, alpha=float(args.alpha))
    elif args.model == "M":
        policy = MemoryLstmPolicy
        policy_kwargs = dict(common_policy_kwargs, memory_dim=int(args.memory_dim))
    else:  # pragma: no cover
        raise ValueError(f"Unknown model: {args.model}")

    # ---- save hparams / run metadata ----
    hparams = dict(
        model=args.model,
        d=args.d,
        alpha=args.alpha,
        memory_dim=args.memory_dim,
        features_hidden=args.features_hidden,
        layout=args.layout,
        fixed_order=args.fixed_order,
        fixed_switch=args.fixed_switch,
        episode_len=args.episode_len,
        seed=args.seed,
        steps=args.steps,
        n_envs=args.n_envs,
        device=args.device,
        start_method=args.start_method,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_class=policy.__name__,
        policy_kwargs=policy_kwargs,
    )
    with open(os.path.join(run_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    if args.print_hparams:
        print("HPARAMS:", hparams)
    if args.print_policy_kwargs:
        print("POLICY:", policy.__name__)
        print("POLICY_KWARGS:", policy_kwargs)

    # ---- build vectorized env ----
    env = SubprocVecEnv(
        [make_env(env_kwargs, args.seed, i) for i in range(args.n_envs)],
        start_method=args.start_method,
    )

    model = RecurrentPPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
        learning_rate=float(args.lr),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ent_coef=float(args.ent_coef),
        clip_range=float(args.clip_range),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
    )

    cb = SimpleLoggerCallback(os.path.join(run_dir, "train_log.txt"))

    # Redirect SB3 logs to file
    with open(sb3_log_path, "w", encoding="utf-8") as fp:
        with redirect_stdout(fp), redirect_stderr(fp):
            model.learn(total_timesteps=int(args.steps), callback=cb)

    model.save(os.path.join(run_dir, "model.zip"))
    print(f"Training finished. Logs saved to {sb3_log_path}")


if __name__ == "__main__":
    main()
    
