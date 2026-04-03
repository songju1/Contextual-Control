from __future__ import annotations

from typing import Dict, Tuple

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates


# ============================================================
# Feature extractors
# ============================================================

class TokenObsExtractorWithCtx(BaseFeaturesExtractor):
    """
    Use the full observation:
      obs = [9 local cells, ctx_token]
    Output dimension = d
    """

    def __init__(self, observation_space: gym.spaces.Box, d: int = 8, hidden: int = 128):
        super().__init__(observation_space, features_dim=int(d))
        in_dim = int(np.prod(observation_space.shape))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(d)),
            nn.Tanh(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.mlp(obs.float())


class TokenObsExtractorNoCtx(BaseFeaturesExtractor):
    """
    Use only the first 9 local cells.
    This enforces:
      pre-intervention state does NOT directly read ctx.
    Output dimension = d
    """

    def __init__(self, observation_space: gym.spaces.Box, d: int = 8, hidden: int = 128):
        super().__init__(observation_space, features_dim=int(d))
        self.mlp = nn.Sequential(
            nn.Linear(9, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(d)),
            nn.Tanh(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs[..., :9].float()
        return self.mlp(x)


# Backward-compatible alias
TokenObsExtractor = TokenObsExtractorWithCtx


# ============================================================
# Helpers
# ============================================================

def _ctx_is_A(obs: th.Tensor) -> th.Tensor:
    """
    Return boolean mask of shape (batch,) indicating ctx=A.
    Convention:
      obs[..., -1] = 0 -> A
      obs[..., -1] = 1 -> B
    """
    if obs.dim() == 1:
        ctx = obs[-1].view(1)
    else:
        ctx = obs[:, -1]
    return ctx < 0.5


def _clone_with_ctx(obs: th.Tensor, ctx_value: float) -> th.Tensor:
    out = obs.clone()
    if out.dim() == 1:
        out[-1] = float(ctx_value)
    else:
        out[:, -1] = float(ctx_value)
    return out


def _default_recurrent_kwargs(kwargs: dict, hidden_size: int) -> dict:
    """
    Common recurrent defaults:
      - separate actor/critic LSTM states
      - critic recurrent enabled
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("lstm_hidden_size", int(hidden_size))
    kwargs.setdefault("n_lstm_layers", 1)
    kwargs.setdefault("shared_lstm", False)
    kwargs.setdefault("enable_critic_lstm", True)
    return kwargs


def _distribution_probs(dist: Distribution) -> th.Tensor:
    """
    Return action probabilities for categorical-like distributions.
    QTOW uses Discrete(4), so this is the expected case.
    """
    inner = getattr(dist, "distribution", None)
    if inner is None or not hasattr(inner, "probs"):
        raise NotImplementedError(
            "Debug action-prob extraction currently supports distributions with `.distribution.probs` only."
        )
    return inner.probs


def _distribution_logits(dist: Distribution) -> th.Tensor:
    inner = getattr(dist, "distribution", None)
    if inner is not None and hasattr(inner, "logits"):
        return inner.logits
    probs = _distribution_probs(dist)
    return th.log(probs.clamp_min(1e-12))


# ============================================================
# Debug mixin
# ============================================================

class RecurrentPolicyDebugMixin:
    """
    Adds rollout-debug utilities without changing training behavior.

    Main use cases:
      - save pre-intervention latent state S_pre
      - save actual action logits/probs
      - save counterfactual action probs under ctx=0/1

    For L/M, counterfactual probs are computed by cloning obs and replacing the
    ctx token, then recomputing the forward path.

    For I, counterfactual probs are computed from the SAME pre-intervention
    latent state by swapping the intervention context only.
    """

    def _get_pre_intervention_latents(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, RNNStates]:
        raise NotImplementedError

    def get_pre_intervention_latents(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, RNNStates]:
        return self._get_pre_intervention_latents(obs, lstm_states, episode_starts)

    def get_post_intervention_latents(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, RNNStates]:
        return self._get_pre_intervention_latents(obs, lstm_states, episode_starts)

    def _debug_dist_from_latent(self, latent_pi: th.Tensor) -> Tuple[Distribution, th.Tensor, th.Tensor]:
        dist: Distribution = self._get_action_dist_from_latent(latent_pi)
        logits = _distribution_logits(dist)
        probs = _distribution_probs(dist)
        return dist, logits, probs

    def _pack_rnn_states(self, states: RNNStates) -> Dict[str, th.Tensor]:
        return {
            "rnn_actor_h": states.pi[0],
            "rnn_actor_c": states.pi[1],
            "rnn_critic_h": states.vf[0],
            "rnn_critic_c": states.vf[1],
        }

    def counterfactual_action_stats(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        ctx_value: float,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Generic fallback: recompute using obs with replaced ctx token.
        This is exact for L, trivial for M, and overridden by I to keep
        pre-state fixed while changing context only.
        """
        obs_cf = _clone_with_ctx(obs, ctx_value)
        latent_pi_cf, _, _ = self.get_post_intervention_latents(obs_cf, lstm_states, episode_starts)
        _, logits_cf, probs_cf = self._debug_dist_from_latent(latent_pi_cf)
        return logits_cf, probs_cf

    def forward_debug(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        include_counterfactual: bool = True,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates, Dict[str, th.Tensor]]:
        """
        Same action/value/log_prob/new_states contract as `forward`, plus a
        debug dictionary with tensors useful for QTOW2E analysis.
        """
        latent_pi_pre, latent_vf_pre, new_states = self.get_pre_intervention_latents(
            obs, lstm_states, episode_starts
        )
        latent_pi, latent_vf, _ = self.get_post_intervention_latents(obs, lstm_states, episode_starts)

        dist, logits, probs = self._debug_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf).flatten()

        debug: Dict[str, th.Tensor] = {
            "pre_latent_pi": latent_pi_pre,
            "pre_latent_vf": latent_vf_pre,
            "post_latent_pi": latent_pi,
            "post_latent_vf": latent_vf,
            "action_logits": logits,
            "action_probs": probs,
            "values": values,
            "ctx_token": obs[..., -1].float().clone(),
        }
        debug.update(self._pack_rnn_states(new_states))

        if include_counterfactual:
            logits_ctx0, probs_ctx0 = self.counterfactual_action_stats(
                obs, lstm_states, episode_starts, ctx_value=0.0
            )
            logits_ctx1, probs_ctx1 = self.counterfactual_action_stats(
                obs, lstm_states, episode_starts, ctx_value=1.0
            )
            debug.update(
                {
                    "action_logits_ctx0": logits_ctx0,
                    "action_probs_ctx0": probs_ctx0,
                    "action_logits_ctx1": logits_ctx1,
                    "action_probs_ctx1": probs_ctx1,
                }
            )

        return actions, values, log_prob, new_states, debug


# ============================================================
# L model: Label model
# ============================================================

class LabelLstmPolicy(RecurrentPolicyDebugMixin, RecurrentActorCriticPolicy):
    """
    L model (Label model)

    - ctx token is part of the ordinary observation input
    - recurrent policy can directly read the context label
    - corresponds to the "label-assisted" baseline in the paper
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        d: int = 8,
        features_hidden: int = 128,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs.setdefault("features_extractor_class", TokenObsExtractorWithCtx)
        kwargs.setdefault(
            "features_extractor_kwargs",
            dict(d=int(d), hidden=int(features_hidden)),
        )
        kwargs = _default_recurrent_kwargs(kwargs, hidden_size=int(d))
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.base_dim = int(d)
        self.model_kind = "L"

    def _get_pre_intervention_latents(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        latent_pi, new_pi = self._process_sequence(
            features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        latent_vf, new_vf = self._process_sequence(
            features, lstm_states.vf, episode_starts, self.lstm_critic
        )
        return latent_pi, latent_vf, RNNStates(pi=new_pi, vf=new_vf)


# Backward-compatible old name
PlainLstmPolicy = LabelLstmPolicy


# ============================================================
# M model: Memory model
# ============================================================

class MemoryLstmPolicy(RecurrentPolicyDebugMixin, RecurrentActorCriticPolicy):
    """
    M model (Memory model)

    - pre-state does NOT receive ctx token
    - no intervention operators
    - instead, the recurrent hidden state is enlarged:
          hidden_size = d + memory_dim

    Interpretation:
      the agent pays for context handling by extra internal memory.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        d: int = 8,
        memory_dim: int = 8,
        features_hidden: int = 128,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs.setdefault("features_extractor_class", TokenObsExtractorNoCtx)
        kwargs.setdefault(
            "features_extractor_kwargs",
            dict(d=int(d), hidden=int(features_hidden)),
        )

        hidden_size = int(d) + int(memory_dim)
        kwargs = _default_recurrent_kwargs(kwargs, hidden_size=hidden_size)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.base_dim = int(d)
        self.memory_dim = int(memory_dim)
        self.total_hidden_dim = hidden_size
        self.model_kind = "M"

    def _get_pre_intervention_latents(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        latent_pi, new_pi = self._process_sequence(
            features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        latent_vf, new_vf = self._process_sequence(
            features, lstm_states.vf, episode_starts, self.lstm_critic
        )
        return latent_pi, latent_vf, RNNStates(pi=new_pi, vf=new_vf)


# ============================================================
# I model: Intervention model
# ============================================================

class InterventionLstmPolicy(RecurrentPolicyDebugMixin, RecurrentActorCriticPolicy):
    """
    I model (Intervention model)

    Core design:
      1) pre-intervention recurrent state uses NO ctx token
      2) ctx is used only to choose an intervention operator
      3) the intervention is residual:

           z' = z + alpha * D_A(z)   if ctx=A
           z' = z + alpha * D_B(z)   if ctx=B

    This matches the paper's intended separation:
      shared latent state S is context-free before intervention,
      and context acts as an operator, not as a stored label.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        d: int = 8,
        alpha: float = 0.1,
        features_hidden: int = 128,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs.setdefault("features_extractor_class", TokenObsExtractorNoCtx)
        kwargs.setdefault(
            "features_extractor_kwargs",
            dict(d=int(d), hidden=int(features_hidden)),
        )
        kwargs = _default_recurrent_kwargs(kwargs, hidden_size=int(d))
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.d = int(d)
        self.alpha = float(alpha)
        self.model_kind = "I"

        # Residual intervention operators
        self.DA = nn.Linear(self.d, self.d, bias=False)
        self.DB = nn.Linear(self.d, self.d, bias=False)

        # Start near identity overall map
        nn.init.zeros_(self.DA.weight)
        nn.init.zeros_(self.DB.weight)

    # -----------------------------
    # Intervention helpers
    # -----------------------------
    def _apply_residual(self, latent: th.Tensor, maskA: th.Tensor) -> th.Tensor:
        """
        latent: (batch, d)
        maskA : (batch,) bool
        """
        if maskA.dim() == 1:
            maskA = maskA.unsqueeze(-1)

        da = self.DA(latent)
        db = self.DB(latent)
        delta = th.where(maskA, da, db)
        return latent + self.alpha * delta

    def _apply_residual_with_ctx_value(self, latent: th.Tensor, ctx_value: float) -> th.Tensor:
        batch = latent.shape[0]
        use_A = bool(ctx_value < 0.5)
        maskA = th.full((batch,), use_A, dtype=th.bool, device=latent.device)
        return self._apply_residual(latent, maskA)

    def _get_pre_intervention_latents(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, RNNStates]:
        """
        Returns:
          latent_pi_pre, latent_vf_pre, new_states

        Useful for probing whether pre-intervention state leaks context.
        """
        features = self.extract_features(obs)
        latent_pi, new_pi = self._process_sequence(
            features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        latent_vf, new_vf = self._process_sequence(
            features, lstm_states.vf, episode_starts, self.lstm_critic
        )
        return latent_pi, latent_vf, RNNStates(pi=new_pi, vf=new_vf)

    def get_post_intervention_latents(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, RNNStates]:
        latent_pi, latent_vf, new_states = self._get_pre_intervention_latents(
            obs, lstm_states, episode_starts
        )
        maskA = _ctx_is_A(obs)
        latent_pi = self._apply_residual(latent_pi, maskA)
        latent_vf = self._apply_residual(latent_vf, maskA)
        return latent_pi, latent_vf, new_states

    def counterfactual_action_stats(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        ctx_value: float,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Counterfactual for I keeps the SAME pre-intervention state and only swaps
        the intervention context.
        """
        latent_pi_pre, _, _ = self.get_pre_intervention_latents(obs, lstm_states, episode_starts)
        latent_pi_cf = self._apply_residual_with_ctx_value(latent_pi_pre, ctx_value=ctx_value)
        _, logits_cf, probs_cf = self._debug_dist_from_latent(latent_pi_cf)
        return logits_cf, probs_cf

    # -----------------------------
    # SB3 required overrides
    # -----------------------------
    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        latent_pi, latent_vf, new_states = self.get_post_intervention_latents(
            obs, lstm_states, episode_starts
        )

        dist: Distribution = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf).flatten()

        return actions, values, log_prob, new_states

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf, _ = self.get_post_intervention_latents(
            obs, lstm_states, episode_starts
        )

        dist: Distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf).flatten()

        return values, log_prob, entropy

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def noncommutativity_fro(self) -> float:
        """
        Effective operators:
          A = I + alpha * DA
          B = I + alpha * DB

        Returns:
          ||AB - BA||_F
        """
        with th.no_grad():
            device = self.DA.weight.device
            dtype = self.DA.weight.dtype
            I = th.eye(self.d, device=device, dtype=dtype)
            A = I + self.alpha * self.DA.weight
            B = I + self.alpha * self.DB.weight
            comm = A @ B - B @ A
            return float(th.norm(comm, p="fro").cpu().item())


# Optional short aliases for paper experiments
LabelPolicy = LabelLstmPolicy
MemoryPolicy = MemoryLstmPolicy
InterventionPolicy = InterventionLstmPolicy
