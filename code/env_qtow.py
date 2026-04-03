from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore


# ============================================================
# Cell / context codes
# ============================================================
EMPTY = 0
WALL = 1
G1 = 2
G2 = 3

CTX_A = 0
CTX_B = 1


# ============================================================
# Layouts (9x9)
# ============================================================
LAYOUTS: Dict[str, List[str]] = {
    "hard": [
        "#########",
        "#S..#...#",
        "#.#.#.#.#",
        "#.#...#.#",
        "#.#.###.#",
        "#...#...#",
        "#.###.#.#",
        "#..1#..2#",
        "#########",
    ],
    "easy": [
        "#########",
        "#S..#...#",
        "#.#.#.#.#",
        "#.#...#.#",
        "#.#.#.#.#",
        "#...#...#",
        "#.###.#.#",
        "#..1#..2#",
        "#########",
    ],
}


class QTOWGridEnv(gym.Env):
    """
    QTOW 9x9 gridworld with:
      - 3x3 local observation (9 cells)
      - appended context token (obs dim = 10)
      - one context switch per episode
      - episode order AB or BA

    Context semantics:
      - AB: phase0 -> G1, phase1 -> G2
      - BA: phase0 -> G2, phase1 -> G1

    Reward components:
      - step_penalty every step
      - goal_reward on first hit of the correct goal in each phase
      - both_success_bonus once if both phases succeed
      - wrong_goal_penalty when stepping on the non-target goal
      - optional shortest-path shaping toward the CURRENT phase target
      - optional blocked_penalty when repeated invalid moves occur
      - optional miss0_penalty at episode end if phase0 was never solved

    Compatibility notes:
      - train.py uses `layout_name`, `both_success_bonus`, `step_penalty`
      - eval.py expects info keys such as:
          success_phase0, success_phase1, success_both,
          order, phase, t, hit_g1, hit_g2, agent_pos
      - aliases QTOWEnv and QtowGridEnv are provided at bottom
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_len: int = 55,
        switch_low: int = 20,
        switch_high: int = 29,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        seed: Optional[int] = None,
        wrong_goal_penalty: float = 0.0,
        boundary_as_wall: bool = True,
        both_success_bonus: float = 1.0,
        shaping: float = 0.0,
        shaping_gate: float = 1e9,
        layout_name: str = "hard",
        layout: Optional[str] = None,
        fixed_order: Optional[str] = None,
        fixed_switch: Optional[int] = None,
        blocked_penalty: float = 0.0,
        blocked_streak: int = 3,
        miss0_penalty: float = 0.0,
    ):
        super().__init__()

        self.episode_len = int(episode_len)
        self.switch_low = int(switch_low)
        self.switch_high = int(switch_high)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)

        self.wrong_goal_penalty = float(wrong_goal_penalty)
        self.boundary_as_wall = bool(boundary_as_wall)
        self.both_success_bonus = float(both_success_bonus)

        self.shaping = float(shaping)
        self.shaping_gate = float(shaping_gate)

        # Accept both `layout_name` and old `layout` keyword.
        if layout is not None:
            layout_name = str(layout)
        self.layout_name = str(layout_name)

        self.fixed_order = None if fixed_order is None else str(fixed_order).upper()
        self.fixed_switch = None if fixed_switch is None else int(fixed_switch)

        self.blocked_penalty = float(blocked_penalty)
        self.blocked_streak = int(blocked_streak)
        self.miss0_penalty = float(miss0_penalty)

        if self.episode_len <= 0:
            raise ValueError(f"episode_len must be positive, got {self.episode_len}")
        if self.blocked_streak < 1:
            raise ValueError(f"blocked_streak must be >= 1, got {self.blocked_streak}")
        if self.fixed_order not in (None, "AB", "BA"):
            raise ValueError(f"fixed_order must be None / 'AB' / 'BA', got {fixed_order}")

        if self.fixed_switch is None:
            if not (0 <= self.switch_low <= self.switch_high < self.episode_len):
                raise ValueError(
                    "Require 0 <= switch_low <= switch_high < episode_len, "
                    f"got switch_low={self.switch_low}, switch_high={self.switch_high}, episode_len={self.episode_len}"
                )
        else:
            if not (0 <= self.fixed_switch < self.episode_len):
                raise ValueError(
                    f"fixed_switch must satisfy 0 <= fixed_switch < episode_len, got {self.fixed_switch}"
                )

        # RNG
        self.rng = np.random.default_rng(seed)

        # Build selected layout + BFS distance maps.
        self._build_layout_and_distance_maps()

        # Action: 0 up, 1 down, 2 left, 3 right
        self.action_space = spaces.Discrete(4)

        # Observation: 9 local cells + 1 context token
        low = np.zeros((10,), dtype=np.float32)
        high = np.array([3] * 9 + [1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Runtime state
        self.t = 0
        self.t_switch = 0
        self.order = 0  # 0: AB, 1: BA
        self.agent_pos = self.start_pos

        self.phase0_hit = False
        self.phase1_hit = False
        self.phase0_rewarded = False
        self.phase1_rewarded = False
        self.both_rewarded = False
        self._blocked_run = 0

    # --------------------------------------------------------
    # Layout + distances
    # --------------------------------------------------------
    def _build_layout_and_distance_maps(self) -> None:
        if self.layout_name not in LAYOUTS:
            raise ValueError(f"Unknown layout_name={self.layout_name}. Choose from {list(LAYOUTS.keys())}")

        layout = LAYOUTS[self.layout_name]
        self.grid = self._parse_layout(layout)
        self.start_pos = self._find_char(layout, "S")
        self.g1_pos = self._find_char(layout, "1")
        self.g2_pos = self._find_char(layout, "2")

        self.dist_to_g1 = self._bfs_dist_map(self.g1_pos)
        self.dist_to_g2 = self._bfs_dist_map(self.g2_pos)

    def _parse_layout(self, layout: List[str]) -> np.ndarray:
        if len(layout) != 9 or any(len(row) != 9 for row in layout):
            raise ValueError("Layout must be 9x9.")

        grid = np.zeros((9, 9), dtype=np.int32)
        for r, row in enumerate(layout):
            for c, ch in enumerate(row):
                if ch == "#":
                    grid[r, c] = WALL
                elif ch in [".", "S"]:
                    grid[r, c] = EMPTY
                elif ch == "1":
                    grid[r, c] = G1
                elif ch == "2":
                    grid[r, c] = G2
                else:
                    raise ValueError(f"Unknown char in layout: {ch}")
        return grid

    def _find_char(self, layout: List[str], ch: str) -> Tuple[int, int]:
        for r, row in enumerate(layout):
            for c, cch in enumerate(row):
                if cch == ch:
                    return (r, c)
        raise ValueError(f"Char {ch} not found in layout")

    def _bfs_dist_map(self, goal_pos: Tuple[int, int]) -> np.ndarray:
        dist = np.full((9, 9), np.inf, dtype=np.float32)
        gr, gc = goal_pos
        if self.grid[gr, gc] == WALL:
            return dist

        q = deque([(gr, gc)])
        dist[gr, gc] = 0.0

        while q:
            r, c = q.popleft()
            d = dist[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and self.grid[nr, nc] != WALL:
                    if np.isinf(dist[nr, nc]):
                        dist[nr, nc] = d + 1.0
                        q.append((nr, nc))
        return dist

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _order_str(self) -> str:
        return "AB" if self.order == 0 else "BA"

    def _current_phase(self) -> int:
        return 0 if self.t < self.t_switch else 1

    def _current_ctx(self) -> int:
        before = self.t < self.t_switch
        if self.order == 0:  # AB
            return CTX_A if before else CTX_B
        else:  # BA
            return CTX_B if before else CTX_A

    def _target_goal(self, phase: int) -> int:
        # AB => phase0:G1 phase1:G2
        # BA => phase0:G2 phase1:G1
        if self.order == 0:
            return G1 if phase == 0 else G2
        return G2 if phase == 0 else G1

    def _target_goal_name(self, phase: int) -> str:
        return "G1" if self._target_goal(phase) == G1 else "G2"

    def _pos_is_goal(self, pos: Tuple[int, int], goal_code: int) -> bool:
        r, c = pos
        return self.grid[r, c] == goal_code

    def _dist_to_target(self, pos: Tuple[int, int], phase: int) -> float:
        r, c = pos
        target = self._target_goal(phase)
        if target == G1:
            return float(self.dist_to_g1[r, c])
        return float(self.dist_to_g2[r, c])

    def _get_local_obs9(self) -> np.ndarray:
        ar, ac = self.agent_pos
        obs = np.zeros((3, 3), dtype=np.float32)
        for i, dr in enumerate([-1, 0, 1]):
            for j, dc in enumerate([-1, 0, 1]):
                rr, cc = ar + dr, ac + dc
                if rr < 0 or rr >= 9 or cc < 0 or cc >= 9:
                    obs[i, j] = float(WALL if self.boundary_as_wall else EMPTY)
                else:
                    obs[i, j] = float(self.grid[rr, cc])
        return obs.reshape(-1)

    def _get_obs(self) -> np.ndarray:
        obs9 = self._get_local_obs9()
        ctx = np.array([float(self._current_ctx())], dtype=np.float32)
        return np.concatenate([obs9, ctx], axis=0)

    def _base_info(self) -> Dict[str, object]:
        phase = int(self._current_phase())
        ctx = int(self._current_ctx())
        return {
            "t": int(self.t),
            "t_switch": int(self.t_switch),
            "order": self._order_str(),
            "context": ctx,
            "ctx": ctx,
            "phase": phase,
            "target_goal": self._target_goal_name(phase),
            "layout": self.layout_name,
            "layout_name": self.layout_name,
            "agent_pos": (int(self.agent_pos[0]), int(self.agent_pos[1])),
            "success_phase0": bool(self.phase0_hit),
            "success_phase1": bool(self.phase1_hit),
            "success_both": int(self.phase0_hit and self.phase1_hit),
        }

    # --------------------------------------------------------
    # Gym API
    # --------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.agent_pos = self.start_pos
        self.phase0_hit = False
        self.phase1_hit = False
        self.phase0_rewarded = False
        self.phase1_rewarded = False
        self.both_rewarded = False
        self._blocked_run = 0

        if self.fixed_switch is None:
            self.t_switch = int(self.rng.integers(self.switch_low, self.switch_high + 1))
        else:
            self.t_switch = int(self.fixed_switch)

        if self.fixed_order is None:
            self.order = int(self.rng.integers(0, 2))
        else:
            self.order = 0 if self.fixed_order == "AB" else 1

        obs = self._get_obs()
        info = self._base_info()
        info.update(
            {
                "hit_g1": False,
                "hit_g2": False,
                "moved": False,
                "blocked_run": 0,
                "blocked_streak": int(self.blocked_streak),
                "dbg_shaping_applied": False,
                "dbg_shaping_r": 0.0,
                "miss0_penalty": float(self.miss0_penalty),
                "miss0_applied": False,
            }
        )
        return obs, info


    def step(self, action):
        action = int(action)
        prev_pos = self.agent_pos
        phase_before = int(self._current_phase())

        # Distance before move for shaping with CURRENT phase target.
        dist_prev = self._dist_to_target(prev_pos, phase_before) if self.shaping != 0.0 else None

        # Proposed move.
        r, c = self.agent_pos
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r + 1, c
        elif action == 2:
            nr, nc = r, c - 1
        elif action == 3:
            nr, nc = r, c + 1
        else:
            raise ValueError(f"Invalid action: {action}")

        moved = False
        if 0 <= nr < 9 and 0 <= nc < 9 and self.grid[nr, nc] != WALL:
            self.agent_pos = (nr, nc)
            moved = True

        # Blocked streak accounting.
        if moved:
            self._blocked_run = 0
        else:
            self._blocked_run += 1

        reward = float(self.step_penalty)

        # Penalty only after repeated blocked moves.
        if (not moved) and (self.blocked_penalty != 0.0) and (self._blocked_run >= self.blocked_streak):
            reward -= float(self.blocked_penalty)

        # ------------------------------------------------------------
        # Potential shaping on phase-before target only.
        # ★ここは「phase1でも常に有効」にする（A学習信号を残す）
        # ------------------------------------------------------------
        shaping_r = 0.0
        shaping_applied = False
        dist_new = None
        if self.shaping != 0.0 and dist_prev is not None:
            dist_new = self._dist_to_target(self.agent_pos, phase_before)
            if np.isfinite(dist_prev) and np.isfinite(dist_new):
                dd = float(dist_prev - dist_new)
                dd = max(-1.0, min(1.0, dd))
                if float(dist_prev) <= float(self.shaping_gate):
                    shaping_applied = True
                    shaping_r = float(self.shaping) * dd
                    reward += shaping_r

        # Determine hits before advancing time.
        phase = int(self._current_phase())
        hit_g1 = self._pos_is_goal(self.agent_pos, G1)
        hit_g2 = self._pos_is_goal(self.agent_pos, G2)

        target_goal = self._target_goal(phase)
        hit_target = self._pos_is_goal(self.agent_pos, target_goal)

        # ------------------------------------------------------------
        # phase1 の goal_reward / success flag は phase0成功後のみ有効
        # shaping はこの前で通常どおり入る
        # ------------------------------------------------------------

        # phase1 の達成報酬・成功フラグを解放する条件
        phase1_reward_active = self.phase0_hit

        # wrong_goal_penalty は「phase1 かつ phase0成功後」のときだけ有効
        apply_wrong_goal_penalty = (
            self.wrong_goal_penalty != 0.0
            and phase == 1
            and self.phase0_hit
            and (hit_g1 or hit_g2)
            and (not hit_target)
        )

        if apply_wrong_goal_penalty:
            reward -= float(self.wrong_goal_penalty)
            
        # Reward each phase at most once.
        if phase == 0:
            if hit_target and (not self.phase0_rewarded):
                reward += float(self.goal_reward)
                self.phase0_rewarded = True
            self.phase0_hit = self.phase0_hit or hit_target
        else:
            if phase1_reward_active:
                if hit_target and (not self.phase1_rewarded):
                    reward += float(self.goal_reward)
                    self.phase1_rewarded = True
                self.phase1_hit = self.phase1_hit or hit_target


                
        if self.phase0_hit and self.phase1_hit and (not self.both_rewarded):
            reward += float(self.both_success_bonus)
            self.both_rewarded = True

        # Advance time.
        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_len

        miss0_applied = False
        if truncated and (self.miss0_penalty != 0.0) and (not self.phase0_hit):
            reward -= float(self.miss0_penalty)
            miss0_applied = True

        phase_after = int(self._current_phase())
        target_before = self._target_goal(phase_before)
        target_after = self._target_goal(phase_after)
        dist_prev_before = self._dist_to_target(prev_pos, phase_before)
        dist_new_before = self._dist_to_target(self.agent_pos, phase_before)

        obs = self._get_obs()
        info = self._base_info()
        info.update(
            {
            "hit_g1": bool(hit_g1),
            "hit_g2": bool(hit_g2),
            "moved": bool(moved),
            "blocked_run": int(self._blocked_run),
            "blocked_streak": int(self.blocked_streak),
            "miss0_penalty": float(self.miss0_penalty),
            "miss0_applied": bool(miss0_applied),
            # debug fields
            "dbg_phase_before": int(phase_before),
            "dbg_target_before": int(target_before),
            "dbg_dist_prev_before": float(dist_prev_before),
            "dbg_dist_new_before": float(dist_new_before),
            "dbg_shaping_applied": bool(shaping_applied),
            "dbg_shaping_r": float(shaping_r),
            "dbg_phase_after": int(phase_after),
            "dbg_target_after": int(target_after),
            # NEW debug
            "dbg_phase1_reward_active": bool(phase1_reward_active),
            "dbg_wrong_goal_penalty_applied": bool(apply_wrong_goal_penalty),
            }
        )
        return obs, float(reward), terminated, truncated, info


# Backward-compatible aliases
QtowGridEnv = QTOWGridEnv
QTOWEnv = QTOWGridEnv
