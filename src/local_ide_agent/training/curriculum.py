"""
Curriculum Learning Scheduler for the simulated coding environment.

Why this matters
----------------
Random task sampling makes training hard: the agent sees "Propose a safe
database migration" before it has learned to handle "Fix a failing test".
A curriculum that starts easy and progressively introduces harder tasks
accelerates convergence by 2-5x in practice.

Design
------
  Difficulty levels:
    0 — EASY:    single-file, diagnostic present (clear target), low pressure
    1 — MEDIUM:  multi-file, mixed pressure, no diagnostic
    2 — HARD:    multi-step, high pressure, ambiguous task, no diagnostic

  Promotion policy:
    Promote to next level  when rolling_success_rate > success_threshold
    Demote  to prev level  when rolling_success_rate < demotion_threshold
    (Success = episode completed AND reward > 0)

Usage
-----
  curriculum = CurriculumScheduler(environment)
  curriculum.reset()   # start of training
  ...
  curriculum.record_outcome(reward=0.8, completed=True)  # after each episode
  # The environment's difficulty is auto-updated
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from local_ide_agent.training.environment import SimulatedCodingEnvironment, _TASKS


class Difficulty(IntEnum):
    EASY = 0
    MEDIUM = 1
    HARD = 2


# Task partitioning by difficulty
_EASY_TASKS = [
    t for t in _TASKS
    if t["diag"]  # Clear diagnostic = easy target
]

_MEDIUM_TASKS = [
    t for t in _TASKS
    if not t["diag"] and len(t["files"]) == 1
]

_HARD_TASKS = [
    t for t in _TASKS
    if not t["diag"] and len(t["files"]) > 1
]

# Ensure we always have tasks in each bucket
if not _EASY_TASKS:
    _EASY_TASKS = _TASKS[:4]
if not _MEDIUM_TASKS:
    _MEDIUM_TASKS = _TASKS[4:12]
if not _HARD_TASKS:
    _HARD_TASKS = _TASKS[12:]

_TASK_POOLS: dict[Difficulty, list[dict]] = {
    Difficulty.EASY:   _EASY_TASKS,
    Difficulty.MEDIUM: _MEDIUM_TASKS,
    Difficulty.HARD:   _HARD_TASKS,
}

_MAX_STEPS_BY_DIFFICULTY: dict[Difficulty, int] = {
    Difficulty.EASY:   4,
    Difficulty.MEDIUM: 7,
    Difficulty.HARD:   10,
}


@dataclass
class CurriculumScheduler:
    """
    Wraps a SimulatedCodingEnvironment and manages difficulty progression.

    Parameters
    ----------
    environment:          the environment to control
    window_size:          number of recent episodes used for promotion decisions
    success_threshold:    rolling success rate required to promote difficulty
    demotion_threshold:   rolling success rate at which difficulty is demoted
    start_difficulty:     initial difficulty level
    """

    environment: SimulatedCodingEnvironment
    window_size: int = 20
    success_threshold: float = 0.70
    demotion_threshold: float = 0.35
    start_difficulty: Difficulty = Difficulty.EASY

    # Runtime state
    _difficulty: Difficulty = field(init=False)
    _outcomes: deque = field(init=False)
    _episode_count: int = field(init=False, default=0)
    _promotions: int = field(init=False, default=0)
    _demotions: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._difficulty = self.start_difficulty
        self._outcomes = deque(maxlen=self.window_size)
        self._apply_difficulty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def difficulty(self) -> Difficulty:
        return self._difficulty

    @property
    def rolling_success_rate(self) -> float:
        if not self._outcomes:
            return 0.5  # Neutral until we have data
        return sum(self._outcomes) / len(self._outcomes)

    def record_outcome(self, reward: float, completed: bool) -> None:
        """Call after each episode. Updates rolling stats and may change difficulty."""
        success = completed and reward > 0.0
        self._outcomes.append(1.0 if success else 0.0)
        self._episode_count += 1

        if len(self._outcomes) < max(5, self.window_size // 4):
            return  # Need a minimum sample before promoting/demoting

        rate = self.rolling_success_rate
        if rate >= self.success_threshold and self._difficulty < Difficulty.HARD:
            self._difficulty = Difficulty(int(self._difficulty) + 1)
            self._promotions += 1
            self._outcomes.clear()  # Fresh window for new difficulty
            self._apply_difficulty()

        elif rate <= self.demotion_threshold and self._difficulty > Difficulty.EASY:
            self._difficulty = Difficulty(int(self._difficulty) - 1)
            self._demotions += 1
            self._outcomes.clear()
            self._apply_difficulty()

    def _apply_difficulty(self) -> None:
        """Update the environment's task pool and max steps."""
        pool = _TASK_POOLS[self._difficulty]
        self.environment.tasks = list(pool)
        self.environment.max_steps = _MAX_STEPS_BY_DIFFICULTY[self._difficulty]
        self.environment.cursor = 0
        random.shuffle(self.environment.tasks)

    def stats(self) -> dict[str, object]:
        return {
            "difficulty": self._difficulty.name,
            "difficulty_level": int(self._difficulty),
            "rolling_success_rate": round(self.rolling_success_rate, 3),
            "episode_count": self._episode_count,
            "promotions": self._promotions,
            "demotions": self._demotions,
            "task_pool_size": len(self.environment.tasks),
        }
