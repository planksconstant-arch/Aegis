"""
Episode-level sliding context window (Trajectory Buffer).

The Problem
-----------
Each Observation passed to the policy is stateless — the policy has no
memory of the last N ticks. This means it cannot detect patterns like:
  - "the user has rejected my last 3 suggestions"
  - "we've been bouncing between the same two files for 5 steps"
  - "pressure level has been rising for 4 consecutive observations"

The Solution
------------
A TrajectoryBuffer maintains a fixed-size deque of (observation, decision,
reward) tuples. Before each decision it produces a ContextSummary injected
into the observation's metadata. This gives the policy a compressed view of
recent history without changing the Observation schema.

Context Features Injected
--------------------------
  trajectory_length        — steps in the current episode so far
  recent_avg_reward        — mean reward over recent steps
  recent_reject_rate       — fraction of recent decisions that were rejected
  repeated_file_count      — how many open files have appeared before
  pressure_escalating      — True if pressure has increased over the window
  last_action_name         — most recent strategy_name selected
  consecutive_same_action  — how many times in a row the same action was taken
  trend_reward             — slope of reward over window (positive = improving)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from local_ide_agent.schemas import Decision, Observation


@dataclass
class StepSnapshot:
    """Lightweight snapshot of one (obs, decision, reward) triple."""
    task: str
    open_files: list[str]
    diagnostics: list[str]
    user_present: bool
    pressure: str
    action_name: str
    confidence: float
    reward: float
    accepted: bool


@dataclass
class ContextSummary:
    """Compressed context derived from the trajectory window."""
    trajectory_length: int = 0
    recent_avg_reward: float = 0.0
    recent_reject_rate: float = 0.0
    repeated_file_count: int = 0
    pressure_escalating: bool = False
    last_action_name: str = ""
    consecutive_same_action: int = 0
    trend_reward: float = 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "trajectory_length": self.trajectory_length,
            "recent_avg_reward": round(self.recent_avg_reward, 4),
            "recent_reject_rate": round(self.recent_reject_rate, 4),
            "repeated_file_count": self.repeated_file_count,
            "pressure_escalating": self.pressure_escalating,
            "last_action_name": self.last_action_name,
            "consecutive_same_action": self.consecutive_same_action,
            "trend_reward": round(self.trend_reward, 4),
        }


class TrajectoryBuffer:
    """
    Sliding window over recent (observation, decision, reward) triples.

    Parameters
    ----------
    window:    number of recent steps to retain
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self._steps: deque[StepSnapshot] = deque(maxlen=window)
        self._all_files_seen: set[str] = set()

    def record(
        self,
        observation: Observation,
        decision: Decision,
        reward: float,
        accepted: bool = False,
    ) -> None:
        """Store one step. Call after the environment returns the reward."""
        action_name = str(decision.action.payload.get("strategy_name", decision.action.action_type.value))
        snap = StepSnapshot(
            task=observation.task,
            open_files=list(observation.open_files),
            diagnostics=list(observation.diagnostics),
            user_present=observation.user_present,
            pressure=str(observation.metadata.get("pressure_level", "normal")),
            action_name=action_name,
            confidence=float(decision.confidence),
            reward=reward,
            accepted=accepted,
        )
        self._steps.append(snap)
        self._all_files_seen.update(observation.open_files)

    def reset(self) -> None:
        """Call at the start of each episode to clear per-episode state."""
        self._steps.clear()
        # Keep _all_files_seen across episodes (cross-episode file memory)

    def summarize(self, current_observation: Observation) -> ContextSummary:
        """Produce a ContextSummary from the current window."""
        steps = list(self._steps)
        n = len(steps)

        if n == 0:
            return ContextSummary()

        rewards = [s.reward for s in steps]
        avg_reward = sum(rewards) / n
        reject_rate = sum(1 for s in steps if not s.accepted) / n

        # Repeated files: how many current open files were seen before
        current_files = set(current_observation.open_files)
        old_files = self._all_files_seen - current_files
        repeated = len(current_files & old_files)

        # Pressure escalation: check if pressure increased over window
        pressure_map = {"low": 0, "normal": 1, "high": 2}
        pressures = [pressure_map.get(s.pressure, 1) for s in steps]
        pressure_escalating = len(pressures) >= 2 and pressures[-1] > pressures[0]

        # Consecutive same action
        last_action = steps[-1].action_name if steps else ""
        consecutive = 0
        for s in reversed(steps):
            if s.action_name == last_action:
                consecutive += 1
            else:
                break

        # Reward trend (linear slope via simple least-squares)
        trend = 0.0
        if n >= 3:
            xs = list(range(n))
            x_mean = sum(xs) / n
            y_mean = avg_reward
            numer = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, rewards))
            denom = sum((xi - x_mean) ** 2 for xi in xs) or 1.0
            trend = numer / denom

        return ContextSummary(
            trajectory_length=n,
            recent_avg_reward=avg_reward,
            recent_reject_rate=reject_rate,
            repeated_file_count=repeated,
            pressure_escalating=pressure_escalating,
            last_action_name=last_action,
            consecutive_same_action=consecutive,
            trend_reward=trend,
        )

    def enrich_observation(self, observation: Observation) -> Observation:
        """
        Return a copy of the observation with context summary injected
        into the metadata. The policy can read these as normal metadata keys.
        """
        summary = self.summarize(observation)
        enriched_meta = {**observation.metadata, **summary.to_metadata()}
        return observation.model_copy(update={"metadata": enriched_meta})

    def __len__(self) -> int:
        return len(self._steps)
