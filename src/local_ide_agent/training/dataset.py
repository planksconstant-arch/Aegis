from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class Transition:
    observation_text: str
    action_type: str
    reward: float
    done: bool
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Trajectory:
    trajectory_id: str
    user_id: str
    transitions: list[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def discounted_return(self, gamma: float = 0.99) -> float:
        total = 0.0
        weight = 1.0
        for transition in self.transitions:
            total += transition.reward * weight
            weight *= gamma
        return total


@dataclass
class OfflineRLDataset:
    trajectories: list[Trajectory] = field(default_factory=list)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def stats(self) -> dict[str, float]:
        returns = [item.discounted_return() for item in self.trajectories]
        lengths = [len(item.transitions) for item in self.trajectories]
        if not returns:
            return {"trajectory_count": 0.0, "average_return": 0.0, "average_length": 0.0}
        return {
            "trajectory_count": float(len(self.trajectories)),
            "average_return": mean(returns),
            "average_length": mean(lengths),
        }
