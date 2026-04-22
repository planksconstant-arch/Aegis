from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodexBridge:
    """
    Placeholder for coding-model integration.

    In a real system this adapter could:
    - draft plans,
    - critique candidate actions,
    - estimate reward from outcomes,
    - or provide tool-use priors for the RL policy.
    """

    model_name: str = "codex-style-assistant"

    def score_action(self, task: str, candidate: str) -> float:
        text = f"{task}::{candidate}".lower()
        score = 0.5
        if "fix" in text or "test" in text:
            score += 0.2
        if "delete" in text or "reset" in text:
            score -= 0.3
        return max(0.0, min(score, 1.0))
