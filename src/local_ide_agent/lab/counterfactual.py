from __future__ import annotations

from dataclasses import dataclass

from local_ide_agent.schemas import MemorySnapshot


@dataclass
class StrategyProfile:
    name: str
    autonomy_mode: str
    planning_depth: str
    edit_style: str
    command_risk: str


@dataclass
class CandidateScore:
    profile: StrategyProfile
    score: float
    reasons: list[str]


class CounterfactualPlanner:
    """
    Generates strategy variants and scores them against the user's learned profile.

    This is not the final RL model. It is a novel orchestration layer that creates
    multiple candidate futures and ranks them before any real promotion happens.
    """

    def generate_profiles(self) -> list[StrategyProfile]:
        return [
            StrategyProfile("conservative-maintainer", "conservative", "deep", "minimal-diff", "low"),
            StrategyProfile("balanced-refactorer", "balanced", "medium", "targeted-refactor", "medium"),
            StrategyProfile("aggressive-optimizer", "aggressive", "shallow", "broad-refactor", "high"),
        ]

    def score_profiles(self, snapshot: MemorySnapshot, objective: str) -> list[CandidateScore]:
        style_map = {item.key: item.value for item in snapshot.style_preferences}
        preferred_actions = set(snapshot.preferred_actions)
        objective_lower = objective.lower()

        scores: list[CandidateScore] = []
        for profile in self.generate_profiles():
            score = 0.5
            reasons: list[str] = []

            if style_map.get("autonomy_mode") == profile.autonomy_mode:
                score += 0.25
                reasons.append("Matches stored autonomy preference.")

            if "suggest" in preferred_actions and profile.autonomy_mode in {"conservative", "balanced"}:
                score += 0.10
                reasons.append("Aligns with previously rewarded suggestion-first behavior.")

            if "refactor" in objective_lower and "refactor" in profile.edit_style:
                score += 0.15
                reasons.append("Objective mentions refactoring and profile supports it.")

            if "safe" in objective_lower and profile.command_risk == "low":
                score += 0.15
                reasons.append("Objective emphasizes safety.")

            if profile.command_risk == "high":
                score -= 0.10
                reasons.append("Penalized for higher operational risk.")

            scores.append(CandidateScore(profile=profile, score=round(score, 3), reasons=reasons))

        return sorted(scores, key=lambda item: item.score, reverse=True)
