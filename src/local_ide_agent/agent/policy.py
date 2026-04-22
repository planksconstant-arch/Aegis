from __future__ import annotations

from dataclasses import dataclass, field

from local_ide_agent.schemas import Action, ActionType, Decision, Observation, RiskLevel


class Policy:
    def decide(self, observation: Observation) -> Decision:
        raise NotImplementedError

    def update(self, reward: float) -> None:
        raise NotImplementedError


@dataclass
class HeuristicPolicy(Policy):
    learned_bias: float = 0.0
    reward_history: list[float] = field(default_factory=list)

    def decide(self, observation: Observation) -> Decision:
        style_map = observation.metadata.get("style_preferences", {})
        preferred_actions = set(observation.metadata.get("preferred_actions", []))
        temporal_patterns = observation.metadata.get("temporal_patterns", {})
        confidence_bonus = 0.05 if "suggest" in preferred_actions else 0.0
        conservative_mode = style_map.get("autonomy_mode") == "conservative"
        current_hour = int(observation.metadata.get("local_hour", 12))
        hour_bucket = f"hour_{current_hour}"
        confidence_bonus += float(temporal_patterns.get(hour_bucket, 0.0)) * 0.05

        if observation.diagnostics:
            action = Action(
                action_type=ActionType.SUGGEST,
                description="Suggest a fix for the active diagnostics.",
                risk=RiskLevel.LOW,
                payload={"diagnostics": observation.diagnostics},
            )
            return Decision(
                action=action,
                confidence=min(0.55 + self.learned_bias + confidence_bonus, 0.95),
                requires_approval=False,
                reason="Diagnostics provide a clear low-risk opportunity to help.",
                autonomy_tier="review" if conservative_mode else "auto-low-risk",
            )

        if not observation.user_present:
            action = Action(
                action_type=ActionType.OBSERVE,
                description="Observe and collect more context while user is away.",
                risk=RiskLevel.LOW,
            )
            return Decision(
                action=action,
                confidence=0.45,
                requires_approval=False,
                reason="Background mode should stay conservative until the policy is stronger.",
                autonomy_tier="silent-shadow",
            )

        action = Action(
            action_type=ActionType.ASK_APPROVAL,
            description="Ask whether to draft a plan for the current task.",
            risk=RiskLevel.MEDIUM if conservative_mode else RiskLevel.LOW,
        )
        return Decision(
            action=action,
            confidence=0.50 + confidence_bonus,
            requires_approval=conservative_mode,
            reason="Defaulting to a cautious handoff when context is incomplete.",
            autonomy_tier="review" if conservative_mode else "auto-low-risk",
        )

    def update(self, reward: float) -> None:
        self.reward_history.append(reward)
        self.learned_bias = max(min(sum(self.reward_history) / (len(self.reward_history) * 10), 0.35), -0.2)
