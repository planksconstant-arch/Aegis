from __future__ import annotations

from dataclasses import dataclass

from local_ide_agent.schemas import ActionType, RiskLevel


@dataclass(frozen=True)
class StrategyAction:
    name: str
    action_type: ActionType
    risk: RiskLevel
    autonomy_tier: str
    description: str


ACTION_SPACE: list[StrategyAction] = [
    StrategyAction("minimal_patch", ActionType.SUGGEST, RiskLevel.LOW, "auto-low-risk", "Apply a minimal patch."),
    StrategyAction("targeted_refactor", ActionType.SUGGEST, RiskLevel.MEDIUM, "review", "Propose a targeted refactor."),
    StrategyAction("expand_abstraction", ActionType.SUGGEST, RiskLevel.MEDIUM, "review", "Introduce a broader abstraction."),
    StrategyAction("add_tests", ActionType.SUGGEST, RiskLevel.LOW, "review", "Add or propose tests."),
    StrategyAction("ask_review", ActionType.ASK_APPROVAL, RiskLevel.MEDIUM, "review", "Ask for human review."),
    StrategyAction("silent_shadow", ActionType.OBSERVE, RiskLevel.LOW, "silent-shadow", "Explore silently in a shadow workspace."),
    StrategyAction("promote_candidate", ActionType.EDIT_FILE, RiskLevel.HIGH, "review", "Promote the current best candidate."),
    StrategyAction("no_op", ActionType.NO_OP, RiskLevel.LOW, "review", "Take no action."),
]
