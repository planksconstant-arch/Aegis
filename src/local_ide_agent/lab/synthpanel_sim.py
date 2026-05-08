from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwarmAgent:
    name: str
    role: str
    bias: str


@dataclass
class SwarmFinding:
    agent_name: str
    verdict: str
    confidence: float
    rationale: str


class SynthPanelSimulator:
    """
    SynthPanel — a synthetic stakeholder review panel.

    Simulates a diverse panel of engineering reviewers (Staff Engineer,
    QA Lead, Security Reviewer, DevOps Lead, Product Engineer) who each
    evaluate candidate implementation paths before promotion, providing
    multi-perspective risk assessment.
    """

    def __init__(self) -> None:
        self.agents = [
            SwarmAgent("Asha", "Staff Engineer", "maintainability-first"),
            SwarmAgent("Milan", "QA Lead", "regression-averse"),
            SwarmAgent("Rhea", "Security Reviewer", "risk-sensitive"),
            SwarmAgent("Ishan", "DevOps Lead", "operability-focused"),
            SwarmAgent("Tara", "Product Engineer", "speed-with-safety"),
        ]

    def review_candidate(self, candidate: dict[str, object], objective: str) -> dict[str, object]:
        findings = [self._review(agent, candidate, objective) for agent in self.agents]
        average_confidence = round(sum(item.confidence for item in findings) / max(len(findings), 1), 3)
        approval_count = sum(1 for item in findings if item.verdict == "approve")
        caution_count = sum(1 for item in findings if item.verdict == "caution")
        reject_count = sum(1 for item in findings if item.verdict == "reject")

        return {
            "objective": objective,
            "approval_count": approval_count,
            "caution_count": caution_count,
            "reject_count": reject_count,
            "average_confidence": average_confidence,
            "findings": [
                {
                    "agent_name": item.agent_name,
                    "verdict": item.verdict,
                    "confidence": item.confidence,
                    "rationale": item.rationale,
                }
                for item in findings
            ],
        }

    def _review(self, agent: SwarmAgent, candidate: dict[str, object], objective: str) -> SwarmFinding:
        profile = str(candidate.get("profile", "unknown"))
        score = float(candidate.get("score", 0.5))
        validation_returncode = candidate.get("validation_returncode")
        validation_failed = validation_returncode not in (None, 0)
        objective_lower = objective.lower()

        verdict = "approve"
        confidence = min(0.55 + score / 2, 0.98)
        rationale_parts: list[str] = [f"{agent.role} review on profile '{profile}'."]

        if validation_failed:
            verdict = "caution"
            confidence -= 0.15
            rationale_parts.append("Validation did not pass cleanly.")

        if "safe" in objective_lower and "aggressive" in profile:
            verdict = "reject"
            confidence -= 0.10
            rationale_parts.append("Objective emphasizes safety but profile is aggressive.")

        if agent.role == "Security Reviewer" and str(candidate.get("autonomy_mode")) == "aggressive":
            verdict = "reject"
            confidence -= 0.10
            rationale_parts.append("High-autonomy profile increases promotion risk.")

        if agent.role == "QA Lead" and validation_failed:
            verdict = "reject"
            confidence -= 0.05
            rationale_parts.append("Failing validation is a hard blocker for QA.")

        if agent.role == "Staff Engineer" and str(candidate.get("edit_style")) == "minimal-diff":
            rationale_parts.append("Minimal diff style is easier to review and merge.")

        if agent.role == "DevOps Lead" and candidate.get("validation_command"):
            rationale_parts.append(f"Candidate attempted validation via {candidate['validation_command']}.")

        confidence = round(max(0.1, min(confidence, 0.99)), 3)
        return SwarmFinding(
            agent_name=agent.name,
            verdict=verdict,
            confidence=confidence,
            rationale=" ".join(rationale_parts),
        )
