from __future__ import annotations

from dataclasses import dataclass, field

from local_ide_agent.config import DeploymentSettings
from local_ide_agent.lab.counterfactual import CounterfactualPlanner
from local_ide_agent.lab.executor import CounterfactualExecutor
from local_ide_agent.lab.synthpanel_sim import SynthPanelSimulator
from local_ide_agent.memory.store import MemoryStore
from local_ide_agent.shadow.workspace import ShadowWorkspaceManager


@dataclass
class BackgroundAgent:
    agent_id: str
    role: str
    active: bool = True


@dataclass
class BackgroundDeploymentManager:
    settings: DeploymentSettings
    agents: list[BackgroundAgent] = field(default_factory=list)

    def deploy(self, role: str) -> str:
        if len(self.agents) >= self.settings.max_background_agents:
            return "Deployment refused because the background agent limit was reached."

        agent = BackgroundAgent(
            agent_id=f"{self.settings.runtime_name}-{len(self.agents) + 1}",
            role=role,
        )
        self.agents.append(agent)
        return f"Deployed background agent {agent.agent_id} for role '{role}'."

    def heartbeat(self) -> list[str]:
        return [f"{agent.agent_id}: active={agent.active}" for agent in self.agents]


@dataclass
class ShadowRunRequest:
    user_id: str
    objective: str
    label: str = "recent"


@dataclass
class ShadowBackgroundService:
    manager: BackgroundDeploymentManager
    shadow_manager: ShadowWorkspaceManager
    memory_store: MemoryStore
    planner: CounterfactualPlanner | None = None
    executor: CounterfactualExecutor | None = None
    simulator: SynthPanelSimulator | None = None

    def start_run(self, request: ShadowRunRequest) -> dict[str, object]:
        shadow = self.shadow_manager.create_shadow_copy(request.label)
        deploy_message = self.manager.deploy("shadow-coder")
        summary = {
            "deploy_message": deploy_message,
            "shadow_id": shadow.shadow_id,
            "shadow_root": str(shadow.shadow_root),
            "objective": request.objective,
            "mode": "copy-first-autonomy",
            "next_step": "Run planning and edits inside the shadow workspace before proposing promotion.",
        }
        self.memory_store.record_shadow_run(
            shadow_id=shadow.shadow_id,
            user_id=request.user_id,
            source_root=str(shadow.source_root),
            shadow_root=str(shadow.shadow_root),
            objective=request.objective,
            status="created",
            summary=summary,
        )
        return summary

    def start_counterfactual_run(self, request: ShadowRunRequest) -> dict[str, object]:
        planner = self.planner or CounterfactualPlanner()
        executor = self.executor or CounterfactualExecutor()
        simulator = self.simulator or SynthPanelSimulator()
        snapshot = self.memory_store.snapshot(request.user_id)
        ranked_profiles = planner.score_profiles(snapshot, request.objective)
        candidates: list[dict[str, object]] = []

        for profile_score in ranked_profiles:
            label = f"{request.label}-{profile_score.profile.name}"
            shadow = self.shadow_manager.create_shadow_copy(label)
            deploy_message = self.manager.deploy(f"counterfactual-{profile_score.profile.name}")
            execution = executor.materialize_candidate(shadow, profile_score, snapshot, request.objective)
            candidate = {
                "shadow_id": shadow.shadow_id,
                "shadow_root": str(shadow.shadow_root),
                "profile": profile_score.profile.name,
                "autonomy_mode": profile_score.profile.autonomy_mode,
                "planning_depth": profile_score.profile.planning_depth,
                "edit_style": profile_score.profile.edit_style,
                "score": profile_score.score,
                "reasons": profile_score.reasons,
                "deploy_message": deploy_message,
                "changed_files": execution.changed_files,
                "validation_command": execution.validation_command,
                "validation_returncode": execution.validation_returncode,
                "comparison_report_path": execution.comparison_report_path,
            }
            swarm_report = simulator.review_candidate(candidate, request.objective)
            candidate["swarm_report"] = swarm_report
            candidate["swarm_score"] = round(
                swarm_report["approval_count"] + (swarm_report["average_confidence"] * 0.5) - swarm_report["reject_count"] * 0.5,
                3,
            )
            candidates.append(candidate)
            self.memory_store.record_shadow_run(
                shadow_id=shadow.shadow_id,
                user_id=request.user_id,
                source_root=str(shadow.source_root),
                shadow_root=str(shadow.shadow_root),
                objective=request.objective,
                status="counterfactual-created",
                summary=candidate,
            )

        candidates.sort(key=lambda item: (float(item.get("swarm_score", 0.0)), float(item.get("score", 0.0))), reverse=True)
        winner = candidates[0] if candidates else {}
        winner_explanation = self._build_winner_explanation(winner, snapshot)
        aggregate = {
            "mode": "counterfactual-shadow-lab",
            "objective": request.objective,
            "user_id": request.user_id,
            "winner": winner,
            "winner_explanation": winner_explanation,
            "candidates": candidates,
            "next_step": "Run autonomous edits in the highest-ranked shadow first, then compare diffs before promotion.",
        }
        report_path = self.shadow_manager.write_source_artifact(".agent/counterfactual-latest.json", aggregate)
        aggregate["aggregate_report_path"] = str(report_path)
        return aggregate

    def _build_winner_explanation(self, winner: dict[str, object], snapshot: object) -> str:
        if not winner:
            return "No winning candidate was available."
        reasons = list(winner.get("reasons", []))
        swarm_report = dict(winner.get("swarm_report", {}))
        approvals = int(swarm_report.get("approval_count", 0))
        cautions = int(swarm_report.get("caution_count", 0))
        rejects = int(swarm_report.get("reject_count", 0))
        principles = getattr(snapshot, "global_style_principles", [])

        parts = [
            f"Selected '{winner.get('profile', 'unknown')}' because it aligned with the learned user profile.",
        ]
        if reasons:
            parts.append("Model reasons: " + "; ".join(str(item) for item in reasons[:3]) + ".")
        parts.append(f"Swarm review summary: approvals={approvals}, cautions={cautions}, rejects={rejects}.")
        if principles:
            parts.append("Cross-project style principles considered: " + "; ".join(principles[:2]) + ".")
        validation_returncode = winner.get("validation_returncode")
        if validation_returncode not in (None, 0):
            parts.append("Validation still needs follow-up before promotion.")
        return " ".join(parts)
