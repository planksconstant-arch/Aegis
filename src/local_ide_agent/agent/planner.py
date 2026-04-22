from __future__ import annotations

from local_ide_agent.schemas import Decision, Observation


class Planner:
    """Turns raw observations into a concise planning string for the policy/runtime."""

    def summarize(self, observation: Observation) -> str:
        file_count = len(observation.open_files)
        diagnostic_count = len(observation.diagnostics)
        return (
            f"Task='{observation.task}', open_files={file_count}, "
            f"diagnostics={diagnostic_count}, user_present={observation.user_present}"
        )

    def explain(self, decision: Decision) -> str:
        return f"{decision.action.action_type.value}: {decision.reason}"
