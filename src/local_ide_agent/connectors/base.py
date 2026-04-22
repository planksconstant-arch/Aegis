from __future__ import annotations

from local_ide_agent.schemas import Decision, Observation


class IDEConnector:
    def present_suggestion(self, decision: Decision) -> str:
        raise NotImplementedError

    def request_approval(self, decision: Decision) -> str:
        raise NotImplementedError

    def run_command(self, command: str) -> str:
        raise NotImplementedError

    def apply_edit(self, path: str, content: str) -> str:
        raise NotImplementedError

    def summarize_context(self, observation: Observation) -> dict[str, object]:
        raise NotImplementedError
