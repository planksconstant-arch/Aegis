from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from local_ide_agent.connectors.base import IDEConnector
from local_ide_agent.schemas import Decision, Observation


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class EventLog:
    path: Path
    entries: list[dict[str, object]] = field(default_factory=list)

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self.entries.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class LocalIDEConnector(IDEConnector):
    """
    Workspace-aware connector for local IDE integration.

    This connector is still conservative:
    - all paths must stay inside the configured workspace root,
    - command execution is explicit and bounded,
    - suggestions and approvals are logged for later training.
    """

    def __init__(self, workspace_root: str | Path, event_log_path: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if event_log_path:
            log_path = Path(event_log_path)
            if not log_path.is_absolute():
                log_path = self.workspace_root / log_path
        else:
            log_path = self.workspace_root / ".agent" / "events.jsonl"
        self.event_log = EventLog(path=log_path)

    def present_suggestion(self, decision: Decision) -> str:
        message = f"Suggestion shown in IDE: {decision.action.description}"
        self.event_log.append(
            "suggestion_presented",
            {
                "description": decision.action.description,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        )
        return message

    def request_approval(self, decision: Decision) -> str:
        message = f"Approval requested in IDE: {decision.action.description}"
        self.event_log.append(
            "approval_requested",
            {
                "description": decision.action.description,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        )
        return message

    def run_command(self, command: str) -> str:
        if not command:
            return "No command provided."

        result = self.execute_command(command)
        self.event_log.append(
            "command_executed",
            {
                "command": result.command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )

        if result.returncode != 0:
            return f"Command failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        return result.stdout.strip() or "Command completed successfully."

    def apply_edit(self, path: str, content: str) -> str:
        if not path:
            return "No edit applied because no target path was provided."

        target = self.resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.event_log.append(
            "file_edited",
            {
                "path": str(target),
                "size": len(content),
            },
        )
        return f"Edit applied to {target}"

    def read_file(self, path: str) -> str:
        target = self.resolve_workspace_path(path)
        return target.read_text(encoding="utf-8")

    def list_files(self) -> list[str]:
        return [
            str(item.relative_to(self.workspace_root))
            for item in self.workspace_root.rglob("*")
            if item.is_file() and ".agent" not in item.parts
        ]

    def summarize_context(self, observation: Observation) -> dict[str, object]:
        return {
            "task": observation.task,
            "open_file_count": len(observation.open_files),
            "diagnostic_count": len(observation.diagnostics),
            "top_files": observation.open_files[:5],
            "user_present": observation.user_present,
        }

    def execute_command(self, command: str, timeout_seconds: int = 30) -> CommandResult:
        completed = subprocess.run(
            command,
            cwd=self.workspace_root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def resolve_workspace_path(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if self.workspace_root not in resolved.parents and resolved != self.workspace_root:
            raise ValueError(f"Path escapes workspace root: {path}")
        return resolved
