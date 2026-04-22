from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from local_ide_agent.lab.counterfactual import CandidateScore
from local_ide_agent.schemas import MemorySnapshot
from local_ide_agent.shadow.workspace import ShadowWorkspace


@dataclass
class CandidateExecutionResult:
    changed_files: list[str]
    validation_command: str | None
    validation_returncode: int | None
    validation_stdout: str
    validation_stderr: str
    comparison_report_path: str


class CounterfactualExecutor:
    def materialize_candidate(
        self,
        shadow: ShadowWorkspace,
        candidate: CandidateScore,
        snapshot: MemorySnapshot,
        objective: str,
    ) -> CandidateExecutionResult:
        lab_dir = shadow.shadow_root / ".agent_lab"
        lab_dir.mkdir(parents=True, exist_ok=True)

        style_map = {item.key: item.value for item in snapshot.style_preferences}
        strategy_payload = {
            "profile": candidate.profile.name,
            "autonomy_mode": candidate.profile.autonomy_mode,
            "planning_depth": candidate.profile.planning_depth,
            "edit_style": candidate.profile.edit_style,
            "command_risk": candidate.profile.command_risk,
            "score": candidate.score,
            "reasons": candidate.reasons,
            "objective": objective,
            "user_style": style_map,
            "preferred_actions": snapshot.preferred_actions,
        }

        strategy_path = lab_dir / "strategy.json"
        strategy_path.write_text(json.dumps(strategy_payload, indent=2), encoding="utf-8")

        prompt_path = lab_dir / "PROMPT.md"
        prompt_path.write_text(self._build_prompt(candidate, snapshot, objective), encoding="utf-8")

        plan_path = lab_dir / "PLAN.md"
        plan_path.write_text(self._build_plan(candidate, objective), encoding="utf-8")

        validation_command = self._detect_validation_command(shadow.shadow_root)
        returncode: int | None = None
        stdout = ""
        stderr = ""
        if validation_command:
            completed = subprocess.run(
                validation_command,
                cwd=shadow.shadow_root,
                shell=True,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout[-4000:]
            stderr = completed.stderr[-4000:]

        report = {
            "shadow_id": shadow.shadow_id,
            "objective": objective,
            "profile": candidate.profile.name,
            "score": candidate.score,
            "materialized_files": [
                str(strategy_path.relative_to(shadow.shadow_root)),
                str(prompt_path.relative_to(shadow.shadow_root)),
                str(plan_path.relative_to(shadow.shadow_root)),
            ],
            "validation_command": validation_command,
            "validation_returncode": returncode,
            "validation_stdout_tail": stdout,
            "validation_stderr_tail": stderr,
        }
        report_path = lab_dir / "comparison_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        changed_files = [
            str(strategy_path.relative_to(shadow.shadow_root)),
            str(prompt_path.relative_to(shadow.shadow_root)),
            str(plan_path.relative_to(shadow.shadow_root)),
            str(report_path.relative_to(shadow.shadow_root)),
        ]
        return CandidateExecutionResult(
            changed_files=changed_files,
            validation_command=validation_command,
            validation_returncode=returncode,
            validation_stdout=stdout,
            validation_stderr=stderr,
            comparison_report_path=str(report_path),
        )

    def _build_prompt(self, candidate: CandidateScore, snapshot: MemorySnapshot, objective: str) -> str:
        recent_tasks = "\n".join(f"- {item}" for item in snapshot.recent_tasks[:5]) or "- No recent tasks recorded."
        failures = "\n".join(f"- {item}" for item in snapshot.recent_failures[:5]) or "- No recent failures recorded."
        return (
            f"# Counterfactual Strategy Prompt\n\n"
            f"Objective: {objective}\n\n"
            f"Profile: {candidate.profile.name}\n"
            f"Autonomy mode: {candidate.profile.autonomy_mode}\n"
            f"Planning depth: {candidate.profile.planning_depth}\n"
            f"Edit style: {candidate.profile.edit_style}\n"
            f"Command risk: {candidate.profile.command_risk}\n\n"
            f"Reasons this candidate was selected:\n"
            + "\n".join(f"- {item}" for item in candidate.reasons)
            + "\n\nRecent tasks:\n"
            + recent_tasks
            + "\n\nRecent failures to avoid:\n"
            + failures
            + "\n"
        )

    def _build_plan(self, candidate: CandidateScore, objective: str) -> str:
        return (
            f"# Strategy Plan\n\n"
            f"1. Restate the task in the profile's own style.\n"
            f"2. Inspect only files necessary for: {objective}\n"
            f"3. Prefer {candidate.profile.edit_style} edits.\n"
            f"4. Keep command risk at {candidate.profile.command_risk}.\n"
            f"5. Produce a diff and validation summary before any promotion.\n"
        )

    def _detect_validation_command(self, workspace_root: Path) -> str | None:
        if (workspace_root / "pytest.ini").exists() or (workspace_root / "tests").exists():
            return "python -m pytest -q"
        if (workspace_root / "package.json").exists():
            return "npm test"
        if (workspace_root / "pyproject.toml").exists():
            return "python -m compileall src"
        return None
