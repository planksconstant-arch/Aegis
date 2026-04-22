from __future__ import annotations

import shutil
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from local_ide_agent.config import ShadowWorkspaceSettings


@dataclass
class ShadowWorkspace:
    shadow_id: str
    source_root: Path
    shadow_root: Path
    created_at: str


class ShadowWorkspaceManager:
    def __init__(self, workspace_root: Path, settings: ShadowWorkspaceSettings) -> None:
        self.workspace_root = workspace_root
        self.settings = settings
        self.shadow_root = (workspace_root / settings.root_directory).resolve()
        self.shadow_root.mkdir(parents=True, exist_ok=True)

    def create_shadow_copy(self, label: str = "recent") -> ShadowWorkspace:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        shadow_id = f"{label}-{timestamp}"
        target = self.shadow_root / shadow_id
        ignore = shutil.ignore_patterns(
            "__pycache__",
            ".git",
            ".shadow",
            ".shadow_eval",
            ".agent",
            ".venv",
            "node_modules",
        )
        shutil.copytree(self.workspace_root, target, ignore=ignore, dirs_exist_ok=False)
        self._enforce_retention()
        return ShadowWorkspace(
            shadow_id=shadow_id,
            source_root=self.workspace_root,
            shadow_root=target,
            created_at=datetime.now(UTC).isoformat(),
        )

    def list_shadows(self) -> list[ShadowWorkspace]:
        items: list[ShadowWorkspace] = []
        for path in sorted(self.shadow_root.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            items.append(
                ShadowWorkspace(
                    shadow_id=path.name,
                    source_root=self.workspace_root,
                    shadow_root=path,
                    created_at=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                )
            )
        return items

    def write_source_artifact(self, relative_path: str, payload: dict[str, object]) -> Path:
        target = self.workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def _enforce_retention(self) -> None:
        items = [item for item in self.shadow_root.iterdir() if item.is_dir()]
        if len(items) <= self.settings.retention_limit:
            return
        overflow = len(items) - self.settings.retention_limit
        for item in sorted(items, key=lambda p: p.stat().st_mtime)[:overflow]:
            try:
                shutil.rmtree(item)
            except PermissionError:
                # Older shadow copies can still be temporarily locked on Windows.
                # Skipping cleanup is safer than failing the new autonomous run.
                continue
