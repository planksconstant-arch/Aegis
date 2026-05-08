from __future__ import annotations

import os
import subprocess
import sys
import shutil
from pathlib import Path

from local_ide_agent.config import ShadowWorkspaceSettings
from local_ide_agent.schemas import CandidatePatch, StructuredReward
from local_ide_agent.shadow.workspace import ShadowWorkspaceManager

_MODULE_NOT_FOUND = "No module named"


class PatchEvaluator:
    """
    Evaluates CandidatePatches by applying them in a shadow workspace
    and verifying if the code compiles, lints, and passes tests.
    """

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        settings = ShadowWorkspaceSettings(
            enabled=True,
            root_directory=".shadow_eval",
            retention_limit=2,
        )
        self.shadow_manager = ShadowWorkspaceManager(self.workspace_root, settings)

    def evaluate_candidate(self, candidate: CandidatePatch, target_relative_path: str) -> StructuredReward:
        """Applies the candidate's content to the target file in a shadow workspace and scores it."""
        shadow = self.shadow_manager.get_persistent_eval_workspace()
        target_path = shadow.shadow_root / target_relative_path
        
        # Read the original content to perform a lightning-fast revert
        original_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        file_existed = target_path.exists()
        
        try:
            # 1. Apply the candidate replacement
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(candidate.diff, encoding="utf-8")
            
            # 2. Check compilation (Python syntax)
            compiles = True
            try:
                compile(target_path.read_text(encoding="utf-8"), str(target_path), "exec")
            except SyntaxError:
                compiles = False
                
            # 3. Check Linter (using ruff if available, else flake8/pylint, fallback to True if none exist)
            passes_linter = self._run_linter(shadow.shadow_root, target_relative_path)
            
            # 4. Check Tests (run pytest in the shadow root)
            passes_tests = self._run_tests(shadow.shadow_root)
            
            # Compute score
            reward = 0.0
            if compiles:
                reward += 0.4
            if passes_linter:
                reward += 0.3
            if passes_tests:
                reward += 0.5
                
            # Diff size penalty to encourage minimal edits
            penalty = candidate.diff_size * 0.0001
            reward -= penalty
            
            return StructuredReward(
                compiles=compiles,
                passes_linter=passes_linter,
                passes_tests=passes_tests,
                diff_size_penalty=penalty,
                total_reward=max(-1.0, reward),
            )
            
        finally:
            # Fast revert: write back the original string instead of deleting the whole directory
            if file_existed:
                target_path.write_text(original_content, encoding="utf-8")
            else:
                target_path.unlink(missing_ok=True)

    def _run_linter(self, workspace: Path, target_file: str) -> bool:
        cmd = [sys.executable, "-m", "ruff", "check", target_file]
        try:
            res = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=10)
            # When ruff is not installed, Python runs fine but prints
            # "No module named ruff" to stderr and exits with code 1.
            if res.returncode != 0 and _MODULE_NOT_FOUND in res.stderr:
                return True
            return res.returncode == 0
        except (FileNotFoundError, OSError):
            # Ruff binary not found or blocked by OS policy (e.g. Windows AppControl)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _run_tests(self, workspace: Path) -> bool:
        # Ensure the workspace dir is on PYTHONPATH so bare imports
        # (e.g. "from math_utils import add") resolve on all platforms.
        env = {**os.environ, "PYTHONPATH": str(workspace)}
        cmd = [
            sys.executable, "-m", "pytest", "-q", "--no-header",
            "--rootdir", str(workspace), str(workspace),
        ]
        try:
            res = subprocess.run(
                cmd, cwd=workspace, capture_output=True,
                text=True, timeout=30, env=env,
            )
            # If pytest module is somehow missing, treat as "no test runner"
            if res.returncode != 0 and _MODULE_NOT_FOUND in res.stderr:
                return True
            return res.returncode == 0
        except (FileNotFoundError, OSError):
            # pytest not found or blocked by OS policy (e.g. Windows AppControl)
            return True
        except subprocess.TimeoutExpired:
            return False
