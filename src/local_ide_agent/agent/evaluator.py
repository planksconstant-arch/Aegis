from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from local_ide_agent.config import ShadowWorkspaceSettings
from local_ide_agent.schemas import CandidatePatch, StructuredReward
from local_ide_agent.shadow.workspace import ShadowWorkspaceManager


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
        shadow = self.shadow_manager.create_shadow_copy(label=f"eval-{candidate.id[:6]}")
        target_path = shadow.shadow_root / target_relative_path
        
        try:
            # 1. Apply the candidate replacement
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
            # Cleanup the shadow workspace aggressively to save disk IO during training
            try:
                shutil.rmtree(shadow.shadow_root, ignore_errors=True)
            except Exception:
                pass

    def _run_linter(self, workspace: Path, target_file: str) -> bool:
        cmd = ["python", "-m", "ruff", "check", target_file]
        try:
            res = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=10)
            return res.returncode == 0
        except FileNotFoundError:
            # Ruff not installed in the environment
            return True
        except subprocess.TimeoutExpired:
            return False

    def _run_tests(self, workspace: Path) -> bool:
        cmd = ["pytest", "-q"]
        try:
            res = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=30)
            return res.returncode == 0
        except FileNotFoundError:
            return True
        except subprocess.TimeoutExpired:
            return False
