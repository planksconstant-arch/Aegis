"""
Richer simulated coding environment that exposes a proper MDP interface.

Improvements over the original:
  - 20+ realistic task templates with varying diagnostic/pressure contexts
  - Multi-step episodes via step(action_name) -> (Observation, reward, done)
  - Temporal + pressure randomization per episode reset
  - Shaped reward: correctness bonus + latency bonus + approval-cost penalty
  - Terminal conditions: task solved, budget exhausted, or critical failure
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from local_ide_agent.schemas import Decision, EpisodeResult, Observation

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

_TASKS: list[dict] = [
    {"task": "Fix a failing test in the authentication module", "files": ["src/auth/test_login.py", "src/auth/login.py"], "diag": ["AssertionError: expected 200, got 401"]},
    {"task": "Refactor the database connection helper to use a context manager", "files": ["src/db/connection.py"], "diag": []},
    {"task": "Investigate a lint warning: unused import in payment processor", "files": ["src/payments/processor.py"], "diag": ["W0611 unused import 'datetime'"]},
    {"task": "Draft a safe implementation plan for the new caching layer", "files": ["docs/cache_design.md"], "diag": []},
    {"task": "Add unit tests for the CSV export function", "files": ["src/exports/csv.py", "tests/test_csv.py"], "diag": []},
    {"task": "Resolve a type error in the API response serializer", "files": ["src/api/serializers.py"], "diag": ["TypeError: expected str, got None"]},
    {"task": "Optimize the N+1 query in the user dashboard endpoint", "files": ["src/dashboard/views.py", "src/dashboard/queries.py"], "diag": ["Slow query: 450ms avg"]},
    {"task": "Remove deprecated usage of requests.get without timeout", "files": ["src/integrations/external.py"], "diag": ["DeprecationWarning: timeout not set"]},
    {"task": "Add logging to the background worker for better observability", "files": ["src/workers/background.py"], "diag": []},
    {"task": "Fix a race condition in the task queue consumer", "files": ["src/queue/consumer.py"], "diag": ["RuntimeError: event loop is closed"]},
    {"task": "Migrate the config loader from argparse to pydantic-settings", "files": ["src/config.py"], "diag": []},
    {"task": "Implement retry logic in the HTTP client", "files": ["src/http/client.py"], "diag": ["ConnectionError: max retries exceeded"]},
    {"task": "Reduce startup time by lazy-loading the embedding model", "files": ["src/ml/embedder.py"], "diag": ["Warning: model loaded at import"]},
    {"task": "Harden input validation for the file upload endpoint", "files": ["src/uploads/handler.py"], "diag": ["Security: missing MIME type check"]},
    {"task": "Refactor duplicated error handling across three controllers", "files": ["src/controllers/user.py", "src/controllers/order.py", "src/controllers/product.py"], "diag": []},
    {"task": "Add a smoke test for the deployment pipeline", "files": ["ci/smoke_test.py"], "diag": []},
    {"task": "Investigate test flakiness in the scheduler integration tests", "files": ["tests/integration/test_scheduler.py"], "diag": ["Flaky: 3/10 runs fail"]},
    {"task": "Propose a safe database migration for the new nullable column", "files": ["migrations/0042_add_column.py"], "diag": []},
    {"task": "Fix memory leak in the WebSocket connection manager", "files": ["src/ws/manager.py"], "diag": ["ResourceWarning: unclosed socket"]},
    {"task": "Clean up dead code branches in the feature flag evaluator", "files": ["src/features/evaluator.py"], "diag": []},
    {"task": "Write documentation for the public API surface", "files": ["docs/api.md", "src/api/__init__.py"], "diag": []},
    {"task": "Audit and update all third-party dependency versions", "files": ["requirements.txt", "pyproject.toml"], "diag": ["Safety: 2 packages with known CVEs"]},
]

_PRESSURES = ["low", "normal", "high"]
_PHASES = ["build", "debug", "review", "deploy"]

# ---------------------------------------------------------------------------
# MDP action outcome table
# ---------------------------------------------------------------------------
# Maps (strategy_name, task_category) -> base reward delta
# Categories: test, refactor, lint, plan, security, performance, other

def _task_category(task_text: str) -> str:
    t = task_text.lower()
    if any(w in t for w in ["test", "flaky"]):
        return "test"
    if any(w in t for w in ["refactor", "clean", "dead code", "migrate"]):
        return "refactor"
    if any(w in t for w in ["lint", "import", "warning"]):
        return "lint"
    if any(w in t for w in ["plan", "design", "propose"]):
        return "plan"
    if any(w in t for w in ["security", "harden", "audit", "cve"]):
        return "security"
    if any(w in t for w in ["optim", "n+1", "startup", "memory", "leak"]):
        return "performance"
    return "other"

_REWARD_TABLE: dict[str, dict[str, float]] = {
    "minimal_patch":      {"test": 0.5, "lint": 0.6, "other": 0.3, "refactor": 0.2, "plan": 0.1, "security": 0.3, "performance": 0.2},
    "targeted_refactor":  {"refactor": 0.7, "other": 0.4, "test": 0.3, "lint": 0.3, "plan": 0.2, "security": 0.2, "performance": 0.5},
    "expand_abstraction": {"refactor": 0.5, "plan": 0.6, "other": 0.3, "test": 0.2, "lint": 0.1, "security": 0.2, "performance": 0.3},
    "add_tests":          {"test": 0.8, "other": 0.4, "refactor": 0.3, "lint": 0.2, "plan": 0.2, "security": 0.5, "performance": 0.3},
    "ask_review":         {"plan": 0.5, "security": 0.6, "other": 0.3, "test": 0.2, "refactor": 0.3, "lint": 0.2, "performance": 0.3},
    "silent_shadow":      {"other": 0.2, "test": 0.1, "refactor": 0.2, "lint": 0.1, "plan": 0.3, "security": 0.1, "performance": 0.2},
    "promote_candidate":  {"other": -0.1, "test": -0.1, "refactor": -0.1, "lint": -0.1, "plan": -0.2, "security": -0.3, "performance": -0.1},
    "no_op":              {"other": -0.2, "test": -0.2, "refactor": -0.2, "lint": -0.2, "plan": -0.1, "security": -0.3, "performance": -0.2},
}


@dataclass
class SimulatedCodingEnvironment:
    """
    Multi-step simulated coding MDP.

    Usage
    -----
      obs = env.reset()
      while not done:
          decision = policy.decide(obs)
          obs, reward, done = env.step(decision)
      result = env.episode_result()
    """

    tasks: list[dict] = field(default_factory=lambda: list(_TASKS))
    max_steps: int = 10
    cursor: int = 0

    # Episode state
    _current_task: dict = field(default_factory=dict)
    _step: int = 0
    _cumulative_reward: float = 0.0
    _notes: list[str] = field(default_factory=list)
    _solved: bool = False
    _pressure: str = "normal"
    _phase: str = "build"
    _hour: int = 12
    _user_present: bool = True

    def configure_workspace(self, workspace_root: str | None = None) -> None:
        """
        Optionally point the environment at the real workspace so that reset()
        injects actual open-file names and recent diagnostics from events.jsonl.
        Falls back silently if the workspace doesn't exist.
        """
        self._workspace_root = workspace_root

    def _real_context(self) -> tuple[list[str], list[str]]:
        """
        Read real file names and recent diagnostic strings from the workspace.
        Returns (files, diagnostics). Empty lists = no real context available.
        """
        import json
        from pathlib import Path

        root = getattr(self, "_workspace_root", None)
        if not root:
            return [], []
        root_path = Path(root)
        if not root_path.exists():
            return [], []

        # Collect real source files (limit to 6 most recently modified)
        try:
            src_files = sorted(
                [
                    p for p in root_path.rglob("*.py")
                    if ".agent" not in str(p) and "__pycache__" not in str(p)
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            real_files = [str(p.relative_to(root_path)) for p in src_files[:6]]
        except Exception:
            real_files = []

        # Read recent diagnostics from events.jsonl
        real_diag: list[str] = []
        try:
            events_path = root_path / ".agent" / "events.jsonl"
            if events_path.exists():
                lines = events_path.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines[-50:]):
                    try:
                        record = json.loads(line)
                        payload = record.get("payload", {})
                        stderr = payload.get("stderr", "").strip()
                        if stderr and len(stderr) < 200:
                            real_diag.append(stderr[:120])
                        if len(real_diag) >= 2:
                            break
                    except Exception:
                        continue
        except Exception:
            pass

        return real_files, real_diag

    def reset(self) -> Observation:
        task_dict = self.tasks[self.cursor % len(self.tasks)]
        self.cursor += 1
        self._current_task = task_dict
        self._step = 0
        self._cumulative_reward = 0.0
        self._notes = []
        self._solved = False
        self._pressure = random.choice(_PRESSURES)
        self._phase = random.choice(_PHASES)
        self._hour = random.randint(0, 23)
        self._user_present = random.random() > 0.25  # 75% chance user is present

        # Inject real workspace context (files + diagnostics) if available
        real_files, real_diag = self._real_context()
        open_files = real_files if real_files else task_dict["files"]
        diagnostics = real_diag if real_diag else task_dict["diag"]

        return Observation(
            task=task_dict["task"],
            open_files=open_files,
            diagnostics=diagnostics,
            user_present=self._user_present,
            metadata={
                "pressure_level": self._pressure,
                "session_phase": self._phase,
                "local_hour": self._hour,
                "real_context": bool(real_files),
            },
        )

    def step(self, decision: Decision) -> tuple[Observation, float, bool]:
        """
        Apply an action, compute shaped reward, and return the next observation.

        Returns
        -------
        next_obs: the next observation (may have updated diagnostics)
        reward:   shaped step reward
        done:     True if the episode has ended
        """
        self._step += 1
        strategy_name = str(decision.action.payload.get("strategy_name", "no_op"))
        category = _task_category(self._current_task["task"])

        base_reward = _REWARD_TABLE.get(strategy_name, {}).get(category, 0.1)

        # Pressure bonus: high-pressure tasks reward faster action
        if self._pressure == "high" and strategy_name in {"minimal_patch", "targeted_refactor"}:
            base_reward += 0.15
        elif self._pressure == "high" and strategy_name == "no_op":
            base_reward -= 0.2

        # Diagnostic alignment: if there are diagnostics, patching should help
        remaining_diag = list(self._current_task["diag"])
        if remaining_diag and strategy_name in {"minimal_patch", "add_tests"}:
            base_reward += 0.2
            remaining_diag = []  # Clear diagnostics on fix

        # Approval cost: requiring approval when user is absent is bad
        if decision.requires_approval and not self._user_present:
            base_reward -= 0.15
            self._notes.append("Approval requested while user is absent.")

        # Step penalty (encourage efficiency)
        step_penalty = 0.02 * self._step
        step_reward = base_reward - step_penalty

        # Clamp
        step_reward = max(-1.0, min(1.0, step_reward))
        self._cumulative_reward += step_reward

        # Terminal conditions
        solved = base_reward >= 0.5 and not remaining_diag
        budget_exhausted = self._step >= self.max_steps
        done = solved or budget_exhausted

        if solved:
            self._solved = True
            self._notes.append(f"Task solved in {self._step} steps with '{strategy_name}'.")
        elif budget_exhausted:
            self._notes.append("Episode ended: step budget exhausted.")

        # Build next observation (diagnostics cleared if solved)
        next_obs = Observation(
            task=self._current_task["task"],
            open_files=self._current_task["files"],
            diagnostics=remaining_diag if not solved else [],
            user_present=self._user_present,
            metadata={
                "pressure_level": self._pressure,
                "session_phase": self._phase,
                "local_hour": self._hour,
                "step": self._step,
            },
        )
        return next_obs, step_reward, done

    def episode_result(self) -> EpisodeResult:
        return EpisodeResult(
            reward=round(self._cumulative_reward, 4),
            decision_count=self._step,
            completed=self._solved,
            notes=self._notes,
        )

    # --- Legacy one-shot compatibility (used by old training/loop.py) ---

    def reward(self, decision: Decision) -> EpisodeResult:
        """Legacy single-step reward for backward compatibility."""
        lowered = decision.action.description.lower()
        strategy_name = str(decision.action.payload.get("strategy_name", ""))
        category = _task_category(self._current_task.get("task", ""))

        r = 0.3
        notes: list[str] = []
        if "suggest" in lowered or "fix" in lowered:
            r += 0.3
            notes.append("Aligned with coding assistance behavior.")
        if strategy_name in _REWARD_TABLE:
            r += _REWARD_TABLE[strategy_name].get(category, 0.0)
            notes.append(f"Strategy '{strategy_name}' matched category '{category}'.")
        if decision.requires_approval and not self._user_present:
            r -= 0.15
        r = max(-1.0, min(1.0, r))
        return EpisodeResult(
            reward=round(r, 4),
            decision_count=1,
            completed=r >= 0.5,
            notes=notes,
        )
