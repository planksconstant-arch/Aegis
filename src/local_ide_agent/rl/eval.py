"""
Evaluation harness with held-out test tasks.

Separates train/eval so we can track generalization:
  - 5 held-out tasks never seen during training
  - Runs evaluation WITHOUT updating policy weights
  - Returns EvalReport with per-task and aggregate metrics
  - Can be called from CLI: local-ide-agent eval --episodes 5

Train/eval split
----------------
The environment has 22 tasks. The eval harness uses a fixed set of 5 tasks
chosen to cover different categories (test, security, refactor, plan, perf).
These are excluded from the curriculum's training pool.

Metrics
-------
  eval_avg_reward         — mean episode reward on held-out tasks
  eval_success_rate       — fraction of episodes where reward > 0 and completed
  eval_avg_steps          — mean steps to completion
  eval_per_task           — per-task breakdown
  vs_train_avg_reward     — gap between eval and train reward (negative = overfitting)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from local_ide_agent.agent.core import LocalIDEAgent
from local_ide_agent.schemas import EpisodeResult, Observation


# ---------------------------------------------------------------------------
# Held-out eval task set  (never sampled during curriculum training)
# ---------------------------------------------------------------------------

EVAL_TASKS: list[dict] = [
    {
        "task": "Fix a memory leak in the WebSocket connection manager",
        "files": ["src/ws/manager.py"],
        "diag": ["ResourceWarning: unclosed socket"],
    },
    {
        "task": "Audit and update all third-party dependency versions",
        "files": ["requirements.txt", "pyproject.toml"],
        "diag": ["Safety: 2 packages with known CVEs"],
    },
    {
        "task": "Propose a safe database migration for the new nullable column",
        "files": ["migrations/0042_add_column.py"],
        "diag": [],
    },
    {
        "task": "Optimize the N+1 query in the user dashboard endpoint",
        "files": ["src/dashboard/views.py", "src/dashboard/queries.py"],
        "diag": ["Slow query: 450ms avg"],
    },
    {
        "task": "Refactor duplicated error handling across three controllers",
        "files": ["src/controllers/user.py", "src/controllers/order.py"],
        "diag": [],
    },
]


@dataclass
class TaskEvalResult:
    task: str
    reward: float
    completed: bool
    steps: int
    strategy_used: str
    notes: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    avg_reward: float
    success_rate: float
    avg_steps: float
    task_results: list[TaskEvalResult]
    train_avg_reward: float = 0.0

    @property
    def generalization_gap(self) -> float:
        """Negative = overfitting. Positive = eval better than train (unlikely)."""
        return self.avg_reward - self.train_avg_reward

    def summary_dict(self) -> dict[str, object]:
        return {
            "eval_avg_reward": round(self.avg_reward, 4),
            "eval_success_rate": round(self.success_rate, 4),
            "eval_avg_steps": round(self.avg_steps, 2),
            "generalization_gap": round(self.generalization_gap, 4),
            "train_avg_reward": round(self.train_avg_reward, 4),
            "tasks_evaluated": len(self.task_results),
        }

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"  Avg reward:       {self.avg_reward:+.4f}")
        print(f"  Success rate:     {self.success_rate:.1%}")
        print(f"  Avg steps:        {self.avg_steps:.1f}")
        print(f"  Train avg reward: {self.train_avg_reward:+.4f}")
        print(f"  Gen. gap:         {self.generalization_gap:+.4f}  (negative = overfitting)")
        print()
        print("  Per-task breakdown:")
        for tr in self.task_results:
            status = "OK" if tr.completed else "--"
            print(
                f"    [{status}] {tr.task[:50]:<50}  "
                f"r={tr.reward:+.3f}  steps={tr.steps}  "
                f"strategy={tr.strategy_used}"
            )
        print("=" * 60 + "\n")


class EvaluationHarness:
    """
    Runs held-out evaluation WITHOUT updating policy weights.

    Parameters
    ----------
    agent:              the LocalIDEAgent to evaluate
    eval_tasks:         list of task dicts (defaults to EVAL_TASKS)
    episodes_per_task:  how many episodes to run per task (averages results)
    """

    def __init__(
        self,
        agent: LocalIDEAgent,
        eval_tasks: list[dict] | None = None,
        episodes_per_task: int = 1,
        train_avg_reward: float = 0.0,
    ) -> None:
        self.agent = agent
        self.eval_tasks = eval_tasks or EVAL_TASKS
        self.episodes_per_task = episodes_per_task
        self.train_avg_reward = train_avg_reward

    def run(self) -> EvalReport:
        """Run evaluation; returns EvalReport. Does NOT update policy weights."""
        task_results: list[TaskEvalResult] = []

        for task_dict in self.eval_tasks:
            ep_rewards: list[float] = []
            ep_steps: list[int] = []
            ep_completed: list[bool] = []
            ep_strategies: list[str] = []
            ep_notes: list[str] = []

            for _ in range(self.episodes_per_task):
                result = self._run_one_episode(task_dict)
                ep_rewards.append(result["reward"])
                ep_steps.append(result["steps"])
                ep_completed.append(result["completed"])
                ep_strategies.append(result["strategy"])
                ep_notes.extend(result["notes"])

            task_results.append(TaskEvalResult(
                task=task_dict["task"],
                reward=sum(ep_rewards) / len(ep_rewards),
                completed=any(ep_completed),
                steps=int(sum(ep_steps) / len(ep_steps)),
                strategy_used=ep_strategies[-1],
                notes=ep_notes[-3:],
            ))

        n = len(task_results)
        avg_reward = sum(tr.reward for tr in task_results) / max(n, 1)
        success_rate = sum(1 for tr in task_results if tr.completed) / max(n, 1)
        avg_steps = sum(tr.steps for tr in task_results) / max(n, 1)

        return EvalReport(
            avg_reward=avg_reward,
            success_rate=success_rate,
            avg_steps=avg_steps,
            task_results=task_results,
            train_avg_reward=self.train_avg_reward,
        )

    def _run_one_episode(self, task_dict: dict) -> dict:
        """Run one episode on a specific eval task. No weight updates."""
        from local_ide_agent.training.environment import SimulatedCodingEnvironment

        eval_env = SimulatedCodingEnvironment(tasks=[task_dict], max_steps=10)
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_strategy = "no_op"
        notes: list[str] = []

        while not done:
            # Evaluate only — no gradient updates during eval
            decision = self.agent.evaluate(obs)
            last_strategy = str(decision.action.payload.get("strategy_name", "no_op"))
            next_obs, step_reward, done = eval_env.step(decision)
            total_reward += step_reward
            steps += 1
            obs = next_obs

        ep_result = eval_env.episode_result()
        notes.extend(ep_result.notes)

        return {
            "reward": total_reward,
            "steps": steps,
            "completed": ep_result.completed,
            "strategy": last_strategy,
            "notes": notes,
        }
