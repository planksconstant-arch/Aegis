from __future__ import annotations

import argparse
from pathlib import Path

from local_ide_agent.agent.core import LocalIDEAgent
from local_ide_agent.agent.trajectory_buffer import TrajectoryBuffer
from local_ide_agent.bridge import BridgeServerConfig, run_bridge_server
from local_ide_agent.cli.dashboard import run_dashboard
from local_ide_agent.config import AppSettings
from local_ide_agent.connectors.ide import LocalIDEConnector
from local_ide_agent.deployment.background import BackgroundDeploymentManager, ShadowBackgroundService, ShadowRunRequest
from local_ide_agent.memory.store import MemoryStore
from local_ide_agent.rl.curiosity import RNDModule
from local_ide_agent.rl.eval import EvaluationHarness
from local_ide_agent.rl.benchmark import BenchmarkHarness
from local_ide_agent.rl.n_step import NStepReturnBuffer
from local_ide_agent.rl.policy import ActorCriticPolicy
from local_ide_agent.rl.replay import PrioritizedReplayBuffer
from local_ide_agent.rl.replay import ReplayTransition
from local_ide_agent.rl.state import StateEncoderStack
from local_ide_agent.rl.trainer import ReplayTrainer
from local_ide_agent.sample_payloads import sample_observation
from local_ide_agent.shadow.workspace import ShadowWorkspaceManager
from local_ide_agent.training.curriculum import CurriculumScheduler
from local_ide_agent.training.dataset import OfflineRLDataset
from local_ide_agent.training.environment import SimulatedCodingEnvironment
from local_ide_agent.training.loop import TrainingLoop
from local_ide_agent.training.research import ResearchRLStack


def build_agent(config_path: str | None = None) -> LocalIDEAgent:
    settings = AppSettings.load(config_path)
    workspace_root = Path(settings.bridge.workspace_root).resolve()
    database_path = workspace_root / settings.memory.database_path
    connector = LocalIDEConnector(
        workspace_root=workspace_root,
        event_log_path=settings.bridge.event_log_path,
    )
    memory_store = MemoryStore(database_path=database_path)
    replay_buffer = PrioritizedReplayBuffer(
        capacity=settings.rl.replay_capacity,
        alpha=settings.rl.per_alpha,
        beta_start=settings.rl.per_beta_start,
        beta_steps=settings.rl.per_beta_steps,
    )
    for item in memory_store.load_replay_transitions("default"):
        replay_buffer.add(
            ReplayTransition(
                state_vector=list(item["state_vector"]),
                action_index=int(item["action_index"]),
                reward=float(item["reward"]),
                next_state_vector=list(item["next_state_vector"]),
                done=bool(item["done"]),
                context=dict(item["context"]),
                td_error=float(item["td_error"]),
            )
        )
    policy = ActorCriticPolicy(
        encoder_stack=StateEncoderStack(settings.encoders),
        hp=settings.rl,
        block_high_risk=settings.autonomy.block_high_risk,
    )
    agent = LocalIDEAgent(
        settings=settings,
        connector=connector,
        policy=policy,
        memory_store=memory_store,
        replay_buffer=replay_buffer,
    )
    agent.shadow_service = ShadowBackgroundService(
        manager=BackgroundDeploymentManager(settings=settings.deployment),
        shadow_manager=ShadowWorkspaceManager(workspace_root=workspace_root, settings=settings.shadow),
        memory_store=memory_store,
    )
    from local_ide_agent.agent.llm import LLMClient
    agent.llm = LLMClient(settings.llm)
    return agent


def run_training(config_path: str | None, episodes: int) -> int:
    agent = build_agent(config_path)
    hp = agent.settings.rl

    curiosity = RNDModule(
        state_dim=576,
        embed_dim=hp.curiosity_embed_dim,
        learning_rate=hp.curiosity_lr,
        beta=hp.curiosity_beta,
        normalize=hp.curiosity_normalize,
        weight_path=hp.curiosity_weight_path,
    ) if hp.curiosity_enabled else None

    n_step = NStepReturnBuffer(n=hp.n_step, gamma=hp.gamma)

    env = SimulatedCodingEnvironment()
    # Bridge sim→real: inject actual workspace files and diagnostics when available
    workspace_root = str(Path(agent.settings.bridge.workspace_root).resolve())
    env.configure_workspace(workspace_root)

    curriculum = CurriculumScheduler(
        environment=env,
        window_size=hp.curriculum_window,
        success_threshold=hp.curriculum_success_threshold,
        demotion_threshold=hp.curriculum_demotion_threshold,
    ) if hp.curriculum_enabled else None

    traj_buf = TrajectoryBuffer(window=hp.trajectory_window)

    loop = TrainingLoop(
        agent=agent,
        environment=env,
        replay_trainer=ReplayTrainer(hp=hp),
        curiosity=curiosity,
        n_step_buffer=n_step,
        curriculum=curriculum,
        trajectory_buffer=traj_buf,
    )
    results, metrics = loop.run(episodes)
    average_reward = sum(item.reward for item in results) / max(len(results), 1)
    print(f"Completed {len(results)} episodes.")
    print(f"Average reward: {average_reward:.4f}")
    print(f"Actor loss:     {metrics.actor_loss:.5f}")
    print(f"Critic loss:    {metrics.critic_loss:.5f}")
    print(f"Entropy bonus:  {metrics.entropy_bonus:.5f}")
    print(f"Avg advantage:  {metrics.avg_advantage:.5f}")
    print(f"Avg TD error:   {metrics.avg_td_error:.5f}")
    print(f"Grad norm:      {metrics.grad_norm:.5f}")
    print(f"PER beta:       {metrics.beta_current:.4f}")
    print(f"Epsilon:        {metrics.epsilon_current:.4f}")
    print(f"Replay samples: {metrics.sampled_transitions}")
    if curriculum:
        print(f"Difficulty:     {curriculum.stats()['difficulty']}")
        print(f"Success rate:   {curriculum.stats()['rolling_success_rate']:.2%}")
    return 0


def run_agent(config_path: str | None) -> int:
    agent = build_agent(config_path)
    observation = sample_observation()
    print(agent.tick(observation))
    return 0


def run_background(config_path: str | None, role: str) -> int:
    settings = AppSettings.load(config_path)
    manager = BackgroundDeploymentManager(settings=settings.deployment)
    print(manager.deploy(role))
    for line in manager.heartbeat():
        print(line)
    return 0


def run_shadow_copy(config_path: str | None, objective: str, user_id: str) -> int:
    agent = build_agent(config_path)
    print(agent.shadow_service.start_run(ShadowRunRequest(user_id=user_id, objective=objective)))
    return 0


def run_counterfactual_lab(config_path: str | None, objective: str, user_id: str) -> int:
    agent = build_agent(config_path)
    print(agent.shadow_service.start_counterfactual_run(ShadowRunRequest(user_id=user_id, objective=objective, label="lab")))
    return 0


def run_research_summary(config_path: str | None) -> int:
    settings = AppSettings.load(config_path)
    stack = ResearchRLStack(settings.research, hp=settings.rl)
    plan = stack.build_plan(OfflineRLDataset())
    print(f"Policy architecture: {plan.policy_architecture}")
    print(f"Value architecture: {plan.value_architecture}")
    print(f"Objective: {plan.objective}")
    for note in plan.notes:
        print(f"- {note}")
    return 0


def run_eval(config_path: str | None, train_avg_reward: float = 0.0) -> int:
    agent = build_agent(config_path)
    harness = EvaluationHarness(agent, train_avg_reward=train_avg_reward)
    report = harness.run()
    report.print_report()
    return 0


def run_benchmark(config_path: str | None, target_dir: str, tasks: str) -> int:
    agent = build_agent(config_path)
    target_path = Path(target_dir).resolve()
    tasks_path = Path(tasks).resolve()
    harness = BenchmarkHarness(agent, target_dir=target_path)
    report = harness.run_benchmark(tasks_path)
    report.print_report()
    return 0


def run_live_dashboard(
    config_path: str | None,
    interval: float = 2.0,
    once: bool = False,
) -> int:
    settings = AppSettings.load(config_path)
    workspace_root = Path(settings.bridge.workspace_root).resolve()
    run_dashboard(
        weight_path=str(workspace_root / settings.rl.weight_path),
        events_path=str(workspace_root / settings.bridge.event_log_path),
        db_path=str(workspace_root / settings.memory.database_path),
        rnd_weight_path=str(workspace_root / settings.rl.curiosity_weight_path),
        refresh_interval=interval,
        once=once,
    )
    return 0


def run_bridge(config_path: str | None) -> int:
    settings = AppSettings.load(config_path)
    agent = build_agent(config_path)
    bridge_config = BridgeServerConfig(
        host=settings.bridge.host,
        port=settings.bridge.port,
    )
    run_bridge_server(agent, bridge_config)
    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="Optional YAML settings file")

    parser = argparse.ArgumentParser(description="Local IDE RL agent scaffold", parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run simulated training", parents=[common])
    train_parser.add_argument("--episodes", type=int, default=10)

    subparsers.add_parser("run", help="Run a single agent tick", parents=[common])

    background_parser = subparsers.add_parser("deploy-background", help="Deploy a background agent", parents=[common])
    background_parser.add_argument("--role", default="code-maintainer")

    shadow_parser = subparsers.add_parser("shadow-run", help="Create a shadow copy and schedule autonomous work", parents=[common])
    shadow_parser.add_argument("--objective", default="Improve the current project safely in a copy")
    shadow_parser.add_argument("--user-id", default="default")

    lab_parser = subparsers.add_parser("counterfactual-lab", help="Spawn multiple shadow strategies and rank them", parents=[common])
    lab_parser.add_argument("--objective", default="Explore multiple safe implementation strategies")
    lab_parser.add_argument("--user-id", default="default")

    subparsers.add_parser("research-plan", help="Print the research RL stack summary", parents=[common])

    subparsers.add_parser("serve-bridge", help="Run the local IDE bridge server", parents=[common])

    eval_parser = subparsers.add_parser("eval-sim", help="Evaluate policy on held-out simulated tasks", parents=[common])
    eval_parser.add_argument("--train-avg-reward", type=float, default=0.0,
                             help="Training avg reward to compute generalization gap")

    bench_parser = subparsers.add_parser("eval", help="Run real-world benchmarks against a target directory", parents=[common])
    bench_parser.add_argument("--target-dir", required=True, help="Target project directory")
    bench_parser.add_argument("--tasks", required=True, help="Path to JSON file containing benchmark tasks")

    dash_parser = subparsers.add_parser("dashboard", help="Live terminal training dashboard", parents=[common])
    dash_parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    dash_parser.add_argument("--once", action="store_true", help="Print once then exit")

    args = parser.parse_args()

    if args.command == "train":
        return run_training(args.config, args.episodes)
    if args.command == "run":
        return run_agent(args.config)
    if args.command == "deploy-background":
        return run_background(args.config, args.role)
    if args.command == "shadow-run":
        return run_shadow_copy(args.config, args.objective, args.user_id)
    if args.command == "counterfactual-lab":
        return run_counterfactual_lab(args.config, args.objective, args.user_id)
    if args.command == "research-plan":
        return run_research_summary(args.config)
    if args.command == "serve-bridge":
        return run_bridge(args.config)
    if args.command == "eval-sim":
        return run_eval(args.config, args.train_avg_reward)
    if args.command == "eval":
        return run_benchmark(args.config, args.target_dir, args.tasks)
    if args.command == "dashboard":
        return run_live_dashboard(args.config, args.interval, args.once)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
