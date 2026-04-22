"""
Full training loop with:
  - N-step return computation for faster credit assignment
  - RND curiosity intrinsic reward for sparse-reward settings
  - Reward normalisation (Welford running mean/std) — keeps TD targets bounded
  - Target Q-network soft updates via policy._soft_update_targets()
  - Curriculum learning via CurriculumScheduler
  - Trajectory buffer enriching observations with episode context
  - Proper s -> a -> r -> s' with real next-state encoding
  - Per-episode metrics printed for observability
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from local_ide_agent.agent.core import LocalIDEAgent
from local_ide_agent.agent.trajectory_buffer import TrajectoryBuffer
from local_ide_agent.rl.curiosity import RNDModule
from local_ide_agent.rl.n_step import NStepReturnBuffer
from local_ide_agent.rl.replay import PrioritizedReplayBuffer, ReplayTransition
from local_ide_agent.rl.trainer import ReplayTrainer, TrainingMetrics
from local_ide_agent.schemas import EpisodeResult
from local_ide_agent.training.curriculum import CurriculumScheduler
from local_ide_agent.training.environment import SimulatedCodingEnvironment


class RewardNormaliser:
    """
    Online Welford running mean/std normalisation for reward signals.

    Keeps combined (env + intrinsic) rewards on a stable scale regardless
    of how fast the RND predictor converges. Without this, a normalised
    intrinsic reward of ~100 would completely dominate the env reward ~0.5,
    producing TD targets far outside the critic's representable range.

    r_norm = clip((r - mean) / (std + eps), -clip_range, clip_range)
    """

    def __init__(self, clip_range: float = 10.0, warmup: int = 32) -> None:
        self.clip_range = clip_range
        self.warmup = warmup
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0   # sum of squared deviations

    def update_and_normalise(self, r: float) -> float:
        self._count += 1
        delta = r - self._mean
        self._mean += delta / self._count
        delta2 = r - self._mean
        self._m2 += delta * delta2

        if self._count < self.warmup:
            return r  # return raw reward during warmup

        var = self._m2 / max(self._count - 1, 1)
        std = math.sqrt(var) + 1e-8
        return float(max(-self.clip_range, min(self.clip_range, (r - self._mean) / std)))

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return math.sqrt(self._m2 / max(self._count - 1, 1)) + 1e-8


@dataclass
class TrainingLoop:
    agent: LocalIDEAgent
    environment: SimulatedCodingEnvironment
    replay_trainer: ReplayTrainer

    # Optional advanced components (created automatically if not supplied)
    curiosity: RNDModule | None = None
    n_step_buffer: NStepReturnBuffer | None = None
    curriculum: CurriculumScheduler | None = None
    trajectory_buffer: TrajectoryBuffer | None = None

    # Reward normaliser — keeps combined rewards on a stable scale
    reward_normaliser: RewardNormaliser = field(default_factory=RewardNormaliser)

    # Accumulate training metrics for status reporting
    _recent_rewards: list[float] = field(default_factory=list)
    _total_episodes: int = 0

    def __post_init__(self) -> None:
        # Auto-create components
        if self.curiosity is None:
            self.curiosity = RNDModule(
                state_dim=576,
                embed_dim=64,
                beta=0.05,
                weight_path=".agent/rnd_weights.npz",
            )
        if self.n_step_buffer is None:
            self.n_step_buffer = NStepReturnBuffer(n=5, gamma=0.99)
        if self.trajectory_buffer is None:
            self.trajectory_buffer = TrajectoryBuffer(window=10)
        if self.curriculum is None:
            self.curriculum = CurriculumScheduler(self.environment)

    def run(self, episodes: int) -> tuple[list[EpisodeResult], TrainingMetrics]:
        results: list[EpisodeResult] = []
        last_metrics = TrainingMetrics(0.0, 0.0, 0.0, 0.0, 0)

        for ep in range(episodes):
            # ---------------------------------------------------------------
            # 1.  Curriculum-controlled episode reset
            # ---------------------------------------------------------------
            obs = self.environment.reset()
            self.n_step_buffer.reset()
            self.trajectory_buffer.reset()
            done = False

            while not done:
                # 2a. Enrich observation with trajectory context
                enriched_obs = self.trajectory_buffer.enrich_observation(obs)

                # 2b. Evaluate policy
                decision = self.agent.evaluate(enriched_obs)
                _ = self.agent.execute(decision)

                # 2c. Step environment → get real next obs
                if hasattr(self.environment, "step"):
                    next_obs, step_reward, done = self.environment.step(decision)
                else:
                    ep_result = self.environment.reward(decision, "")
                    next_obs = obs
                    step_reward = ep_result.reward
                    done = True

                # 2d. Add intrinsic curiosity reward
                if self.agent.replay_buffer is not None and hasattr(self.agent.policy, "last_fused_state"):
                    state_vector = list(
                        getattr(self.agent.policy.last_fused_state, "state_vector", [])
                    )
                    action_index = int(getattr(self.agent.policy, "last_action_index", 0))

                    # Curiosity: compute intrinsic reward AND update predictor atomically
                    r_int = self.curiosity.update_and_get_reward(state_vector)
                    raw_combined = step_reward + r_int

                    # Normalise the combined reward to keep TD targets bounded
                    combined_reward = self.reward_normaliser.update_and_normalise(raw_combined)

                    # Encode next state
                    if hasattr(self.agent.policy, "encoder_stack"):
                        next_fused = self.agent.policy.encoder_stack.encode(next_obs)
                        next_state_vector = list(next_fused.state_vector)
                    else:
                        next_state_vector = state_vector

                    td_est = abs(combined_reward - decision.confidence)

                    # 2e. Feed through n-step buffer
                    self.n_step_buffer.add_step(
                        state_vector=state_vector,
                        action_index=action_index,
                        reward=combined_reward,
                        next_state_vector=next_state_vector,
                        done=done,
                        context={
                            "task": obs.task,
                            "autonomy_tier": decision.autonomy_tier,
                            "step": getattr(self.environment, "_step", 0),
                            "r_int": round(r_int, 5),
                        },
                        td_error=td_est,
                    )

                    # 2f. Flush completed n-step transitions into replay buffer
                    for transition in self.n_step_buffer.flush():
                        self.agent.replay_buffer.add(transition)
                        if self.agent.memory_store:
                            self.agent.memory_store.record_replay_transition(
                                user_id="default",
                                action_index=transition.action_index,
                                reward=transition.reward,
                                done=transition.done,
                                td_error=transition.td_error,
                                state_vector=transition.state_vector,
                                next_state_vector=transition.next_state_vector,
                                context=transition.context,
                            )

                # 2g. Record step in trajectory buffer
                self.trajectory_buffer.record(obs, decision, step_reward)
                obs = next_obs

            # ---------------------------------------------------------------
            # 3.  Collect episode result + curriculum update
            # ---------------------------------------------------------------
            ep_result = (
                self.environment.episode_result()
                if hasattr(self.environment, "episode_result")
                else EpisodeResult(reward=step_reward, decision_count=1, completed=done)
            )
            results.append(ep_result)
            self._recent_rewards.append(ep_result.reward)
            if len(self._recent_rewards) > 100:
                self._recent_rewards = self._recent_rewards[-100:]
            self._total_episodes += 1

            # Curriculum promotion/demotion
            self.curriculum.record_outcome(ep_result.reward, ep_result.completed)

            # ---------------------------------------------------------------
            # 4.  Train on replay buffer
            # ---------------------------------------------------------------
            buf_size = len(self.agent.replay_buffer) if self.agent.replay_buffer else 0
            if self.agent.replay_buffer and buf_size >= self.replay_trainer.batch_size:
                policy = self.agent.policy if hasattr(self.agent.policy, "update_step") else None
                last_metrics = self.replay_trainer.train_step(
                    self.agent.replay_buffer,
                    policy=policy,
                )
            else:
                self.agent.policy.update(ep_result.reward)

            # ---------------------------------------------------------------
            # 5.  Per-episode logging
            # ---------------------------------------------------------------
            curric_stats = self.curriculum.stats()
            rnd_stats = self.curiosity.stats()
            r_std = round(self.reward_normaliser.std, 3)
            print(
                f"  ep={ep + 1:3d}/{episodes}  "
                f"reward={ep_result.reward:+.4f}  "
                f"steps={ep_result.decision_count}  "
                f"done={ep_result.completed}  "
                f"diff={curric_stats['difficulty']}  "
                f"success={curric_stats['rolling_success_rate']:.2f}  "
                f"r_std={r_std}  "
                f"r_int={rnd_stats['rnd_reward_mean']:.3f}  "
                f"actor={last_metrics.actor_loss:.4f}  "
                f"critic={last_metrics.critic_loss:.4f}  "
                f"buf={buf_size}"
            )

        return results, last_metrics
