"""
End-to-end integration test for the Aegis RL pipeline.

Validates that reward signal flows through the entire pipeline:
  environment → policy.decide() → env.step() → replay buffer → trainer → weight update

This test catches regressions in:
  - State encoding / fusion
  - Actor-critic forward and backward passes
  - Replay buffer insertion and sampling
  - Trainer gradient updates
  - Weight persistence
  - Patch critic ranking path
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pytest

from local_ide_agent.config import RLHyperparams
from local_ide_agent.rl.curiosity import RNDModule
from local_ide_agent.rl.n_step import NStepReturnBuffer
from local_ide_agent.rl.policy import ActorCriticPolicy
from local_ide_agent.rl.replay import PrioritizedReplayBuffer
from local_ide_agent.rl.trainer import ReplayTrainer
from local_ide_agent.schemas import Decision, Observation
from local_ide_agent.training.environment import SimulatedCodingEnvironment


# ---------------------------------------------------------------------------
# Mock candidate for patch critic ranking tests
# ---------------------------------------------------------------------------

@dataclass
class MockCandidatePatch:
    diff: str
    description: str = "mock patch"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Full pipeline integration test: 5 episodes through the complete loop."""

    @pytest.fixture
    def hp(self) -> RLHyperparams:
        return RLHyperparams(
            weight_path="",            # no disk persistence during tests
            epsilon_start=0.5,
            epsilon_end=0.1,
            epsilon_decay_steps=50,
            replay_capacity=128,
            batch_size=4,
        )

    @pytest.fixture
    def policy(self, hp: RLHyperparams) -> ActorCriticPolicy:
        return ActorCriticPolicy(hp=hp)

    @pytest.fixture
    def env(self) -> SimulatedCodingEnvironment:
        return SimulatedCodingEnvironment()

    @pytest.fixture
    def replay(self, hp: RLHyperparams) -> PrioritizedReplayBuffer:
        return PrioritizedReplayBuffer(capacity=hp.replay_capacity)

    @pytest.fixture
    def trainer(self, hp: RLHyperparams) -> ReplayTrainer:
        return ReplayTrainer(hp=hp)

    @pytest.fixture
    def n_step(self) -> NStepReturnBuffer:
        return NStepReturnBuffer(n=3, gamma=0.99)

    def test_full_loop_5_episodes(
        self,
        policy: ActorCriticPolicy,
        env: SimulatedCodingEnvironment,
        replay: PrioritizedReplayBuffer,
        trainer: ReplayTrainer,
        n_step: NStepReturnBuffer,
    ) -> None:
        """
        Run 5 episodes through the full RL loop and assert:
        1. Rewards flow end-to-end
        2. Replay buffer fills
        3. Trainer produces non-zero metrics
        4. Weights change after training
        """
        # Snapshot initial weights for comparison
        initial_actor_W = policy.actor_head.W.copy()

        all_rewards: list[float] = []
        episodes_completed = 0

        for _ in range(5):
            obs = env.reset()
            n_step.reset()
            done = False
            episode_reward = 0.0

            while not done:
                decision = policy.decide(obs)
                next_obs, reward, done = env.step(decision)

                # Get state vectors for replay
                state_vec = list(policy.last_fused_state.state_vector)
                next_fused = policy.encoder_stack.encode(next_obs)
                next_state_vec = list(next_fused.state_vector)

                # N-step accumulation
                n_step.add_step(
                    state_vector=state_vec,
                    action_index=policy.last_action_index,
                    reward=reward,
                    next_state_vector=next_state_vec,
                    done=done,
                )

                # Flush completed n-step transitions into replay buffer
                for transition in n_step.flush():
                    replay.add(transition)

                episode_reward += reward
                obs = next_obs

            all_rewards.append(episode_reward)
            episodes_completed += 1

            # Train after each episode (if enough data)
            if len(replay) >= trainer.batch_size:
                metrics = trainer.train_step(replay, policy)
                assert metrics.sampled_transitions > 0, "Trainer should sample transitions"

        # --- Assertions ---
        assert episodes_completed == 5, "All 5 episodes should complete"
        assert len(all_rewards) == 5, "Should have 5 episode rewards"
        assert len(replay) > 0, "Replay buffer should have transitions"

        # Verify weights actually changed (gradient updates happened)
        weights_changed = not np.allclose(initial_actor_W, policy.actor_head.W, atol=1e-10)
        assert weights_changed, "Actor weights should change after training"

        # Verify reward history was recorded
        assert len(policy.reward_history) > 0, "Policy should track reward history"

    def test_patch_critic_ranking(self, policy: ActorCriticPolicy) -> None:
        """
        Test the hybrid LLM-RL candidate ranking path:
        1. Create mock candidates
        2. Rank them via patch_critic
        3. Update patch_critic with reward
        4. Verify weights change
        """
        # Create a fake observation to initialize trunk output
        obs = Observation(
            task="Fix a bug",
            open_files=["test.py"],
            diagnostics=["Error: something broke"],
        )
        _ = policy.decide(obs)

        # Create mock candidates
        candidates = [
            MockCandidatePatch(diff="- old_code()\n+ new_code()"),
            MockCandidatePatch(diff="- broken()\n+ fixed()"),
            MockCandidatePatch(diff="- slow()\n+ fast()"),
        ]

        # Rank candidates
        best, q_val = policy.rank_candidates(candidates)
        assert best is not None, "Should select a best candidate"
        assert isinstance(q_val, float), "Q-value should be a float"

        # Snapshot patch critic weights before update
        l1_W_before = policy.patch_critic.l1.W.copy()

        # Update patch critic with a reward signal
        td_error = policy.update_patch_critic(reward=0.8, selected_candidate=best)
        assert td_error >= 0, "TD error should be non-negative"

        # Verify weights changed
        weights_changed = not np.allclose(l1_W_before, policy.patch_critic.l1.W, atol=1e-12)
        assert weights_changed, "Patch critic weights should change after update"

    def test_rnd_curiosity_integrates_with_policy(self, policy: ActorCriticPolicy) -> None:
        """
        Verify RND curiosity module produces valid intrinsic rewards
        and that its update() method returns non-negative loss.
        """
        rnd = RNDModule(state_dim=576, embed_dim=64, beta=0.1)

        obs = Observation(
            task="Refactor code",
            open_files=["src/main.py"],
            diagnostics=[],
        )
        _ = policy.decide(obs)
        state_vec = policy.last_fused_state.state_vector

        # Intrinsic reward should be non-negative
        r_int = rnd.intrinsic_reward(state_vec)
        assert r_int >= 0.0, "Intrinsic reward should be non-negative"

        # Update should return non-negative loss
        loss = rnd.update(state_vec)
        assert loss >= 0.0, "RND update loss should be non-negative"

        # Combined update+reward should work
        r_combined = rnd.update_and_get_reward(state_vec)
        assert isinstance(r_combined, float), "Combined reward should be float"

    def test_separate_patch_critic_optimizers(self, policy: ActorCriticPolicy) -> None:
        """
        Verify that patch_critic_l1_opt and patch_critic_l2_opt are
        distinct optimizer instances (regression test for the shared
        optimizer bug).
        """
        assert policy.patch_critic_l1_opt is not policy.patch_critic_l2_opt, (
            "Patch critic layers must have separate Adam optimizer instances "
            "to ensure correct per-layer bias correction"
        )

    def test_no_duplicate_methods_in_rnd(self) -> None:
        """
        Regression test: ensure RNDModule has exactly one 'update' method
        (the second was dead code that shadowed the first).
        """
        import inspect
        source = inspect.getsource(RNDModule)
        count = source.count("def update(")
        assert count == 1, f"RNDModule should have exactly 1 update() method, found {count}"
