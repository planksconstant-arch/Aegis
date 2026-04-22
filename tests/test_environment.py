"""
Tests for the training environment and reward normaliser.
"""
from __future__ import annotations

import pytest

from local_ide_agent.training.environment import SimulatedCodingEnvironment
from local_ide_agent.training.loop import RewardNormaliser


@pytest.fixture()
def env():
    return SimulatedCodingEnvironment()


@pytest.fixture()
def dummy_decision(policy, sample_obs):
    return policy.decide(sample_obs)


class TestSimulatedCodingEnvironment:
    def test_reset_returns_observation(self, env):
        from local_ide_agent.schemas import Observation
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.task

    def test_step_returns_triple(self, env, policy, sample_obs):
        env.reset()
        decision = policy.decide(sample_obs)
        result = env.step(decision)
        assert len(result) == 3
        obs, reward, done = result
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_reward_clamped(self, env, policy, sample_obs):
        env.reset()
        for _ in range(10):
            decision = policy.decide(sample_obs)
            _, reward, done = env.step(decision)
            assert -1.0 <= reward <= 1.0
            if done:
                break

    def test_step_budget_ends_episode(self, env, policy, sample_obs):
        env.reset()
        done = False
        steps = 0
        while not done:
            decision = policy.decide(sample_obs)
            _, _, done = env.step(decision)
            steps += 1
            if steps > env.max_steps + 5:
                pytest.fail("Episode did not end after max_steps")

    def test_episode_result_after_step(self, env, policy, sample_obs):
        env.reset()
        decision = policy.decide(sample_obs)
        env.step(decision)
        result = env.episode_result()
        assert result is not None
        assert isinstance(result.reward, float)

    def test_configure_workspace_missing_path(self, env):
        """configure_workspace with a non-existent path should not crash."""
        env.configure_workspace("/nonexistent/path/xyz")
        obs = env.reset()
        # Falls back to simulated files
        assert obs.open_files

    def test_real_context_flag_false_without_workspace(self, env):
        obs = env.reset()
        # Without a workspace configured, real_context is False or absent
        assert not obs.metadata.get("real_context", False)


class TestRewardNormaliser:
    def test_returns_raw_during_warmup(self):
        norm = RewardNormaliser(warmup=10, clip_range=5.0)
        r = norm.update_and_normalise(3.0)
        # During warmup, should return raw value
        assert r == 3.0

    def test_normalises_after_warmup(self):
        norm = RewardNormaliser(warmup=5, clip_range=5.0)
        for _ in range(5):
            norm.update_and_normalise(0.5)
        # After warmup, reward should be normalised
        r = norm.update_and_normalise(0.5)
        assert abs(r) <= 5.0

    def test_clips_extreme_values(self):
        norm = RewardNormaliser(warmup=5, clip_range=3.0)
        for _ in range(5):
            norm.update_and_normalise(0.5)
        r = norm.update_and_normalise(1_000_000.0)
        assert r <= 3.0

    def test_std_increases_with_variance(self):
        norm = RewardNormaliser(warmup=1)
        for v in [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
            norm.update_and_normalise(v)
        assert norm.std > 0.1

    def test_stable_over_many_steps(self):
        import math
        norm = RewardNormaliser(warmup=10)
        for i in range(200):
            r = norm.update_and_normalise(float(i % 5) * 0.3 - 0.5)
            assert math.isfinite(r)
