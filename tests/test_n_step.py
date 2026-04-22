"""
Tests for NStepReturnBuffer and compute_n_step_returns.
"""
from __future__ import annotations

import numpy as np
import pytest

from local_ide_agent.rl.n_step import NStepReturnBuffer, compute_n_step_returns


def _make_step(i: int, done: bool = False):
    return {
        "state_vector": [float(i)] * 8,
        "action_index": i % 6,
        "reward": float(i) * 0.1,
        "next_state_vector": [float(i + 1)] * 8,
        "done": done,
    }


class TestNStepReturnBuffer:
    def test_empty_buffer_flushes_nothing(self):
        buf = NStepReturnBuffer(n=5, gamma=0.99)
        assert buf.flush() == []

    def test_flush_on_done_produces_transitions(self):
        buf = NStepReturnBuffer(n=5, gamma=0.99)
        for i in range(5):
            buf.add_step(**_make_step(i, done=(i == 4)))
        transitions = buf.flush()
        assert len(transitions) > 0

    def test_n_step_reward_is_discounted_sum(self):
        """With n=3 and rewards [0.1, 0.2, 0.3], the n-step reward for step 0 is:
        0.1 + γ*0.2 + γ²*0.3"""
        gamma = 0.99
        buf = NStepReturnBuffer(n=3, gamma=gamma)
        rewards = [0.1, 0.2, 0.3]
        for i, r in enumerate(rewards):
            buf.add_step(
                state_vector=[float(i)] * 8,
                action_index=0,
                reward=r,
                next_state_vector=[float(i + 1)] * 8,
                done=(i == 2),
            )
        transitions = buf.flush()
        assert len(transitions) > 0
        # First transition should have reward = 0.1 + 0.99*0.2 + 0.99²*0.3
        expected = 0.1 + gamma * 0.2 + gamma ** 2 * 0.3
        assert abs(transitions[0].reward - expected) < 1e-5

    def test_reset_clears_buffer(self):
        buf = NStepReturnBuffer(n=5, gamma=0.99)
        buf.add_step(**_make_step(0))
        buf.reset()
        assert buf.flush() == []

    def test_transitions_have_required_fields(self):
        buf = NStepReturnBuffer(n=3, gamma=0.99)
        for i in range(4):
            buf.add_step(**_make_step(i, done=(i == 3)))
        for t in buf.flush():
            assert hasattr(t, "state_vector")
            assert hasattr(t, "action_index")
            assert hasattr(t, "reward")
            assert hasattr(t, "next_state_vector")
            assert hasattr(t, "done")


class TestComputeNStepReturns:
    def test_single_step_with_done(self):
        returns = compute_n_step_returns(
            rewards=[1.0],
            dones=[True],
            next_values=[0.5],
            n=3,
            gamma=0.99,
        )
        assert len(returns) == 1
        # Done=True, so bootstrap = 0; return = 1.0
        assert abs(returns[0] - 1.0) < 1e-6

    def test_multistep_without_done(self):
        returns = compute_n_step_returns(
            rewards=[0.1, 0.1, 0.1],
            dones=[False, False, False],
            next_values=[1.0, 1.0, 1.0],
            n=2,
            gamma=0.99,
        )
        assert len(returns) == 3

    def test_returns_are_finite(self):
        rewards = [0.5 * ((-1) ** i) for i in range(20)]
        dones = [False] * 19 + [True]
        next_vals = [0.0] * 20
        returns = compute_n_step_returns(rewards, dones, next_vals, n=5, gamma=0.99)
        assert all(np.isfinite(r) for r in returns)
