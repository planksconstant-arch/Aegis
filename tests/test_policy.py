"""
Tests for ActorCriticPolicy — forward pass, action selection, gradient updates.
"""
from __future__ import annotations

import numpy as np
import pytest

from local_ide_agent.rl.actions import ACTION_SPACE
from local_ide_agent.rl.policy import ActorCriticPolicy, QNetwork
from local_ide_agent.rl.state import StateEncoderStack


class TestQNetwork:
    def test_forward_returns_scalar(self, hp):
        trunk_out = (list(hp.trunk_hidden_sizes) or [128])[-1]
        q = QNetwork(trunk_out, len(ACTION_SPACE), hidden=16, name="test_q")
        trunk = np.random.randn(trunk_out)
        oh = np.zeros(len(ACTION_SPACE)); oh[0] = 1.0
        result = q.forward(trunk, oh)
        assert isinstance(result, float)

    def test_backward_produces_gradients(self, hp):
        trunk_out = (list(hp.trunk_hidden_sizes) or [128])[-1]
        from local_ide_agent.rl.nn import AdamOptimizer
        q = QNetwork(trunk_out, len(ACTION_SPACE), hidden=16, name="test_q")
        opt1, opt2 = AdamOptimizer(lr=1e-3), AdamOptimizer(lr=1e-3)
        trunk = np.random.randn(trunk_out)
        oh = np.zeros(len(ACTION_SPACE)); oh[0] = 1.0
        q.forward(trunk, oh)
        q.backward_and_update(0.5, opt1, opt2)  # should not raise


class TestActorCriticPolicy:
    def test_post_init_creates_networks(self, policy):
        assert policy.trunk is not None
        assert policy.actor_head is not None
        assert policy.q1 is not None
        assert policy.q2 is not None
        assert policy.q1_target is not None
        assert policy.q2_target is not None

    def test_target_weights_equal_online_on_init(self, policy):
        """Target nets must start as exact copies of online nets."""
        np.testing.assert_array_equal(policy.q1.l1.W, policy.q1_target.l1.W)
        np.testing.assert_array_equal(policy.q2.l1.W, policy.q2_target.l1.W)

    def test_decide_returns_decision(self, policy, sample_obs):
        decision = policy.decide(sample_obs)
        assert decision is not None
        assert decision.action is not None
        assert decision.action.action_type is not None

    def test_epsilon_decreases_over_steps(self, policy):
        eps_start = policy.epsilon
        for _ in range(50):
            policy._step_count += 1
        eps_after = policy.epsilon
        assert eps_after <= eps_start

    def test_update_step_returns_td_error(self, policy, sample_obs):
        policy.decide(sample_obs)  # populate last_trunk_output and last_probs
        state_dim = 576
        next_sv = np.random.randn(state_dim).tolist()
        td = policy.update_step(reward=0.5, next_state_vector=next_sv, done=False)
        assert isinstance(td, float)
        assert td >= 0.0

    def test_soft_update_moves_targets(self, policy, sample_obs):
        """After 10 gradient steps, target weights should differ slightly from online."""
        w_before = policy.q1_target.l1.W.copy()
        # Force a weight change in q1 online
        policy.q1.l1.W += 1.0
        policy._soft_update_targets()
        w_after = policy.q1_target.l1.W
        # Targets should have moved towards online (tau=0.005 * 1.0 change = 0.005 shift)
        assert not np.allclose(w_before, w_after)

    def test_action_space_size(self, policy, sample_obs):
        """Policy must produce a valid action index within ACTION_SPACE bounds."""
        policy.decide(sample_obs)
        assert 0 <= policy.last_action_index < len(ACTION_SPACE)

    def test_reward_history_accumulates(self, policy, sample_obs):
        policy.decide(sample_obs)
        sv = np.random.randn(576).tolist()
        for r in [0.1, 0.5, -0.2, 0.8]:
            policy.update_step(reward=r, next_state_vector=sv, done=False)
        assert len(policy.reward_history) >= 4
