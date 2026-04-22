"""
Tests for RNDModule — intrinsic reward computation and predictor learning.
"""
from __future__ import annotations

import numpy as np
import pytest

from local_ide_agent.rl.curiosity import RNDModule


STATE_DIM = 576


@pytest.fixture()
def rnd():
    return RNDModule(
        state_dim=STATE_DIM,
        embed_dim=32,
        learning_rate=1e-3,
        beta=0.1,
        normalize=False,    # disable normalisation for deterministic tests
        weight_path=None,
    )


class TestRNDModule:
    def test_intrinsic_reward_positive(self, rnd):
        sv = np.random.randn(STATE_DIM).tolist()
        r = rnd.intrinsic_reward(sv)
        assert isinstance(r, float)
        assert r >= 0.0

    def test_reward_decreases_with_familiarisation(self, rnd):
        """After 20 updates on the same state, intrinsic reward should drop."""
        sv = np.random.randn(STATE_DIM).tolist()
        r_before = rnd.intrinsic_reward(sv)
        for _ in range(20):
            rnd.update(sv)
        r_after = rnd.intrinsic_reward(sv)
        assert r_after < r_before, (
            f"Intrinsic reward should decrease after learning: {r_before:.4f} -> {r_after:.4f}"
        )

    def test_novel_state_has_higher_reward_than_familiar(self, rnd):
        familiar_sv = np.random.randn(STATE_DIM).tolist()
        novel_sv = np.random.randn(STATE_DIM).tolist()
        # Familiarise with first state
        for _ in range(30):
            rnd.update(familiar_sv)
        r_familiar = rnd.intrinsic_reward(familiar_sv)
        r_novel = rnd.intrinsic_reward(novel_sv)
        assert r_novel > r_familiar

    def test_update_and_get_reward_returns_float(self, rnd):
        sv = np.random.randn(STATE_DIM).tolist()
        r = rnd.update_and_get_reward(sv)
        assert isinstance(r, float)
        assert r >= 0.0

    def test_update_and_get_reward_stable_over_many_steps(self, rnd):
        """Must not crash or produce NaN over 100 sequential steps."""
        sv = np.random.randn(STATE_DIM).tolist()
        for i in range(100):
            r = rnd.update_and_get_reward(sv)
            assert np.isfinite(r), f"NaN/Inf at step {i}"

    def test_stats_keys(self, rnd):
        stats = rnd.stats()
        assert "rnd_updates" in stats
        assert "rnd_reward_mean" in stats
        assert "rnd_beta" in stats

    def test_update_count_increments(self, rnd):
        sv = np.random.randn(STATE_DIM).tolist()
        assert rnd._update_count == 0
        rnd.update_and_get_reward(sv)
        assert rnd._update_count == 1
        rnd.update_and_get_reward(sv)
        assert rnd._update_count == 2

    def test_no_cross_contamination_between_calls(self, rnd):
        """Interleaving intrinsic_reward and update_and_get_reward must not crash."""
        sv1 = np.random.randn(STATE_DIM).tolist()
        sv2 = np.random.randn(STATE_DIM).tolist()
        for _ in range(10):
            rnd.intrinsic_reward(sv1)
            rnd.update_and_get_reward(sv2)   # this was the bug — must not raise
            rnd.intrinsic_reward(sv2)
            rnd.update_and_get_reward(sv1)
