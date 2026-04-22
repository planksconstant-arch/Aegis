"""
Tests for CurriculumScheduler — difficulty promotion, demotion, and stats.
"""
from __future__ import annotations

import pytest

from local_ide_agent.training.curriculum import CurriculumScheduler, Difficulty
from local_ide_agent.training.environment import SimulatedCodingEnvironment


@pytest.fixture()
def env():
    return SimulatedCodingEnvironment()


@pytest.fixture()
def curriculum(env):
    return CurriculumScheduler(
        environment=env,
        window_size=10,
        success_threshold=0.70,
        demotion_threshold=0.35,
    )


class TestCurriculumScheduler:
    def test_starts_at_easy(self, curriculum):
        assert curriculum.difficulty == Difficulty.EASY

    def test_promotes_to_medium_on_consistent_success(self, curriculum):
        for _ in range(5):
            curriculum.record_outcome(reward=1.0, completed=True)
        assert curriculum.difficulty == Difficulty.MEDIUM

    def test_promotes_to_hard_after_two_promotions(self, curriculum):
        for _ in range(5):
            curriculum.record_outcome(reward=1.0, completed=True)
        assert curriculum.difficulty == Difficulty.MEDIUM
        for _ in range(5):
            curriculum.record_outcome(reward=1.0, completed=True)
        assert curriculum.difficulty == Difficulty.HARD

    def test_demotes_on_consistent_failure(self, curriculum):
        # Promote to MEDIUM first
        for _ in range(5):
            curriculum.record_outcome(reward=1.0, completed=True)
        assert curriculum.difficulty == Difficulty.MEDIUM
        # Now fail consistently
        for _ in range(5):
            curriculum.record_outcome(reward=-0.5, completed=False)
        assert curriculum.difficulty == Difficulty.EASY

    def test_stays_at_easy_on_demotion(self, curriculum):
        for _ in range(10):
            curriculum.record_outcome(reward=-0.5, completed=False)
        assert curriculum.difficulty == Difficulty.EASY   # can't go below EASY

    def test_stays_at_hard_on_promotion(self, curriculum):
        # Promote all the way to HARD
        for _ in range(20):
            curriculum.record_outcome(reward=1.0, completed=True)
        assert curriculum.difficulty == Difficulty.HARD
        # More successes should not error
        for _ in range(10):
            curriculum.record_outcome(reward=1.0, completed=True)
        assert curriculum.difficulty == Difficulty.HARD

    def test_stats_contains_required_keys(self, curriculum):
        stats = curriculum.stats()
        assert "difficulty" in stats
        assert "rolling_success_rate" in stats
        assert "promotions" in stats
        assert "demotions" in stats
        assert "episode_count" in stats

    def test_episode_count_increments(self, curriculum):
        for i in range(5):
            curriculum.record_outcome(reward=0.5, completed=True)
        assert curriculum.stats()["episode_count"] == 5

    def test_promotions_counter_increments(self, curriculum):
        for _ in range(10):
            curriculum.record_outcome(reward=1.0, completed=True)
        assert curriculum.stats()["promotions"] >= 1
