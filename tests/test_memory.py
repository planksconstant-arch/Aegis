"""
Tests for MemoryStore — feedback recording, action success rates, snapshot.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_ide_agent.memory.store import FeedbackRecord, MemoryStore


@pytest.fixture()
def store(tmp_path):
    """MemoryStore backed by a temp SQLite file, cleaned up after each test."""
    db = tmp_path / "test_agent.db"
    return MemoryStore(database_path=db)


@pytest.fixture()
def feedback_record():
    return FeedbackRecord(
        user_id="test_user",
        client_id="test_client",
        action_type="minimal_patch",
        decision_description="Apply a small targeted fix",
        accepted=True,
        reward=0.6,
        style_updates={"indent": "4"},
    )


class TestMemoryStore:
    def test_record_feedback_does_not_raise(self, store, feedback_record):
        store.record_feedback(feedback_record)  # should not raise

    def test_action_success_rates_updates_on_feedback(self, store):
        for i in range(5):
            store.record_feedback(FeedbackRecord(
                user_id="u1",
                client_id="test_client",
                action_type="add_tests",
                decision_description="Generate tests",
                accepted=(i % 2 == 0),
                reward=0.4 if i % 2 == 0 else -0.1,
                style_updates={},
            ))
        rates = store.get_action_success_rates("u1", min_count=1)
        assert len(rates) == 1
        assert rates[0]["action_type"] == "add_tests"
        assert rates[0]["total_count"] == 5
        assert rates[0]["accept_count"] == 3   # i=0,2,4

    def test_action_success_rates_multiple_actions(self, store):
        for action, reward, accepted in [
            ("minimal_patch", 0.8, True),
            ("minimal_patch", 0.6, True),
            ("no_op", -0.2, False),
        ]:
            store.record_feedback(FeedbackRecord(
                user_id="u2",
                client_id="test_client",
                action_type=action,
                decision_description="",
                accepted=accepted,
                reward=reward,
                style_updates={},
            ))
        rates = store.get_action_success_rates("u2", min_count=1)
        action_names = [r["action_type"] for r in rates]
        assert "minimal_patch" in action_names
        assert "no_op" in action_names
        # minimal_patch should rank higher (positive avg_reward)
        assert rates[0]["action_type"] == "minimal_patch"

    def test_min_count_filter(self, store):
        store.record_feedback(FeedbackRecord(
            user_id="u3",
            client_id="test_client",
            action_type="rare_action",
            decision_description="",
            accepted=True,
            reward=0.5,
            style_updates={},
        ))
        # With min_count=5, action with only 1 record should be filtered out
        rates = store.get_action_success_rates("u3", min_count=5)
        assert all(r["action_type"] != "rare_action" for r in rates)

    def test_snapshot_returns_object(self, store, feedback_record):
        store.record_feedback(feedback_record)
        snap = store.snapshot("test_user")
        assert snap is not None
        assert hasattr(snap, "preferred_actions")
        assert hasattr(snap, "recent_tasks")

    def test_user_isolation(self, store):
        store.record_feedback(FeedbackRecord(
            user_id="alice", client_id="test_client", action_type="add_tests",
            decision_description="", accepted=True, reward=0.5, style_updates={},
        ))
        store.record_feedback(FeedbackRecord(
            user_id="bob", client_id="test_client", action_type="no_op",
            decision_description="", accepted=False, reward=-0.2, style_updates={},
        ))
        alice_rates = store.get_action_success_rates("alice", min_count=1)
        bob_rates   = store.get_action_success_rates("bob", min_count=1)
        assert all(r["action_type"] != "no_op" for r in alice_rates)
        assert all(r["action_type"] != "add_tests" for r in bob_rates)
