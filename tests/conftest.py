"""
Shared pytest fixtures available to all test modules.
"""
from __future__ import annotations

import pytest

from local_ide_agent.config import AppSettings, RLHyperparams
from local_ide_agent.rl.policy import ActorCriticPolicy
from local_ide_agent.rl.state import StateEncoderStack


@pytest.fixture()
def hp() -> RLHyperparams:
    """Minimal hyperparams for fast tests."""
    return RLHyperparams(
        trunk_hidden_sizes=[64, 32],
        critic_hidden_sizes=[16],
        replay_capacity=128,
        batch_size=4,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100,
        weight_path="",          # disable disk I/O in tests
        curiosity_weight_path="",
        n_step=3,
    )


@pytest.fixture()
def policy(hp) -> ActorCriticPolicy:
    return ActorCriticPolicy(
        encoder_stack=StateEncoderStack(),
        hp=hp,
    )


@pytest.fixture()
def sample_obs():
    from local_ide_agent.schemas import Observation
    return Observation(
        task="Fix the failing auth test",
        open_files=["src/auth.py", "tests/test_auth.py"],
        diagnostics=["AssertionError: expected 200, got 401"],
        user_present=True,
        metadata={"pressure_level": "normal"},
    )
