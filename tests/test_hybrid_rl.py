from __future__ import annotations

import ast
import uuid
import numpy as np
from pathlib import Path

from local_ide_agent.config import RLHyperparams
from local_ide_agent.rl.policy import ActorCriticPolicy
from local_ide_agent.schemas import CandidatePatch
from local_ide_agent.agent.evaluator import PatchEvaluator


def test_hybrid_rl_candidate_ranking(tmp_path: Path):
    """
    Test that the RL agent can rank candidate patches and learn from structured rewards
    in a simulated 'fix the bug' scenario.
    """
    # 1. Initialize Policy
    hp = RLHyperparams(critic_hidden_sizes=[32], trunk_hidden_sizes=[64], learning_rate=0.01)
    policy = ActorCriticPolicy(hp=hp)

    # 2. Setup a dummy workspace with a buggy Python file
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    buggy_file = workspace / "math_utils.py"
    buggy_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    
    test_file = workspace / "test_math.py"
    test_file.write_text("from math_utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")

    # 3. Simulate an observation -> network trunk state
    # (In a real training loop, we'd use Policy.decide(obs) to populate last_trunk_output)
    policy.last_trunk_output = np.random.randn(64)

    # 4. Mock the LLM generating two candidate patches
    candidates = [
        CandidatePatch(
            id=str(uuid.uuid4()),
            diff="def add(a, b):\n    return a * b\n", # Still wrong
            source_model="deepseek-coder",
            diff_size=40,
        ),
        CandidatePatch(
            id=str(uuid.uuid4()),
            diff="def add(a, b):\n    return a + b\n", # Correct!
            source_model="deepseek-coder",
            diff_size=40,
        )
    ]

    # 5. Policy ranks the candidates based on current (random) weights
    best_candidate, initial_q = policy.rank_candidates(candidates)
    
    assert best_candidate in candidates
    # Initial Q should be close to 0 given initialization
    assert isinstance(initial_q, float)

    # 6. Evaluate the selected candidate using PatchEvaluator
    evaluator = PatchEvaluator(workspace)
    # We pretend the agent picked the first (wrong) candidate first
    wrong_candidate = candidates[0]
    wrong_reward = evaluator.evaluate_candidate(wrong_candidate, "math_utils.py")
    
    # It compiles, but fails the pytest assert
    assert wrong_reward.compiles is True
    assert wrong_reward.passes_tests is False

    # Perform Bellman update on the chosen candidate
    td_error_1 = policy.update_patch_critic(wrong_reward.total_reward, wrong_candidate)

    # 7. Evaluate the correct candidate
    correct_candidate = candidates[1]
    correct_reward = evaluator.evaluate_candidate(correct_candidate, "math_utils.py")
    
    # It compiles AND passes tests
    assert correct_reward.compiles is True
    assert correct_reward.passes_tests is True
    assert correct_reward.total_reward > wrong_reward.total_reward

    # Perform Bellman update on the correct candidate
    td_error_2 = policy.update_patch_critic(correct_reward.total_reward, correct_candidate)

    # The RL agent has now processed structured rewards!
    assert td_error_1 >= 0
    assert td_error_2 >= 0
    assert "patch_embed" in wrong_candidate.metadata
    assert "patch_embed" in correct_candidate.metadata

