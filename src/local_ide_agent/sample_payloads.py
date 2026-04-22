from __future__ import annotations

from local_ide_agent.schemas import Observation


def sample_observation() -> Observation:
    return Observation(
        task="Help with the current coding task in the IDE",
        open_files=["README.md", "src/local_ide_agent/main.py"],
        diagnostics=["Example warning: missing tests for new code path"],
        user_present=True,
        metadata={"source": "manual-sample", "client_id": "cli-sample", "user_id": "default"},
    )
