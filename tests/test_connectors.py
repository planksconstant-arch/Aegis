import pytest
from local_ide_agent.connectors.ide import LocalIDEConnector
from local_ide_agent.schemas import Action, Decision

@pytest.fixture
def connector(tmp_path):
    return LocalIDEConnector(workspace_root=tmp_path)

def test_connector_initialization(connector):
    assert "events.jsonl" in str(connector.event_log.path)

def test_connector_can_handle_apply_patch(connector):
    # Ensure our hybrid action is supported by the IDE executor
    action = Action(
        action_type="apply_patch",
        description="Apply fixing patch",
        risk="low",
        payload={"diff": "some diff", "target_file": "file.py"}
    )
    decision = Decision(action=action, confidence=0.9, requires_approval=False, reason="")
    
    assert connector.present_suggestion(decision) is not None
