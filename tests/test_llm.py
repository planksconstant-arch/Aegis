import pytest
from local_ide_agent.agent.llm import LLMClient
from local_ide_agent.config import LLMSettings

def test_llm_client_initialization():
    settings = LLMSettings(base_url="http://localhost:11434/v1", model_name="deepseek-coder", max_candidates=2)
    client = LLMClient(settings)
    assert client.settings.model_name == "deepseek-coder"
    assert client.settings.max_candidates == 2

def test_llm_system_prompt_formatting():
    settings = LLMSettings()
    client = LLMClient(settings)
    
    from local_ide_agent.schemas import Observation
    obs = Observation(task="Fix a syntax error", open_files=[], diagnostics=["SyntaxError: unexpected EOF"])
    prompt = client._build_prompt(obs, "print(hello\n")
    
    assert "Fix a syntax error" in prompt
    assert "print(hello" in prompt
    assert "SyntaxError: unexpected EOF" in prompt
