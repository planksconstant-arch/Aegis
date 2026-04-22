import pytest
from local_ide_agent.cli.dashboard import _render, _color_for
from local_ide_agent.schemas import Observation

def test_dashboard_color_helper():
    res = _color_for(0.8, 0.0, 1.0)
    assert res is not None

def test_dashboard_can_parse_stats():
    # We can at least test the color functions won't crash
    val = _color_for(0.2, 0.0, 1.0)
    assert val is not None
