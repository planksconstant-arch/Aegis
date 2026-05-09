from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(str, Enum):
    OBSERVE = "observe"
    SUGGEST = "suggest"
    EDIT_FILE = "edit_file"
    RUN_COMMAND = "run_command"
    ASK_APPROVAL = "ask_approval"
    NO_OP = "no_op"
    # New action type for LLM-generated patch
    APPLY_PATCH = "apply_patch"


class CandidatePatch(BaseModel):
    diff: str
    source_model: str
    diff_size: int
    id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredReward(BaseModel):
    compiles: bool = False
    passes_linter: bool = False
    passes_tests: bool = False
    diff_size_penalty: float = 0.0
    total_reward: float = 0.0


class Observation(BaseModel):
    task: str
    open_files: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    user_present: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    action_type: ActionType
    description: str
    risk: RiskLevel = RiskLevel.LOW
    payload: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    action: Action
    confidence: float = 0.0
    requires_approval: bool = False
    reason: str = ""
    autonomy_tier: str = "review"


class EpisodeResult(BaseModel):
    reward: float
    decision_count: int
    completed: bool
    notes: list[str] = Field(default_factory=list)


class ClientRegistration(BaseModel):
    client_id: str
    ide_name: str
    workspace_root: str
    user_id: str = "default"
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackRecord(BaseModel):
    client_id: str = "default"
    user_id: str = "default"
    reward: float = 0.0
    accepted: bool = False
    action_type: str = ""
    notes: str = ""
    acceptance_latency_seconds: float | None = None
    post_accept_edit_distance: int | None = None
    reverted_within_commits: int | None = None
    accepted_at_hour: int | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)
    style_updates: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StylePreference(BaseModel):
    key: str
    value: str
    weight: float = 1.0


class MemorySnapshot(BaseModel):
    recent_tasks: list[str] = Field(default_factory=list)
    recent_failures: list[str] = Field(default_factory=list)
    preferred_actions: list[str] = Field(default_factory=list)
    style_preferences: list[StylePreference] = Field(default_factory=list)
    temporal_patterns: dict[str, float] = Field(default_factory=dict)
    global_style_principles: list[str] = Field(default_factory=list)
