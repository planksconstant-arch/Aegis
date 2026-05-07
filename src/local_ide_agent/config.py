from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AutonomySettings(BaseModel):
    enabled: bool = True
    allow_background_execution: bool = True
    auto_apply_low_risk: bool = False
    require_approval_for_medium_risk: bool = True
    block_high_risk: bool = True


class LLMSettings(BaseModel):
    enabled: bool = False
    base_url: str | None = "http://localhost:11434/v1"  # Optional, mainly for local or custom endpoints
    api_key: str | None = None  # If None, relies on environment variables (litellm standard)
    model_name: str = "ollama/deepseek-coder"  # Now expects a litellm-compatible string
    max_candidates: int = 3
    system_prompt: str = "You are a coding assistant. Generate a diff to fix the issue."


class TrainingSettings(BaseModel):
    warmup_days: int = 14
    episodes_per_day: int = 50
    max_steps_per_episode: int = 25
    policy_backend: str = "heuristic"


class DeploymentSettings(BaseModel):
    runtime_name: str = "mirofish"
    max_background_agents: int = 3
    heartbeat_seconds: int = 30


class BridgeSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    workspace_root: str = "."
    event_log_path: str = ".agent/events.jsonl"


class MemorySettings(BaseModel):
    database_path: str = ".agent/agent.db"
    max_recent_items: int = 10


class ShadowWorkspaceSettings(BaseModel):
    enabled: bool = True
    root_directory: str = ".shadow"
    retention_limit: int = 5
    auto_clone_recent_workspace: bool = True


class ResearchSettings(BaseModel):
    embedding_dimensions: int = 256
    sequence_window: int = 64
    offline_batch_size: int = 32
    policy_model: str = "sequence-transformer"
    value_model: str = "twin-q-network"


class RLHyperparams(BaseModel):
    """Tunable hyperparameters for the full RL pipeline."""

    # Discount and return
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO-style clipping
    ppo_clip_epsilon: float = 0.2

    # Loss coefficients
    entropy_coefficient: float = 0.01
    critic_coefficient: float = 0.5

    # Optimiser
    learning_rate: float = 3e-4
    batch_size: int = 32

    # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 100_000
    replay_capacity: int = 4096

    # Conservative Q-Learning offline regulariser
    cql_alpha: float = 0.5

    # Epsilon-greedy exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000

    # Weight persistence
    weight_path: str = ".agent/policy_weights.npz"

    # Architecture
    twin_q: bool = True
    trunk_hidden_sizes: list[int] = [256, 128]
    critic_hidden_sizes: list[int] = [64]

    # ---- RND Curiosity ----
    curiosity_enabled: bool = True
    curiosity_beta: float = 0.005         # weight on intrinsic reward (small to not swamp env reward)
    curiosity_lr: float = 1e-3
    curiosity_embed_dim: int = 64
    curiosity_weight_path: str = ".agent/rnd_weights.npz"
    curiosity_normalize: bool = True

    # ---- N-Step Returns ----
    n_step: int = 5                        # lookahead steps for TD targets

    # ---- Curriculum Learning ----
    curriculum_enabled: bool = True
    curriculum_window: int = 20            # rolling window for success rate
    curriculum_success_threshold: float = 0.70
    curriculum_demotion_threshold: float = 0.35

    # ---- Trajectory Buffer ----
    trajectory_window: int = 10            # steps of episode context to retain


class EncoderBackendSettings(BaseModel):
    code_backend: str = "deterministic"
    behavior_backend: str = "deterministic"
    context_backend: str = "deterministic"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    onnx_model_path: str | None = None


class AppSettings(BaseModel):
    autonomy: AutonomySettings = Field(default_factory=AutonomySettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    bridge: BridgeSettings = Field(default_factory=BridgeSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    shadow: ShadowWorkspaceSettings = Field(default_factory=ShadowWorkspaceSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    encoders: EncoderBackendSettings = Field(default_factory=EncoderBackendSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rl: RLHyperparams = Field(default_factory=RLHyperparams)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppSettings":
        if path is None:
            return cls()

        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.model_validate(raw)
