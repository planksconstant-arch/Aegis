from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from pathlib import Path

from local_ide_agent.config import EncoderBackendSettings
from local_ide_agent.schemas import Observation

# Learned attention fusion (replaces fixed weighted-sum)
from local_ide_agent.rl.attention import LearnedCrossAttentionFusion

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None


def _project_text(text: str, size: int, scale: float = 1.0) -> list[float]:
    values = [0.0] * size
    for index, char in enumerate(text):
        slot = index % size
        values[slot] += ((ord(char) % 97) / 96.0) * scale
    return [round(tanh(value / 4), 6) for value in values]


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    maximum = max(abs(item) for item in values) or 1.0
    return [round(item / maximum, 6) for item in values]


def _fit_dimension(values: list[float], size: int) -> list[float]:
    if len(values) == size:
        return [round(item, 6) for item in values]
    if len(values) > size:
        return [round(item, 6) for item in values[:size]]
    if not values:
        return [0.0] * size
    padded = list(values)
    while len(padded) < size:
        padded.extend(values[: min(len(values), size - len(padded))])
    return [round(item, 6) for item in padded[:size]]


@dataclass
class FusedState:
    code_embed: list[float]
    behavior_embed: list[float]
    context_embed: list[float]
    state_vector: list[float]


class EncoderBackend:
    output_size: int

    def encode(self, observation: Observation) -> list[float]:
        raise NotImplementedError


class DeterministicCodeBackend(EncoderBackend):
    output_size = 384

    def encode(self, observation: Observation) -> list[float]:
        joined = " ".join([observation.task, *observation.open_files, *observation.diagnostics])
        return _project_text(joined, self.output_size, scale=1.15)


class SentenceTransformerCodeBackend(EncoderBackend):
    output_size = 384

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name) if SentenceTransformer is not None else None

    def encode(self, observation: Observation) -> list[float]:
        if self.model is None:
            return DeterministicCodeBackend().encode(observation)
        joined = " ".join([observation.task, *observation.open_files, *observation.diagnostics])
        embedding = self.model.encode(joined)
        return _fit_dimension([float(item) for item in embedding], self.output_size)


class ONNXCodeBackend(EncoderBackend):
    output_size = 384

    def __init__(self, model_path: str | None) -> None:
        self.model_path = model_path
        self.session = None
        if ort is not None and model_path and Path(model_path).exists():
            self.session = ort.InferenceSession(model_path)

    def encode(self, observation: Observation) -> list[float]:
        if self.session is None:
            return DeterministicCodeBackend().encode(observation)
        joined = " ".join([observation.task, *observation.open_files, *observation.diagnostics])
        fallback = _project_text(joined, self.output_size, scale=1.1)
        return fallback


class DeterministicBehaviorBackend(EncoderBackend):
    output_size = 128

    def encode(self, observation: Observation) -> list[float]:
        metadata = observation.metadata
        behavior_bits = [
            f"latency={metadata.get('last_acceptance_latency', 0)}",
            f"edit_distance={metadata.get('last_edit_distance', 0)}",
            f"reverts={metadata.get('recent_reverts', 0)}",
            f"user_present={observation.user_present}",
            f"phase={metadata.get('session_phase', 'build')}",
        ]
        return _project_text(" ".join(behavior_bits), self.output_size, scale=0.9)


class DeterministicContextBackend(EncoderBackend):
    output_size = 64

    def encode(self, observation: Observation) -> list[float]:
        metadata = observation.metadata
        session_bits = [
            f"hour={metadata.get('local_hour', 12)}",
            f"pressure={metadata.get('pressure_level', 'normal')}",
            f"phase={metadata.get('session_phase', 'build')}",
            f"task_count={len(observation.open_files)}",
        ]
        return _project_text(" ".join(session_bits), self.output_size, scale=0.8)


class CrossAttentionFusion:
    def fuse(self, code_embed: list[float], behavior_embed: list[float], context_embed: list[float]) -> list[float]:
        prefix = _normalize(code_embed[:256])
        middle = _normalize(behavior_embed[:128])
        suffix = _normalize(context_embed[:64])
        cross_terms = []
        for idx in range(128):
            code_value = code_embed[idx] if idx < len(code_embed) else 0.0
            behavior_value = behavior_embed[idx] if idx < len(behavior_embed) else 0.0
            context_value = context_embed[idx % len(context_embed)] if context_embed else 0.0
            cross_terms.append(round((code_value * 0.5) + (behavior_value * 0.3) + (context_value * 0.2), 6))
        return prefix + middle + suffix + cross_terms


class StateEncoderStack:
    def __init__(self, settings: EncoderBackendSettings | None = None) -> None:
        self.settings = settings or EncoderBackendSettings()
        self.code_encoder = self._build_code_backend(self.settings)
        self.behavior_encoder = DeterministicBehaviorBackend()
        self.context_encoder = DeterministicContextBackend()
        # Learned cross-attention fusion (replaces fixed 0.5/0.3/0.2 sum)
        self.fusion = LearnedCrossAttentionFusion(
            d_code=384,
            d_behavior=128,
            d_context=64,
            d_k=64,
            d_v=256,
            d_out=576,
        )

    def _build_code_backend(self, settings: EncoderBackendSettings) -> EncoderBackend:
        if settings.code_backend == "sentence-transformer":
            return SentenceTransformerCodeBackend(settings.sentence_transformer_model)
        if settings.code_backend == "onnx":
            return ONNXCodeBackend(settings.onnx_model_path)
        return DeterministicCodeBackend()

    def encode(self, observation: Observation) -> FusedState:
        code_embed = self.code_encoder.encode(observation)
        behavior_embed = self.behavior_encoder.encode(observation)
        context_embed = self.context_encoder.encode(observation)
        # Learned cross-attention fusion -> 576-d state vector
        state_vector = self.fusion.fuse(code_embed, behavior_embed, context_embed)
        return FusedState(
            code_embed=code_embed,
            behavior_embed=behavior_embed,
            context_embed=context_embed,
            state_vector=list(state_vector),
        )
