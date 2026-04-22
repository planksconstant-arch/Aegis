from __future__ import annotations

import json
from datetime import datetime

from local_ide_agent.agent.planner import Planner
from local_ide_agent.agent.policy import Policy
from local_ide_agent.agent.rewarding import derive_feedback_reward
from local_ide_agent.config import AppSettings
from local_ide_agent.connectors.base import IDEConnector
from local_ide_agent.memory.store import MemoryStore
from local_ide_agent.rl.replay import PrioritizedReplayBuffer, ReplayTransition
from local_ide_agent.schemas import ActionType, Decision, FeedbackRecord, Observation, RiskLevel


class LocalIDEAgent:
    def __init__(
        self,
        settings: AppSettings,
        connector: IDEConnector,
        policy: Policy,
        memory_store: MemoryStore | None = None,
        replay_buffer: PrioritizedReplayBuffer | None = None,
    ) -> None:
        self.settings = settings
        self.connector = connector
        self.policy = policy
        self.planner = Planner()
        self.memory_store = memory_store
        self.replay_buffer = replay_buffer

    def evaluate(self, observation: Observation) -> Decision:
        enriched = observation.model_copy(deep=True)
        user_id = str(enriched.metadata.get("user_id", "default"))
        enriched.metadata.setdefault("local_hour", datetime.now().hour)
        if self.memory_store:
            snapshot = self.memory_store.snapshot(user_id, self.settings.memory.max_recent_items)
            enriched.metadata["recent_tasks"] = snapshot.recent_tasks
            enriched.metadata["recent_failures"] = snapshot.recent_failures
            enriched.metadata["preferred_actions"] = snapshot.preferred_actions
            enriched.metadata["style_preferences"] = {item.key: item.value for item in snapshot.style_preferences}
            enriched.metadata["temporal_patterns"] = snapshot.temporal_patterns
            enriched.metadata["global_style_principles"] = snapshot.global_style_principles
        return self.policy.decide(enriched)

    def execute(self, decision: Decision) -> str:
        if not self.settings.autonomy.enabled:
            return "Autonomy disabled. Decision recorded but not executed."

        if decision.action.risk == RiskLevel.HIGH and self.settings.autonomy.block_high_risk:
            return "Blocked a high-risk action."

        if decision.autonomy_tier == "silent-shadow":
            return "Low-confidence action routed to shadow-only exploration."

        if decision.action.risk == RiskLevel.MEDIUM and self.settings.autonomy.require_approval_for_medium_risk:
            return self.connector.request_approval(decision)

        if decision.autonomy_tier == "review":
            return self.connector.present_suggestion(decision)

        if decision.action.risk == RiskLevel.LOW and not self.settings.autonomy.auto_apply_low_risk:
            return self.connector.present_suggestion(decision)

        if decision.action.action_type == ActionType.RUN_COMMAND:
            return self.connector.run_command(decision.action.payload.get("command", ""))

        if decision.action.action_type == ActionType.EDIT_FILE:
            return self.connector.apply_edit(
                decision.action.payload.get("path", ""),
                decision.action.payload.get("content", ""),
            )

        return self.connector.present_suggestion(decision)

    def tick(self, observation: Observation) -> str:
        summary = self.planner.summarize(observation)
        decision = self.evaluate(observation)
        result = self.execute(decision)
        if self.replay_buffer is not None and hasattr(self.policy, "last_fused_state"):
            state_vector = list(getattr(self.policy.last_fused_state, "state_vector", []))
            action_index = int(getattr(self.policy, "last_action_index", 0))
            user_id = str(observation.metadata.get("user_id", "default"))
            self.replay_buffer.add(
                ReplayTransition(
                    state_vector=state_vector,
                    action_index=action_index,
                    reward=0.0,
                    next_state_vector=state_vector,
                    done=False,
                    context={"task": observation.task, "autonomy_tier": decision.autonomy_tier},
                    td_error=max(0.05, 1 - decision.confidence),
                )
            )
            if self.memory_store:
                self.memory_store.record_replay_transition(
                    user_id=user_id,
                    action_index=action_index,
                    reward=0.0,
                    done=False,
                    td_error=max(0.05, 1 - decision.confidence),
                    state_vector=state_vector,
                    next_state_vector=state_vector,
                    context={"task": observation.task, "autonomy_tier": decision.autonomy_tier},
                )
        if self.memory_store:
            client_id = str(observation.metadata.get("client_id", "local-cli"))
            user_id = str(observation.metadata.get("user_id", "default"))
            self.memory_store.record_episode(
                client_id=client_id,
                user_id=user_id,
                task=observation.task,
                observation_json=observation.model_dump_json(),
                decision_json=decision.model_dump_json(),
                result_text=result,
            )
        return f"{summary}\nDecision={self.planner.explain(decision)}\nResult={result}"

    def record_feedback(self, feedback: FeedbackRecord) -> None:
        reward = feedback.reward
        if reward == 0.0:
            reward, components = derive_feedback_reward(feedback)
            feedback.reward = reward
            feedback.reward_components = components
        self.policy.update(feedback.reward)
        if self.memory_store:
            self.memory_store.record_feedback(feedback)
