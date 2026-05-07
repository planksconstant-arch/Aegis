from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from local_ide_agent.agent.core import LocalIDEAgent
from local_ide_agent.connectors.registry import ConnectorRegistry
from local_ide_agent.deployment.background import ShadowRunRequest
from local_ide_agent.lab.mirofish_sim import MirofishStyleSimulator
from local_ide_agent.schemas import ClientRegistration, FeedbackRecord, Observation


@dataclass
class BridgeServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


class IDEBridgeHandler(BaseHTTPRequestHandler):
    agent: LocalIDEAgent
    workspace_root: Path
    connector_registry = ConnectorRegistry()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self.respond(HTTPStatus.OK, {"status": "ok"})
            return

        if parsed.path == "/files":
            connector = self.agent.connector
            files = connector.list_files() if hasattr(connector, "list_files") else []
            self.respond(HTTPStatus.OK, {"files": files})
            return

        if parsed.path == "/connectors":
            self.respond(HTTPStatus.OK, {"connectors": self.connector_registry.list_descriptors()})
            return

        if parsed.path == "/memory":
            user_id = parse_qs(parsed.query).get("user_id", ["default"])[0]
            snapshot = self.agent.memory_store.snapshot(user_id) if self.agent.memory_store else None
            self.respond(
                HTTPStatus.OK,
                {"memory": snapshot.model_dump() if snapshot else {}},
            )
            return

        if parsed.path == "/training-status":
            self.respond(HTTPStatus.OK, _build_training_status(self.agent))
            return

        self.respond(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/observe":
            payload = self.read_json()
            observation = Observation.model_validate(payload)
            decision = self.agent.evaluate(observation)
            response = {
                "decision": decision.model_dump(),
                "execution_preview": self.agent.planner.explain(decision),
            }
            self.respond(HTTPStatus.OK, response)
            return

        if self.path == "/generate-fix":
            payload = self.read_json()
            obs = Observation.model_validate(payload)
            file_content = payload.get("file_content", "")
            
            candidates = self.agent.llm.generate_candidates(obs, file_content)
            if not candidates:
                self.respond(HTTPStatus.OK, {"diff": ""})
                return
                
            best_candidate, q_val = self.agent.policy.rank_candidates(candidates)
            self.respond(HTTPStatus.OK, {"diff": best_candidate.diff, "score": q_val})
            return

        if self.path == "/act":
            payload = self.read_json()
            observation = Observation.model_validate(payload)
            result = self.agent.tick(observation)
            self.respond(HTTPStatus.OK, {"result": result})
            return

        if self.path == "/file/read":
            payload = self.read_json()
            connector = self.agent.connector
            if not hasattr(connector, "read_file"):
                self.respond(HTTPStatus.NOT_IMPLEMENTED, {"error": "connector does not support file reads"})
                return
            content = connector.read_file(payload["path"])
            self.respond(HTTPStatus.OK, {"path": payload["path"], "content": content})
            return

        if self.path == "/command":
            payload = self.read_json()
            output = self.agent.connector.run_command(payload["command"])
            self.respond(HTTPStatus.OK, {"output": output})
            return

        if self.path == "/clients/register":
            payload = self.read_json()
            registration = ClientRegistration.model_validate(payload)
            if not self.agent.memory_store:
                self.respond(HTTPStatus.NOT_IMPLEMENTED, {"error": "memory store unavailable"})
                return
            self.agent.memory_store.register_client(registration)
            self.respond(HTTPStatus.OK, {"status": "registered", "client_id": registration.client_id})
            return

        if self.path == "/feedback":
            payload = self.read_json()
            feedback = FeedbackRecord.model_validate(payload)
            self.agent.record_feedback(feedback)
            self.respond(HTTPStatus.OK, {"status": "recorded"})
            return

        if self.path == "/shadow/run":
            payload = self.read_json()
            shadow_service = getattr(self.agent, "shadow_service", None)
            if shadow_service is None:
                self.respond(HTTPStatus.NOT_IMPLEMENTED, {"error": "shadow service unavailable"})
                return
            request = ShadowRunRequest(
                user_id=str(payload.get("user_id", "default")),
                objective=str(payload.get("objective", "Improve the current project safely in a shadow copy")),
                label=str(payload.get("label", "recent")),
            )
            result = shadow_service.start_run(request)
            self.respond(HTTPStatus.OK, result)
            return

        if self.path == "/lab/counterfactual-run":
            payload = self.read_json()
            shadow_service = getattr(self.agent, "shadow_service", None)
            if shadow_service is None:
                self.respond(HTTPStatus.NOT_IMPLEMENTED, {"error": "shadow service unavailable"})
                return
            request = ShadowRunRequest(
                user_id=str(payload.get("user_id", "default")),
                objective=str(payload.get("objective", "Explore multiple safe implementation paths in shadow copies")),
                label=str(payload.get("label", "lab")),
            )
            result = shadow_service.start_counterfactual_run(request)
            self.respond(HTTPStatus.OK, result)
            return

        if self.path == "/lab/mirofish-review":
            payload = self.read_json()
            simulator = MirofishStyleSimulator()
            objective = str(payload.get("objective", "Review a candidate strategy"))
            candidate = dict(payload.get("candidate", {}))
            self.respond(HTTPStatus.OK, simulator.review_candidate(candidate, objective))
            return

        self.respond(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw_body.decode("utf-8"))

    def respond(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_bridge_server(agent: LocalIDEAgent, config: BridgeServerConfig) -> None:
    handler = type(
        "ConfiguredIDEBridgeHandler",
        (IDEBridgeHandler,),
        {"agent": agent, "workspace_root": getattr(agent.connector, "workspace_root", Path.cwd())},
    )
    server = ThreadingHTTPServer((config.host, config.port), handler)
    print(f"IDE bridge listening on http://{config.host}:{config.port}")
    server.serve_forever()


def _build_training_status(agent: LocalIDEAgent) -> dict:
    """
    Collect live RL health metrics from the running agent.
    Returns a JSON-serializable dict suitable for external dashboards.
    """
    import os
    import time

    status: dict = {"status": "ok"}

    # Policy metrics
    policy = getattr(agent, "policy", None)
    if policy is not None:
        status["epsilon"] = round(getattr(policy, "epsilon", 1.0), 4)
        status["step_count"] = getattr(policy, "_step_count", 0)
        weight_path = getattr(getattr(policy, "hp", None), "weight_path", None)
        if weight_path and os.path.exists(weight_path):
            age_seconds = round(time.time() - os.path.getmtime(weight_path), 1)
            status["weight_file_age_seconds"] = age_seconds
            status["weight_file_path"] = weight_path
        else:
            status["weight_file_age_seconds"] = None

    # Replay buffer metrics
    buf = getattr(agent, "replay_buffer", None)
    if buf is not None:
        status["replay_buffer_size"] = len(buf)
        status["replay_buffer_capacity"] = getattr(buf, "capacity", None)
        status["per_beta"] = round(getattr(buf, "beta", 0.4), 4)
        fill_pct = len(buf) / max(getattr(buf, "capacity", 1), 1)
        status["replay_buffer_fill_pct"] = round(fill_pct * 100, 1)

    # Recent reward history
    reward_history = getattr(policy, "reward_history", [])
    if reward_history:
        recent = reward_history[-100:]
        status["avg_reward_last_100"] = round(sum(recent) / len(recent), 4)
        status["min_reward_last_100"] = round(min(recent), 4)
        status["max_reward_last_100"] = round(max(recent), 4)

    # Memory store stats
    if agent.memory_store:
        try:
            snap = agent.memory_store.snapshot("default", limit=5)
            status["memory_recent_tasks"] = len(snap.recent_tasks)
            status["memory_preferred_actions"] = snap.preferred_actions[:3]
        except Exception:
            pass
        try:
            rates = agent.memory_store.get_action_success_rates("default", min_count=1)
            status["action_success_rates"] = rates[:8]
        except Exception:
            pass

    return status
