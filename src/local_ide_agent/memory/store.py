from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from local_ide_agent.schemas import ClientRegistration, FeedbackRecord, MemorySnapshot, StylePreference


@dataclass
class MemoryStore:
    database_path: Path

    def __post_init__(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path = self.database_path.with_suffix(".json")
        self.use_fallback = False
        try:
            self._initialize()
        except sqlite3.Error:
            self.use_fallback = True
            self._initialize_fallback()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    ide_name TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    observation_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    result_text TEXT NOT NULL,
                    reward REAL NOT NULL DEFAULT 0,
                    accepted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    reward REAL NOT NULL,
                    accepted INTEGER NOT NULL,
                    notes TEXT NOT NULL,
                    acceptance_latency_seconds REAL,
                    post_accept_edit_distance INTEGER,
                    reverted_within_commits INTEGER,
                    accepted_at_hour INTEGER,
                    reward_components_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shadow_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shadow_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    source_root TEXT NOT NULL,
                    shadow_root TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS replay_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action_index INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    done INTEGER NOT NULL,
                    td_error REAL NOT NULL,
                    state_vector_json TEXT NOT NULL,
                    next_state_vector_json TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS style_preferences (
                    user_id TEXT NOT NULL,
                    pref_key TEXT NOT NULL,
                    pref_value TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, pref_key)
                );

                CREATE TABLE IF NOT EXISTS action_success_rates (
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    total_count INTEGER NOT NULL DEFAULT 0,
                    accept_count INTEGER NOT NULL DEFAULT 0,
                    reward_sum REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, action_type)
                );
                """
            )

    def _initialize_fallback(self) -> None:
        if self.fallback_path.exists():
            return
        self.fallback_path.write_text(
            json.dumps(
                {
                    "clients": {},
                    "episodes": [],
                    "shadow_runs": [],
                    "replay_transitions": [],
                    "feedback": [],
                    "style_preferences": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _read_fallback(self) -> dict[str, object]:
        payload = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        payload.setdefault("clients", {})
        payload.setdefault("episodes", [])
        payload.setdefault("shadow_runs", [])
        payload.setdefault("replay_transitions", [])
        payload.setdefault("feedback", [])
        payload.setdefault("style_preferences", {})
        return payload

    def _write_fallback(self, payload: dict[str, object]) -> None:
        self.fallback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register_client(self, registration: ClientRegistration) -> None:
        if self.use_fallback:
            payload = self._read_fallback()
            clients = dict(payload["clients"])
            clients[registration.client_id] = {
                "client_id": registration.client_id,
                "user_id": registration.user_id,
                "ide_name": registration.ide_name,
                "workspace_root": registration.workspace_root,
                "capabilities": registration.capabilities,
                "metadata": registration.metadata,
                "last_seen": datetime.now(UTC).isoformat(),
            }
            payload["clients"] = clients
            self._write_fallback(payload)
            return

        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO clients (
                    client_id, user_id, ide_name, workspace_root, capabilities_json, metadata_json, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    ide_name=excluded.ide_name,
                    workspace_root=excluded.workspace_root,
                    capabilities_json=excluded.capabilities_json,
                    metadata_json=excluded.metadata_json,
                    last_seen=excluded.last_seen
                """,
                (
                    registration.client_id,
                    registration.user_id,
                    registration.ide_name,
                    registration.workspace_root,
                    json.dumps(registration.capabilities),
                    json.dumps(registration.metadata),
                    now,
                ),
            )

    def record_episode(
        self,
        *,
        client_id: str,
        user_id: str,
        task: str,
        observation_json: str,
        decision_json: str,
        result_text: str,
        reward: float = 0.0,
        accepted: bool = False,
    ) -> None:
        if self.use_fallback:
            payload = self._read_fallback()
            episodes = list(payload["episodes"])
            episodes.append(
                {
                    "client_id": client_id,
                    "user_id": user_id,
                    "task": task,
                    "observation_json": observation_json,
                    "decision_json": decision_json,
                    "result_text": result_text,
                    "reward": reward,
                    "accepted": accepted,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            payload["episodes"] = episodes
            self._write_fallback(payload)
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (
                    client_id, user_id, task, observation_json, decision_json, result_text, reward, accepted, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    user_id,
                    task,
                    observation_json,
                    decision_json,
                    result_text,
                    reward,
                    1 if accepted else 0,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def record_feedback(self, feedback: FeedbackRecord) -> None:
        if self.use_fallback:
            payload = self._read_fallback()
            feedback_rows = list(payload["feedback"])
            feedback_rows.append(
                {
                    "client_id": feedback.client_id,
                    "user_id": feedback.user_id,
                    "action_type": feedback.action_type,
                    "reward": feedback.reward,
                    "accepted": feedback.accepted,
                    "notes": feedback.notes,
                    "acceptance_latency_seconds": feedback.acceptance_latency_seconds,
                    "post_accept_edit_distance": feedback.post_accept_edit_distance,
                    "reverted_within_commits": feedback.reverted_within_commits,
                    "accepted_at_hour": feedback.accepted_at_hour,
                    "reward_components": feedback.reward_components,
                    "metadata": feedback.metadata,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            payload["feedback"] = feedback_rows
            style_preferences = dict(payload["style_preferences"])
            for key, value in feedback.style_updates.items():
                existing = style_preferences.get(feedback.user_id, {})
                current = existing.get(key, {"value": value, "weight": 0})
                current["value"] = value
                current["weight"] = float(current["weight"]) + 1.0
                existing[key] = current
                style_preferences[feedback.user_id] = existing
            payload["style_preferences"] = style_preferences
            self._write_fallback(payload)
            return

        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    client_id, user_id, action_type, reward, accepted, notes,
                    acceptance_latency_seconds, post_accept_edit_distance, reverted_within_commits,
                    accepted_at_hour, reward_components_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.client_id,
                    feedback.user_id,
                    feedback.action_type,
                    feedback.reward,
                    1 if feedback.accepted else 0,
                    feedback.notes,
                    feedback.acceptance_latency_seconds,
                    feedback.post_accept_edit_distance,
                    feedback.reverted_within_commits,
                    feedback.accepted_at_hour,
                    json.dumps(feedback.reward_components),
                    json.dumps(feedback.metadata),
                    now,
                ),
            )

            for key, value in feedback.style_updates.items():
                conn.execute(
                    """
                    INSERT INTO style_preferences (user_id, pref_key, pref_value, weight, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, pref_key) DO UPDATE SET
                        pref_value=excluded.pref_value,
                        weight=style_preferences.weight + 1,
                        updated_at=excluded.updated_at
                    """,
                    (feedback.user_id, key, value, 1.0, now),
                )

            # Upsert into action_success_rates (O(1) incremental update)
            conn.execute(
                """
                INSERT INTO action_success_rates
                    (user_id, action_type, total_count, accept_count, reward_sum, updated_at)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(user_id, action_type) DO UPDATE SET
                    total_count=action_success_rates.total_count + 1,
                    accept_count=action_success_rates.accept_count + excluded.accept_count,
                    reward_sum=action_success_rates.reward_sum + excluded.reward_sum,
                    updated_at=excluded.updated_at
                """,
                (
                    feedback.user_id,
                    feedback.action_type,
                    1 if feedback.accepted else 0,
                    feedback.reward,
                    now,
                ),
            )

    def record_shadow_run(
        self,
        *,
        shadow_id: str,
        user_id: str,
        source_root: str,
        shadow_root: str,
        objective: str,
        status: str,
        summary: dict[str, object],
    ) -> None:
        if self.use_fallback:
            payload = self._read_fallback()
            shadow_runs = list(payload["shadow_runs"])
            shadow_runs.append(
                {
                    "shadow_id": shadow_id,
                    "user_id": user_id,
                    "source_root": source_root,
                    "shadow_root": shadow_root,
                    "objective": objective,
                    "status": status,
                    "summary": summary,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            payload["shadow_runs"] = shadow_runs
            self._write_fallback(payload)
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO shadow_runs (
                    shadow_id, user_id, source_root, shadow_root, objective, status, summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shadow_id,
                    user_id,
                    source_root,
                    shadow_root,
                    objective,
                    status,
                    json.dumps(summary),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def record_replay_transition(
        self,
        *,
        user_id: str,
        action_index: int,
        reward: float,
        done: bool,
        td_error: float,
        state_vector: list[float],
        next_state_vector: list[float],
        context: dict[str, object],
    ) -> None:
        if self.use_fallback:
            payload = self._read_fallback()
            rows = list(payload["replay_transitions"])
            rows.append(
                {
                    "user_id": user_id,
                    "action_index": action_index,
                    "reward": reward,
                    "done": done,
                    "td_error": td_error,
                    "state_vector": state_vector,
                    "next_state_vector": next_state_vector,
                    "context": context,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            payload["replay_transitions"] = rows[-5000:]
            self._write_fallback(payload)
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO replay_transitions (
                    user_id, action_index, reward, done, td_error,
                    state_vector_json, next_state_vector_json, context_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    action_index,
                    reward,
                    1 if done else 0,
                    td_error,
                    json.dumps(state_vector),
                    json.dumps(next_state_vector),
                    json.dumps(context),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def load_replay_transitions(self, user_id: str, limit: int = 512) -> list[dict[str, object]]:
        if self.use_fallback:
            payload = self._read_fallback()
            rows = [item for item in payload["replay_transitions"] if item["user_id"] == user_id]
            return rows[-limit:]

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action_index, reward, done, td_error, state_vector_json, next_state_vector_json, context_json
                FROM replay_transitions
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            {
                "action_index": int(row[0]),
                "reward": float(row[1]),
                "done": bool(row[2]),
                "td_error": float(row[3]),
                "state_vector": json.loads(row[4]),
                "next_state_vector": json.loads(row[5]),
                "context": json.loads(row[6]),
            }
            for row in reversed(rows)
        ]

    def get_action_success_rates(
        self,
        user_id: str,
        min_count: int = 3,
    ) -> list[dict[str, object]]:
        """
        Return per-action success stats from the materialized view.
        Each row: {action_type, total_count, accept_count, accept_rate, avg_reward}
        Ordered by avg_reward descending.
        """
        if self.use_fallback:
            # Recompute from raw feedback in fallback mode
            payload = self._read_fallback()
            counts: dict[str, dict] = {}
            for row in payload["feedback"]:
                if row["user_id"] != user_id:
                    continue
                at = str(row["action_type"])
                d = counts.setdefault(at, {"total": 0, "accepted": 0, "reward_sum": 0.0})
                d["total"] += 1
                d["accepted"] += int(bool(row.get("accepted")))
                d["reward_sum"] += float(row.get("reward", 0.0))
            result = []
            for at, d in counts.items():
                if d["total"] < min_count:
                    continue
                result.append({
                    "action_type": at,
                    "total_count": d["total"],
                    "accept_count": d["accepted"],
                    "accept_rate": round(d["accepted"] / d["total"], 3),
                    "avg_reward": round(d["reward_sum"] / d["total"], 4),
                })
            return sorted(result, key=lambda x: x["avg_reward"], reverse=True)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action_type, total_count, accept_count, reward_sum
                FROM action_success_rates
                WHERE user_id = ? AND total_count >= ?
                ORDER BY (reward_sum / total_count) DESC
                """,
                (user_id, min_count),
            ).fetchall()
        return [
            {
                "action_type": row[0],
                "total_count": int(row[1]),
                "accept_count": int(row[2]),
                "accept_rate": round(int(row[2]) / max(int(row[1]), 1), 3),
                "avg_reward": round(float(row[3]) / max(int(row[1]), 1), 4),
            }
            for row in rows
        ]

    def snapshot(self, user_id: str, limit: int = 10) -> MemorySnapshot:
        if self.use_fallback:
            payload = self._read_fallback()
            episodes = [item for item in payload["episodes"] if item["user_id"] == user_id]
            feedback_rows = [item for item in payload["feedback"] if item["user_id"] == user_id]
            styles = payload["style_preferences"].get(user_id, {})
            hour_scores: dict[str, list[float]] = {}

            successful_actions: dict[str, list[float]] = {}
            for row in feedback_rows:
                action_type = str(row["action_type"])
                successful_actions.setdefault(action_type, []).append(float(row["reward"]))
                hour = row.get("accepted_at_hour")
                if hour is not None and bool(row.get("accepted")):
                    hour_scores.setdefault(f"hour_{int(hour)}", []).append(float(row["reward"]))

            preferred_actions = [
                key for key, values in sorted(
                    successful_actions.items(),
                    key=lambda item: (sum(item[1]) / max(len(item[1]), 1)),
                    reverse=True,
                )
                if sum(values) / max(len(values), 1) > 0
            ][:limit]
            temporal_patterns = {
                key: round(sum(values) / max(len(values), 1), 3)
                for key, values in hour_scores.items()
            }
            global_style_principles = _derive_global_style_principles(styles)

            return MemorySnapshot(
                recent_tasks=[str(row["task"]) for row in episodes[-limit:]][::-1],
                recent_failures=[
                    str(row["notes"]) for row in feedback_rows
                    if not bool(row["accepted"]) and str(row["notes"])
                ][-limit:][::-1],
                preferred_actions=preferred_actions,
                style_preferences=[
                    StylePreference(key=key, value=str(value["value"]), weight=float(value["weight"]))
                    for key, value in styles.items()
                ],
                temporal_patterns=temporal_patterns,
                global_style_principles=global_style_principles,
            )

        with self._connect() as conn:
            task_rows = conn.execute(
                """
                SELECT task FROM episodes
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            failure_rows = conn.execute(
                """
                SELECT notes FROM feedback
                WHERE user_id = ? AND accepted = 0 AND notes != ''
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            preferred_rows = conn.execute(
                """
                SELECT action_type, AVG(reward) AS avg_reward
                FROM feedback
                WHERE user_id = ?
                GROUP BY action_type
                HAVING avg_reward > 0
                ORDER BY avg_reward DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            style_rows = conn.execute(
                """
                SELECT pref_key, pref_value, weight
                FROM style_preferences
                WHERE user_id = ?
                ORDER BY weight DESC, pref_key ASC
                """,
                (user_id,),
            ).fetchall()
            temporal_rows = conn.execute(
                """
                SELECT accepted_at_hour, AVG(reward) AS avg_reward
                FROM feedback
                WHERE user_id = ? AND accepted = 1 AND accepted_at_hour IS NOT NULL
                GROUP BY accepted_at_hour
                """,
                (user_id,),
            ).fetchall()

        style_map = {row[0]: {"value": row[1], "weight": float(row[2])} for row in style_rows}
        return MemorySnapshot(
            recent_tasks=[row[0] for row in task_rows],
            recent_failures=[row[0] for row in failure_rows],
            preferred_actions=[row[0] for row in preferred_rows],
            style_preferences=[
                StylePreference(key=row[0], value=row[1], weight=float(row[2]))
                for row in style_rows
            ],
            temporal_patterns={f"hour_{int(row[0])}": round(float(row[1]), 3) for row in temporal_rows},
            global_style_principles=_derive_global_style_principles(style_map),
        )


def _derive_global_style_principles(style_map: dict[str, object]) -> list[str]:
    principles: list[str] = []
    if "comment_style" in style_map:
        value = style_map["comment_style"]["value"] if isinstance(style_map["comment_style"], dict) else style_map["comment_style"]
        principles.append(f"Comments should stay {value}.")
    if "autonomy_mode" in style_map:
        value = style_map["autonomy_mode"]["value"] if isinstance(style_map["autonomy_mode"], dict) else style_map["autonomy_mode"]
        principles.append(f"Default autonomy mode is {value}.")
    if not principles:
        principles.append("Prefer consistent, low-surprise edits across projects.")
    return principles
