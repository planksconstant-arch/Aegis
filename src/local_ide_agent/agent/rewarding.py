from __future__ import annotations

from local_ide_agent.schemas import FeedbackRecord


def derive_feedback_reward(feedback: FeedbackRecord) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    reward = 0.0

    components["acceptance"] = 0.6 if feedback.accepted else -0.4
    reward += components["acceptance"]

    if feedback.acceptance_latency_seconds is not None:
        if feedback.acceptance_latency_seconds <= 10:
            components["latency"] = 0.25
        elif feedback.acceptance_latency_seconds <= 60:
            components["latency"] = 0.10
        else:
            components["latency"] = -0.10
        reward += components["latency"]

    if feedback.post_accept_edit_distance is not None:
        if feedback.post_accept_edit_distance == 0:
            components["edit_distance"] = 0.20
        elif feedback.post_accept_edit_distance <= 20:
            components["edit_distance"] = 0.05
        else:
            components["edit_distance"] = -0.20
        reward += components["edit_distance"]

    if feedback.reverted_within_commits is not None:
        if feedback.reverted_within_commits <= 1:
            components["revert"] = -0.45
        elif feedback.reverted_within_commits <= 5:
            components["revert"] = -0.20
        else:
            components["revert"] = -0.05
        reward += components["revert"]

    return round(reward, 3), components
