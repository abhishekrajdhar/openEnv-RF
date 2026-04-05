"""Deterministic graders for customer support tasks."""

from __future__ import annotations

from typing import Dict, Tuple

from .models import CustomerSupportState, EvaluationSnapshot, ResolutionPayload
from .tasks import SupportTask


def _normalize_text(text: str | None) -> str:
    return (text or "").strip().lower()


def _contains_all_terms(text: str | None, terms: list[str]) -> float:
    normalized = _normalize_text(text)
    if not terms:
        return 1.0
    hits = sum(1 for term in terms if term.lower() in normalized)
    return hits / len(terms)


def compute_progress(task: SupportTask, state: CustomerSupportState) -> EvaluationSnapshot:
    expected = task.expected
    visible_ids = {artifact.artifact_id for artifact in state.visible_artifacts}
    discovered = [artifact_id for artifact_id in expected.required_artifacts if artifact_id in visible_ids]
    tag_hits = sum(1 for tag in expected.required_tags if tag in state.tags)
    tag_coverage = tag_hits / len(expected.required_tags) if expected.required_tags else 1.0
    reply_coverage = _contains_all_terms(state.draft_reply, expected.reply_must_include)
    hallucination_penalty = min(float(state.hidden_context.get("invalid_action_count", 0)) * 0.02, 0.10)

    final_score = 0.0
    final_score += 0.20 * (len(discovered) / len(expected.required_artifacts))
    final_score += 0.15 if state.route == expected.route else 0.0
    final_score += 0.15 if state.priority == expected.priority else 0.0
    final_score += 0.15 * tag_coverage
    final_score += 0.15 * reply_coverage

    resolution = state.last_resolution
    if resolution:
        final_score += 0.12 if resolution.resolution_code == expected.resolution_code else 0.0
        final_score += 0.04 if abs(resolution.refund_amount - expected.refund_amount) < 0.01 else 0.0
        final_score += 0.02 if abs(resolution.shipping_refund - expected.shipping_refund) < 0.01 else 0.0
        final_score += 0.02 if abs(resolution.goodwill_credit - expected.goodwill_credit) < 0.01 else 0.0
    final_score = max(0.0, final_score - hallucination_penalty)

    return EvaluationSnapshot(
        discovered_required_artifacts=discovered,
        tag_coverage=round(tag_coverage, 4),
        reply_coverage=round(reply_coverage, 4),
        routing_correct=state.route == expected.route,
        priority_correct=state.priority == expected.priority,
        hallucination_penalty=round(hallucination_penalty, 4),
        final_score=round(min(final_score, 1.0), 4),
    )


def grade_submission(task: SupportTask, resolution: ResolutionPayload, state: CustomerSupportState) -> Tuple[float, Dict[str, float]]:
    expected = task.expected
    components = {
        "resolution_code": 1.0 if resolution.resolution_code == expected.resolution_code else 0.0,
        "refund_amount": 1.0 if abs(resolution.refund_amount - expected.refund_amount) < 0.01 else 0.0,
        "shipping_refund": 1.0 if abs(resolution.shipping_refund - expected.shipping_refund) < 0.01 else 0.0,
        "goodwill_credit": 1.0 if abs(resolution.goodwill_credit - expected.goodwill_credit) < 0.01 else 0.0,
        "reply_quality": _contains_all_terms(resolution.message or state.draft_reply, expected.reply_must_include),
        "route": 1.0 if state.route == expected.route else 0.0,
        "priority": 1.0 if state.priority == expected.priority else 0.0,
        "tags": sum(1 for tag in expected.required_tags if tag in state.tags) / len(expected.required_tags),
        "artifacts": (
            sum(1 for artifact_id in expected.required_artifacts if artifact_id in {a.artifact_id for a in state.visible_artifacts})
            / len(expected.required_artifacts)
        ),
    }

    score = (
        0.18 * components["resolution_code"]
        + 0.12 * components["refund_amount"]
        + 0.05 * components["shipping_refund"]
        + 0.05 * components["goodwill_credit"]
        + 0.15 * components["reply_quality"]
        + 0.15 * components["route"]
        + 0.10 * components["priority"]
        + 0.10 * components["tags"]
        + 0.10 * components["artifacts"]
    )
    return round(min(score, 1.0), 4), components
