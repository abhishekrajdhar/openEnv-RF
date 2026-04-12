"""Deterministic graders for customer support tasks."""

from __future__ import annotations

from typing import Dict, Tuple

from .models import CustomerSupportState, EvaluationSnapshot, ResolutionPayload
from .tasks import SupportTask

MIN_SCORE = 0.001
MAX_SCORE = 0.999


def _normalize_text(text: str | None) -> str:
    return (text or "").strip().lower()


def _contains_all_terms(text: str | None, terms: list[str]) -> float:
    normalized = _normalize_text(text)
    if not terms:
        return MAX_SCORE
    hits = sum(1 for term in terms if term.lower() in normalized)
    return _strict_score(hits / len(terms))


def _strict_score(value: float) -> float:
    return round(min(max(value, MIN_SCORE), MAX_SCORE), 4)


def compute_progress(task: SupportTask, state: CustomerSupportState) -> EvaluationSnapshot:
    expected = task.expected
    visible_ids = {artifact.artifact_id for artifact in state.visible_artifacts}
    discovered = [artifact_id for artifact_id in expected.required_artifacts if artifact_id in visible_ids]
    discovered_conflicts = [artifact_id for artifact_id in expected.conflicting_artifacts if artifact_id in visible_ids]
    artifact_coverage = len(discovered) / len(expected.required_artifacts) if expected.required_artifacts else MAX_SCORE
    conflict_coverage = (
        len(discovered_conflicts) / len(expected.conflicting_artifacts)
        if expected.conflicting_artifacts
        else MAX_SCORE
    )
    tag_hits = sum(1 for tag in expected.required_tags if tag in state.tags)
    tag_coverage = tag_hits / len(expected.required_tags) if expected.required_tags else MAX_SCORE
    reply_coverage = _contains_all_terms(state.draft_reply, expected.reply_must_include)
    invalid_action_penalty = min(float(state.hidden_context.get("invalid_action_count", 0)) * 0.02, 0.10)
    raw_unsupported_claim_penalty = min(float(state.hidden_context.get("unsupported_claim_count", 0)) * 0.03, 0.12)
    raw_hallucination_penalty = min(invalid_action_penalty + raw_unsupported_claim_penalty, 0.18)
    unsupported_claim_penalty = _strict_score(raw_unsupported_claim_penalty)
    hallucination_penalty = _strict_score(raw_hallucination_penalty)

    resolution = state.last_resolution
    resolution_code_score = MAX_SCORE if resolution and resolution.resolution_code == expected.resolution_code else MIN_SCORE
    refund_score = MAX_SCORE if resolution and abs(resolution.refund_amount - expected.refund_amount) < 0.01 else MIN_SCORE
    shipping_score = MAX_SCORE if resolution and abs(resolution.shipping_refund - expected.shipping_refund) < 0.01 else MIN_SCORE
    credit_score = MAX_SCORE if resolution and abs(resolution.goodwill_credit - expected.goodwill_credit) < 0.01 else MIN_SCORE

    task_completion_accuracy = _strict_score(
        0.35 * artifact_coverage
        + 0.15 * conflict_coverage
        + 0.15 * (MAX_SCORE if state.route == expected.route else MIN_SCORE)
        + 0.10 * (MAX_SCORE if state.priority == expected.priority else MIN_SCORE)
        + 0.10 * tag_coverage
        + 0.15 * resolution_code_score,
    )
    policy_adherence = _strict_score(
        0.25 * (MAX_SCORE if state.route == expected.route else MIN_SCORE)
        + 0.20 * (MAX_SCORE if state.priority == expected.priority else MIN_SCORE)
        + 0.20 * tag_coverage
        + 0.20 * conflict_coverage
        + 0.15 * resolution_code_score,
    )
    tool_usage_score = _strict_score(0.65 * artifact_coverage + 0.35 * conflict_coverage)
    response_quality = _strict_score(reply_coverage)
    user_satisfaction_proxy = _strict_score(
        0.45 * reply_coverage
        + 0.20 * refund_score
        + 0.10 * shipping_score
        + 0.10 * credit_score
        + 0.15 * (MAX_SCORE if state.route == expected.route else MIN_SCORE),
    )

    final_score = MIN_SCORE
    final_score += 0.15 * artifact_coverage
    final_score += 0.10 * conflict_coverage
    final_score += 0.12 if state.route == expected.route else MIN_SCORE
    final_score += 0.12 if state.priority == expected.priority else MIN_SCORE
    final_score += 0.11 * tag_coverage
    final_score += 0.10 * reply_coverage

    if resolution:
        final_score += 0.15 * resolution_code_score
        final_score += 0.08 * refund_score
        final_score += 0.03 * shipping_score
        final_score += 0.04 * credit_score
    final_score = _strict_score(final_score - raw_hallucination_penalty)

    return EvaluationSnapshot(
        discovered_required_artifacts=discovered,
        discovered_conflicting_artifacts=discovered_conflicts,
        artifact_coverage=_strict_score(artifact_coverage),
        conflict_coverage=_strict_score(conflict_coverage),
        tag_coverage=_strict_score(tag_coverage),
        reply_coverage=_strict_score(reply_coverage),
        task_completion_accuracy=task_completion_accuracy,
        policy_adherence=policy_adherence,
        tool_usage_score=tool_usage_score,
        response_quality=response_quality,
        user_satisfaction_proxy=user_satisfaction_proxy,
        routing_correct=_strict_score(MAX_SCORE if state.route == expected.route else MIN_SCORE),
        priority_correct=_strict_score(MAX_SCORE if state.priority == expected.priority else MIN_SCORE),
        hallucination_penalty=hallucination_penalty,
        unsupported_claim_penalty=round(unsupported_claim_penalty, 4),
        final_score=final_score,
    )


def grade_submission(task: SupportTask, resolution: ResolutionPayload, state: CustomerSupportState) -> Tuple[float, Dict[str, float]]:
    expected = task.expected
    visible_ids = {a.artifact_id for a in state.visible_artifacts}
    components = {
        "resolution_code": MAX_SCORE if resolution.resolution_code == expected.resolution_code else MIN_SCORE,
        "refund_amount": MAX_SCORE if abs(resolution.refund_amount - expected.refund_amount) < 0.01 else MIN_SCORE,
        "shipping_refund": MAX_SCORE if abs(resolution.shipping_refund - expected.shipping_refund) < 0.01 else MIN_SCORE,
        "goodwill_credit": MAX_SCORE if abs(resolution.goodwill_credit - expected.goodwill_credit) < 0.01 else MIN_SCORE,
        "reply_quality": _contains_all_terms(resolution.message or state.draft_reply, expected.reply_must_include),
        "route": MAX_SCORE if state.route == expected.route else MIN_SCORE,
        "priority": MAX_SCORE if state.priority == expected.priority else MIN_SCORE,
        "tags": _strict_score(sum(1 for tag in expected.required_tags if tag in state.tags) / len(expected.required_tags)),
        "artifacts": _strict_score(
            sum(1 for artifact_id in expected.required_artifacts if artifact_id in visible_ids)
            / len(expected.required_artifacts)
        ),
        "conflicts": _strict_score(
            sum(1 for artifact_id in expected.conflicting_artifacts if artifact_id in visible_ids)
            / len(expected.conflicting_artifacts)
            if expected.conflicting_artifacts
            else MAX_SCORE
        ),
    }

    score = (
        0.18 * components["resolution_code"]
        + 0.12 * components["refund_amount"]
        + 0.05 * components["shipping_refund"]
        + 0.05 * components["goodwill_credit"]
        + 0.15 * components["reply_quality"]
        + 0.13 * components["route"]
        + 0.08 * components["priority"]
        + 0.08 * components["tags"]
        + 0.08 * components["artifacts"]
        + 0.08 * components["conflicts"]
    )
    strict_components = {k: _strict_score(v) for k, v in components.items()}
    return _strict_score(score), strict_components
