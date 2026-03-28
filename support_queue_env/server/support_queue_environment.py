"""Core environment implementation."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any, Dict

from ..compat import Environment, StepResult
from ..graders import compute_progress, grade_submission
from ..models import (
    CustomerSupportAction,
    CustomerSupportObservation,
    CustomerSupportReward,
    CustomerSupportState,
    ResolutionPayload,
)
from ..tasks import SupportTask, build_tasks


MAX_STEPS = 12
ALL_TASKS = build_tasks()
TASK_ORDER = [
    "delayed_shipping_refund",
    "defective_return_window",
    "subscription_cancellation_dispute",
]


class SupportQueueEnvironment(Environment):
    """Customer support triage environment with deterministic tasks."""

    def __init__(self) -> None:
        self._tasks = ALL_TASKS
        self._task_index = 0
        self._current_task: SupportTask | None = None
        self._state: CustomerSupportState | None = None
        self._seen_action_signatures: set[str] = set()

    def reset(self, task_id: str | None = None) -> StepResult[CustomerSupportObservation]:
        if task_id is None:
            task_id = TASK_ORDER[self._task_index % len(TASK_ORDER)]
            self._task_index += 1
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = self._tasks[task_id]
        self._current_task = task
        self._state = CustomerSupportState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            done=False,
            task_id=task.task.task_id,
            instructions=task.instructions,
            visible_artifacts=deepcopy(task.initial_artifacts),
            hidden_context={"available_queries": list(task.searchable_artifacts.keys())},
            max_steps=MAX_STEPS,
        )
        self._seen_action_signatures = set()
        self._refresh_evaluation()
        reward = self._reward_model(
            reward_delta=0.0,
            rationale="Environment reset.",
            partial_signals={"initial_state": self._state.progress_score},
        )
        return StepResult(
            observation=self._build_observation("Ticket ready for review.", reward),
            reward=0.0,
            done=False,
            info={"task_id": task_id},
        )

    def step(self, action: CustomerSupportAction) -> StepResult[CustomerSupportObservation]:
        state = self._require_state()
        task = self._require_task()

        if state.done:
            reward = self._reward_model(
                reward_delta=-0.05,
                rationale="Episode already ended; reset before taking more actions.",
                penalties={"post_done_action": -0.05},
            )
            return StepResult(
                observation=self._build_observation("Episode already complete.", reward),
                reward=reward.reward_delta,
                done=True,
                info={"error": "episode_done"},
            )

        state.step_count += 1
        last_progress = state.progress_score
        penalties: Dict[str, float] = {}
        partial_signals: Dict[str, float] = {}
        status = ""

        repeated = self._record_action_signature(action)
        if repeated:
            penalties["repeat_action"] = -0.03

        if action.action_type == "search_policy":
            status = self._search_artifact(action.argument, expected_types={"policy"})
        elif action.action_type == "open_order":
            status = self._search_artifact(action.argument, expected_types={"order"})
        elif action.action_type == "open_account":
            status = self._search_artifact(action.argument, expected_types={"account"})
        elif action.action_type == "open_log":
            status = self._search_artifact(action.argument, expected_types={"log"})
        elif action.action_type == "set_priority":
            status = self._set_priority(action.argument)
        elif action.action_type == "route_ticket":
            status = self._set_route(action.argument)
        elif action.action_type == "add_tag":
            status = self._add_tag(action.argument)
        elif action.action_type == "draft_reply":
            state.draft_reply = (action.message or "").strip()
            status = "Saved draft reply."
        elif action.action_type == "submit_resolution":
            status, partial_signals = self._submit_resolution(action.resolution, task)
        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

        self._refresh_evaluation()
        delta = round(state.progress_score - last_progress, 4)
        reward_delta = delta + sum(partial_signals.values()) + sum(penalties.values())

        if state.step_count >= state.max_steps and not state.done:
            state.done = True
            penalties["max_steps"] = penalties.get("max_steps", 0.0) - 0.10
            reward_delta += penalties["max_steps"]
            status = f"{status} Step budget exhausted."

        reward = self._reward_model(
            reward_delta=round(reward_delta, 4),
            rationale=status,
            partial_signals=partial_signals,
            penalties=penalties,
            grader_score=state.evaluation.final_score if state.done else None,
        )
        return StepResult(
            observation=self._build_observation(status, reward),
            reward=reward.reward_delta,
            done=state.done,
            info={"evaluation": state.evaluation.model_dump()},
        )

    def state(self) -> CustomerSupportState:
        return self._require_state()

    def _require_state(self) -> CustomerSupportState:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        return self._state

    def _require_task(self) -> SupportTask:
        if self._current_task is None:
            raise RuntimeError("No active task; call reset().")
        return self._current_task

    def _reward_model(
        self,
        reward_delta: float,
        rationale: str,
        partial_signals: Dict[str, float] | None = None,
        penalties: Dict[str, float] | None = None,
        grader_score: float | None = None,
    ) -> CustomerSupportReward:
        state = self._require_state()
        state.cumulative_reward = round(state.cumulative_reward + reward_delta, 4)
        return CustomerSupportReward(
            reward_delta=round(reward_delta, 4),
            cumulative_reward=state.cumulative_reward,
            progress_score=state.progress_score,
            partial_signals=partial_signals or {},
            penalties=penalties or {},
            grader_score=grader_score,
            rationale=rationale,
        )

    def _build_observation(
        self, last_action_status: str, reward: CustomerSupportReward
    ) -> CustomerSupportObservation:
        state = self._require_state()
        task = self._require_task()
        return CustomerSupportObservation(
            instructions=state.instructions,
            task=task.task,
            visible_artifacts=deepcopy(state.visible_artifacts),
            tags=list(state.tags),
            priority=state.priority,
            route=state.route,
            draft_reply=state.draft_reply,
            last_action_status=last_action_status,
            action_history=list(state.hidden_context.get("action_history", [])),
            reward_details=reward,
            remaining_steps=max(state.max_steps - state.step_count, 0),
        )

    def _record_action_signature(self, action: CustomerSupportAction) -> bool:
        state = self._require_state()
        signature = json.dumps(action.model_dump(mode="json"), sort_keys=True)
        history = state.hidden_context.setdefault("action_history", [])
        history.append(signature)
        already_seen = signature in self._seen_action_signatures
        self._seen_action_signatures.add(signature)
        return already_seen

    def _search_artifact(self, query: str | None, expected_types: set[str]) -> str:
        task = self._require_task()
        state = self._require_state()
        normalized = (query or "").strip().lower()
        artifact = task.searchable_artifacts.get(normalized)
        if artifact is None and query:
            artifact = next(
                (
                    candidate
                    for key, candidate in task.searchable_artifacts.items()
                    if normalized in key.lower() or key.lower() in normalized
                ),
                None,
            )
        if artifact is None or artifact.artifact_type not in expected_types:
            return f"No matching {'/'.join(sorted(expected_types))} found for query '{query}'."
        if artifact.artifact_id not in {item.artifact_id for item in state.visible_artifacts}:
            state.visible_artifacts.append(deepcopy(artifact))
        return f"Opened {artifact.artifact_type} {artifact.artifact_id}: {artifact.title}."

    def _set_priority(self, priority: str | None) -> str:
        state = self._require_state()
        allowed = {"low", "normal", "high", "urgent"}
        if priority not in allowed:
            return f"Priority '{priority}' is invalid."
        state.priority = priority  # type: ignore[assignment]
        return f"Priority set to {priority}."

    def _set_route(self, route: str | None) -> str:
        state = self._require_state()
        state.route = (route or "").strip().lower() or None
        return f"Route set to {state.route}."

    def _add_tag(self, tag: str | None) -> str:
        state = self._require_state()
        normalized = (tag or "").strip().lower()
        if not normalized:
            return "Tag cannot be empty."
        if normalized not in state.tags:
            state.tags.append(normalized)
        return f"Tag '{normalized}' added."

    def _submit_resolution(
        self, resolution: ResolutionPayload | None, task: SupportTask
    ) -> tuple[str, Dict[str, float]]:
        state = self._require_state()
        if resolution is None:
            return "Resolution payload is required.", {"invalid_submission": -0.08}
        if resolution.message and not state.draft_reply:
            state.draft_reply = resolution.message
        state.last_resolution = resolution
        final_score, components = grade_submission(task, resolution, state)
        state.done = True
        state.progress_score = final_score
        state.evaluation = compute_progress(task, state)
        signals = {f"final_{key}": round((value - 0.5) * 0.02, 4) for key, value in components.items()}
        signals["final_score_bonus"] = round(final_score * 0.3, 4)
        return f"Resolution submitted with grader score {final_score:.2f}.", signals

    def _refresh_evaluation(self) -> None:
        task = self._require_task()
        state = self._require_state()
        state.evaluation = compute_progress(task, state)
        state.progress_score = state.evaluation.final_score
