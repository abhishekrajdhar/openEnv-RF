"""Typed models for the customer support environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import Field

from .compat import Action, Observation, OpenEnvModel, State


ActionName = Literal[
    "search_policy",
    "open_order",
    "open_account",
    "open_log",
    "set_priority",
    "route_ticket",
    "add_tag",
    "draft_reply",
    "submit_resolution",
]


class ResolutionPayload(OpenEnvModel):
    resolution_code: Literal[
        "refund_shipping_fee",
        "return_and_refund",
        "refund_and_credit",
        "escalate_only",
        "request_more_info",
    ]
    refund_amount: float = Field(default=0.001)
    goodwill_credit: float = Field(default=0.001)
    shipping_refund: float = Field(default=0.001)
    message: str = ""


class CustomerSupportAction(Action):
    action_type: ActionName
    argument: str | None = None
    message: str | None = None
    resolution: ResolutionPayload | None = None


class VisibleArtifact(OpenEnvModel):
    artifact_type: Literal["policy", "order", "account", "log"]
    artifact_id: str
    title: str
    content: str


class ToolDescriptor(OpenEnvModel):
    action_type: ActionName
    system_name: str
    description: str


class TaskSummary(OpenEnvModel):
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    customer_message: str
    visible_order_id: str | None = None
    visible_customer_id: str | None = None
    success_criteria: List[str]
    reasoning_challenges: List[str] = Field(default_factory=list)


class CustomerSupportReward(OpenEnvModel):
    reward_delta: float = Field(gt=0.001, lt=0.999)
    cumulative_reward: float = Field(gt=0.001, lt=0.999)
    progress_score: float = Field(gt=0.001, lt=0.999)
    partial_signals: Dict[str, float] = Field(default_factory=dict)
    penalties: Dict[str, float] = Field(default_factory=dict)
    grader_score: float = Field(default=0.01, gt=0.001, lt=0.999)
    rationale: str


class CustomerSupportObservation(Observation):
    instructions: str
    task: TaskSummary
    visible_artifacts: List[VisibleArtifact] = Field(default_factory=list)
    available_tools: List[ToolDescriptor] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    route: str | None = None
    draft_reply: str | None = None
    last_action_status: str
    action_history: List[str] = Field(default_factory=list)
    reward_details: CustomerSupportReward
    remaining_steps: int


class ExpectedOutcome(OpenEnvModel):
    route: str
    priority: Literal["low", "normal", "high", "urgent"]
    required_tags: List[str]
    required_artifacts: List[str]
    conflicting_artifacts: List[str] = Field(default_factory=list)
    resolution_code: str
    refund_amount: float = Field(default=0.001)
    goodwill_credit: float = Field(default=0.001)
    shipping_refund: float = Field(default=0.001)
    reply_must_include: List[str] = Field(default_factory=list)


class EvaluationSnapshot(OpenEnvModel):
    discovered_required_artifacts: List[str] = Field(default_factory=list)
    discovered_conflicting_artifacts: List[str] = Field(default_factory=list)
    artifact_coverage: float = Field(default=0.01, gt=0.001, lt=0.999)
    conflict_coverage: float = Field(default=0.01, gt=0.001, lt=0.999)
    tag_coverage: float = Field(default=0.01, gt=0.001, lt=0.999)
    reply_coverage: float = Field(default=0.01, gt=0.001, lt=0.999)
    task_completion_accuracy: float = Field(default=0.01, gt=0.001, lt=0.999)
    policy_adherence: float = Field(default=0.01, gt=0.001, lt=0.999)
    tool_usage_score: float = Field(default=0.01, gt=0.001, lt=0.999)
    response_quality: float = Field(default=0.01, gt=0.001, lt=0.999)
    user_satisfaction_proxy: float = Field(default=0.01, gt=0.001, lt=0.999)
    routing_correct: float = Field(default=0.01, gt=0.001, lt=0.999)
    priority_correct: float = Field(default=0.01, gt=0.001, lt=0.999)
    hallucination_penalty: float = Field(default=0.01, gt=0.001, lt=0.999)
    unsupported_claim_penalty: float = Field(default=0.01, gt=0.001, lt=0.999)
    final_score: float = Field(default=0.01, gt=0.001, lt=0.999)


class CustomerSupportState(State):
    task_id: str
    instructions: str
    visible_artifacts: List[VisibleArtifact] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    route: str | None = None
    draft_reply: str | None = None
    last_resolution: ResolutionPayload | None = None
    cumulative_reward: float = Field(default=0.01, gt=0.001, lt=0.999)
    progress_score: float = Field(default=0.01, gt=0.001, lt=0.999)
    max_steps: int = 12
    hidden_context: Dict[str, Any] = Field(default_factory=dict)
    evaluation: EvaluationSnapshot = Field(default_factory=EvaluationSnapshot)
