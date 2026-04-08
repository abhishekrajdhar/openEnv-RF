"""Deterministic customer support tasks with multi-step evidence conflicts."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from .models import ExpectedOutcome, TaskSummary, VisibleArtifact


class SupportTask(BaseModel):
    task: TaskSummary
    instructions: str
    initial_artifacts: List[VisibleArtifact]
    searchable_artifacts: Dict[str, VisibleArtifact]
    expected: ExpectedOutcome
    reference_notes: List[str] = Field(default_factory=list)


def build_tasks() -> dict[str, SupportTask]:
    common_instructions = (
        "You are an AI support operations agent for Northstar Commerce. "
        "Resolve the ticket by gathering evidence, setting the right priority and route, "
        "adding the correct tags, drafting a customer-facing reply, and submitting a final resolution."
    )

    easy = SupportTask(
        task=TaskSummary(
            task_id="delayed_shipping_refund",
            title="Delayed shipment needs shipping-fee refund",
            difficulty="easy",
            customer_message=(
                "My order A100 hasn't arrived and the promised delivery window ended six days ago. "
                "Can you help?"
            ),
            visible_order_id="A100",
            visible_customer_id="C100",
            success_criteria=[
                "Inspect the order and the shipping-delay policy.",
                "Route the ticket to logistics with normal priority.",
                "Tag the case as delayed_shipment.",
                "Refund the shipping fee and explain the refund in the reply.",
            ],
            reasoning_challenges=[
                "Use order timing together with policy rules before issuing a refund.",
            ],
        ),
        instructions=common_instructions,
        initial_artifacts=[
            VisibleArtifact(
                artifact_type="order",
                artifact_id="A100",
                title="Order A100",
                content=(
                    "Customer C100. Status: In transit. Carrier ETA: 2026-03-18. "
                    "Today: 2026-03-24. Shipping fee paid: $8.99. No prior refund."
                ),
            )
        ],
        searchable_artifacts={
            "shipping delay policy": VisibleArtifact(
                artifact_type="policy",
                artifact_id="P_DELAY",
                title="Shipping delay policy",
                content=(
                    "If a standard shipment is more than 5 days late beyond ETA, refund the shipping fee, "
                    "route to logistics, and apologize with an updated follow-up commitment."
                ),
            ),
            "customer C100": VisibleArtifact(
                artifact_type="account",
                artifact_id="C100",
                title="Customer C100",
                content="Segment: standard. Tenure: 2 years. No special handling flags.",
            ),
        },
        expected=ExpectedOutcome(
            route="logistics",
            priority="normal",
            required_tags=["delayed_shipment"],
            required_artifacts=["A100", "P_DELAY"],
            resolution_code="refund_shipping_fee",
            shipping_refund=8.99,
            reply_must_include=["refund", "8.99", "delay"],
        ),
        reference_notes=["Simple delay refund with one policy lookup."],
    )

    medium = SupportTask(
        task=TaskSummary(
            task_id="defective_return_window",
            title="Opened appliance is defective within return window",
            difficulty="medium",
            customer_message=(
                "The blender from order B200 started smoking on the second use. "
                "I opened it 20 days ago. I want a refund."
            ),
            visible_order_id="B200",
            visible_customer_id="C200",
            success_criteria=[
                "Inspect the order plus both the defective-item and opened-box policies.",
                "Route to returns with high priority.",
                "Tag the case as defective_item and safety_risk.",
                "Approve return and full refund with a clear reply.",
            ],
            reasoning_challenges=[
                "Resolve a conflict between the normal opened-box rule and the defective-item exception.",
                "Combine product safety information with return eligibility.",
            ],
        ),
        instructions=common_instructions,
        initial_artifacts=[
            VisibleArtifact(
                artifact_type="order",
                artifact_id="B200",
                title="Order B200",
                content=(
                    "Customer C200. Product: PulsePro Blender. Paid: $79.99. Delivered: 2026-03-02. "
                    "Return window: 30 days. Item status: customer reported defect. "
                    "Packaging opened 20 days ago. Notes: smoke observed on second use."
                ),
            )
        ],
        searchable_artifacts={
            "defective return policy": VisibleArtifact(
                artifact_type="policy",
                artifact_id="P_DEFECT",
                title="Defective item policy",
                content=(
                    "Defective items reported within 30 days qualify for return label issuance, "
                    "full refund, and high-priority routing to returns. Safety issues should be tagged."
                ),
            ),
            "opened box policy": VisibleArtifact(
                artifact_type="policy",
                artifact_id="P_OPEN_BOX",
                title="Opened-box appliance policy",
                content=(
                    "Opened but non-defective kitchen appliances are usually ineligible for cash refunds "
                    "and may receive exchange or store credit only."
                ),
            ),
            "customer C200": VisibleArtifact(
                artifact_type="account",
                artifact_id="C200",
                title="Customer C200",
                content="Segment: standard. Prior orders: 6. Prior defects: 0.",
            ),
        },
        expected=ExpectedOutcome(
            route="returns",
            priority="high",
            required_tags=["defective_item", "safety_risk"],
            required_artifacts=["B200", "P_DEFECT", "P_OPEN_BOX"],
            conflicting_artifacts=["P_DEFECT", "P_OPEN_BOX"],
            resolution_code="return_and_refund",
            refund_amount=79.99,
            reply_must_include=["refund", "return label", "defective"],
        ),
        reference_notes=["Requires conflict resolution between standard opened-box rules and the defect exception."],
    )

    hard = SupportTask(
        task=TaskSummary(
            task_id="subscription_cancellation_dispute",
            title="VIP customer charged after cancellation request",
            difficulty="hard",
            customer_message=(
                "I asked to cancel my Pro Pantry subscription before renewal, but I was charged again. "
                "Please fix this. Order/subscription ref S300."
            ),
            visible_order_id="S300",
            visible_customer_id="C300",
            success_criteria=[
                "Inspect the subscription record, cancellation log, billing policy, and VIP save-playbook.",
                "Route to billing with urgent priority.",
                "Tag the case as billing_dispute and vip_customer.",
                "Issue the prorated refund plus goodwill credit and explain why.",
            ],
            reasoning_challenges=[
                "Resolve conflict between a standard retention playbook and a billing-failure remediation rule.",
                "Use logs, policy, and VIP context together before deciding between refund and save-offer treatment.",
            ],
        ),
        instructions=common_instructions,
        initial_artifacts=[
            VisibleArtifact(
                artifact_type="order",
                artifact_id="S300",
                title="Subscription S300",
                content=(
                    "Customer C300. Product: Pro Pantry subscription. Renewal charge: $24.50 on 2026-03-21. "
                    "Current status: active. Loyalty tier: VIP."
                ),
            )
        ],
        searchable_artifacts={
            "cancellation log S300": VisibleArtifact(
                artifact_type="log",
                artifact_id="L_CANCEL_300",
                title="Cancellation log for S300",
                content=(
                    "2026-03-19 09:14 UTC: customer requested cancellation in chat. "
                    "2026-03-19 09:16 UTC: automation failed to persist cancellation due to billing sync timeout."
                ),
            ),
            "billing cancellation policy": VisibleArtifact(
                artifact_type="policy",
                artifact_id="P_BILLING",
                title="Billing cancellation policy",
                content=(
                    "If a customer requested cancellation before renewal and internal automation failed, "
                    "refund the latest renewal charge, apply a goodwill credit for VIP customers, "
                    "route urgently to billing, and confirm the cancellation has been completed."
                ),
            ),
            "vip retention save playbook": VisibleArtifact(
                artifact_type="policy",
                artifact_id="P_SAVE",
                title="VIP retention save playbook",
                content=(
                    "For dissatisfied VIP subscribers with active plans, agents may offer a discount or credit "
                    "instead of refunding if there is no verified billing or cancellation-system error."
                ),
            ),
            "customer C300": VisibleArtifact(
                artifact_type="account",
                artifact_id="C300",
                title="Customer C300",
                content=(
                    "Segment: VIP. Lifetime value: $3,480. Prior support CSAT: 9.8/10. "
                    "Retention risk flag: elevated if unresolved."
                ),
            ),
        },
        expected=ExpectedOutcome(
            route="billing",
            priority="urgent",
            required_tags=["billing_dispute", "vip_customer"],
            required_artifacts=["S300", "L_CANCEL_300", "P_BILLING", "P_SAVE", "C300"],
            conflicting_artifacts=["L_CANCEL_300", "P_BILLING", "P_SAVE"],
            resolution_code="refund_and_credit",
            refund_amount=24.50,
            goodwill_credit=10.00,
            reply_must_include=["cancel", "refund", "credit", "vip"],
        ),
        reference_notes=["Requires resolving a retention-playbook conflict using logs, billing policy, and VIP context."],
    )

    return {
        easy.task.task_id: easy,
        medium.task.task_id: medium,
        hard.task.task_id: hard,
    }
