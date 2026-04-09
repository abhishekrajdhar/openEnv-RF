from support_queue_env.client import SupportQueueEnvClient
from support_queue_env.models import CustomerSupportAction, ResolutionPayload


def test_easy_task_can_score_full_credit():
    client = SupportQueueEnvClient()
    client.reset(task_id="delayed_shipping_refund")
    client.step(CustomerSupportAction(action_type="search_policy", argument="shipping delay policy"))
    client.step(CustomerSupportAction(action_type="set_priority", argument="normal"))
    client.step(CustomerSupportAction(action_type="route_ticket", argument="logistics"))
    client.step(CustomerSupportAction(action_type="add_tag", argument="delayed_shipment"))
    client.step(
        CustomerSupportAction(
            action_type="draft_reply",
            message="Sorry about the delay. I refunded 8.99 for the shipping refund and will follow up on the delay.",
        )
    )
    result = client.step(
        CustomerSupportAction(
            action_type="submit_resolution",
            resolution=ResolutionPayload(
                resolution_code="refund_shipping_fee",
                shipping_refund=8.99,
                message="Sorry about the delay. I refunded 8.99 for the shipping refund and will follow up on the delay.",
            ),
        )
    )
    assert result.done is True
    assert result.info["evaluation"]["final_score"] >= 0.99
    assert 0.0 <= result.reward <= 1.0


def test_wrong_submission_is_partial_not_binary():
    client = SupportQueueEnvClient()
    client.reset(task_id="defective_return_window")
    client.step(CustomerSupportAction(action_type="search_policy", argument="defective return policy"))
    result = client.step(
        CustomerSupportAction(
            action_type="submit_resolution",
            resolution=ResolutionPayload(
                resolution_code="escalate_only",
                message="I am escalating this.",
            ),
        )
    )
    assert result.done is True
    assert 0.0 < result.info["evaluation"]["final_score"] < 1.0


def test_task_alias_and_invalid_action_penalty_are_supported():
    client = SupportQueueEnvClient()
    observation = client.reset(task_id="01")
    assert observation.task.task_id == "delayed_shipping_refund"
    result = client.step(CustomerSupportAction(action_type="route_ticket", argument="unknown_team"))
    assert 0.0 < result.reward < 1.0
    assert result.observation.reward_details.penalties["invalid_action"] < 0.0


def test_conflicting_evidence_improves_hard_task_progress():
    client = SupportQueueEnvClient()
    client.reset(task_id="subscription_cancellation_dispute")
    client.step(CustomerSupportAction(action_type="open_log", argument="cancellation log S300"))
    client.step(CustomerSupportAction(action_type="search_policy", argument="billing cancellation policy"))
    before = client.state().evaluation.conflict_coverage
    client.step(CustomerSupportAction(action_type="search_policy", argument="vip retention save playbook"))
    after = client.state().evaluation.conflict_coverage
    assert before < after
    assert 0.99 <= after < 1.0


def test_unsupported_claims_in_resolution_increase_hallucination_penalty():
    client = SupportQueueEnvClient()
    client.reset(task_id="delayed_shipping_refund")
    result = client.step(
        CustomerSupportAction(
            action_type="submit_resolution",
            resolution=ResolutionPayload(
                resolution_code="request_more_info",
                message="I issued a refund based on policy and confirmed it in the log.",
            ),
        )
    )
    assert result.done is True
    assert result.observation.reward_details.penalties["unsupported_claims"] < 0.0
    assert result.info["evaluation"]["unsupported_claim_penalty"] > 0.0
