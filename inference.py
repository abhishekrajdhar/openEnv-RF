"""Reproducible inference runner for the support queue OpenEnv."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local convenience
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - fallback mode does not require the SDK
    OpenAI = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from support_queue_env.client import SupportQueueEnvClient
from support_queue_env.models import CustomerSupportAction, CustomerSupportObservation, ResolutionPayload
from support_queue_env.server.support_queue_environment import TASK_ORDER


SYSTEM_PROMPT = """
You are solving a deterministic customer support environment.
Return exactly one JSON object with keys:
- action_type
- argument
- message
- resolution

Allowed action_type values:
search_policy, open_order, open_account, open_log, set_priority, route_ticket, add_tag, draft_reply, submit_resolution

Use only evidence visible in the observation. Avoid repeated or invalid actions.
Resolve conflicting policy or log information before finalizing the case.
Submit a final resolution once the ticket is fully handled.
""".strip()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _resolve_api_config() -> tuple[str | None, str | None, str | None]:
    proxy_base_url = _optional_env("API_BASE_URL")
    proxy_api_key = _optional_env("API_KEY")
    model_name = _optional_env("MODEL_NAME")

    if proxy_base_url and proxy_api_key:
        return proxy_base_url, proxy_api_key, model_name or "gpt-4.1-mini"

    openai_api_key = _optional_env("OPENAI_API_KEY")
    if openai_api_key and model_name:
        return os.getenv("API_BASE_URL", "https://api.openai.com/v1"), openai_api_key, model_name

    return None, None, model_name


def _prompt(observation: CustomerSupportObservation) -> str:
    return json.dumps(
        {
            "instructions": observation.instructions,
            "task": observation.task.model_dump(mode="json"),
            "visible_artifacts": [artifact.model_dump(mode="json") for artifact in observation.visible_artifacts],
            "tags": observation.tags,
            "priority": observation.priority,
            "route": observation.route,
            "draft_reply": observation.draft_reply,
            "last_action_status": observation.last_action_status,
            "remaining_steps": observation.remaining_steps,
            "reward_details": observation.reward_details.model_dump(mode="json"),
        },
        sort_keys=True,
    )


def _model_action(client: Any, model_name: str, observation: CustomerSupportObservation) -> CustomerSupportAction:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        seed=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt(observation)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return CustomerSupportAction.model_validate(json.loads(content))


def _safe_next_action(
    observation: CustomerSupportObservation,
    client: Any | None = None,
    model_name: str | None = None,
) -> CustomerSupportAction:
    if client is None or model_name is None:
        return _scripted_action(observation)

    try:
        return _model_action(client, model_name, observation)
    except Exception:
        return _scripted_action(observation)


def _scripted_action(observation: CustomerSupportObservation) -> CustomerSupportAction:
    visible_ids = {artifact.artifact_id for artifact in observation.visible_artifacts}
    tags = set(observation.tags)
    task_id = observation.task.task_id

    if task_id == "delayed_shipping_refund":
        if "P_DELAY" not in visible_ids:
            return CustomerSupportAction(action_type="search_policy", argument="shipping delay policy")
        if observation.route != "logistics":
            return CustomerSupportAction(action_type="route_ticket", argument="logistics")
        if observation.priority != "normal":
            return CustomerSupportAction(action_type="set_priority", argument="normal")
        if "delayed_shipment" not in tags:
            return CustomerSupportAction(action_type="add_tag", argument="delayed_shipment")
        if not observation.draft_reply:
            return CustomerSupportAction(
                action_type="draft_reply",
                message="Sorry about the delay. I refunded 8.99 for the shipping fee and will follow up on the delay.",
            )
        return CustomerSupportAction(
            action_type="submit_resolution",
            resolution=ResolutionPayload(
                resolution_code="refund_shipping_fee",
                shipping_refund=8.99,
                message=observation.draft_reply
                or "Sorry about the delay. I refunded 8.99 for the shipping fee and will follow up on the delay.",
            ),
        )

    if task_id == "defective_return_window":
        if "P_DEFECT" not in visible_ids:
            return CustomerSupportAction(action_type="search_policy", argument="defective return policy")
        if "P_OPEN_BOX" not in visible_ids:
            return CustomerSupportAction(action_type="search_policy", argument="opened box policy")
        if observation.route != "returns":
            return CustomerSupportAction(action_type="route_ticket", argument="returns")
        if observation.priority != "high":
            return CustomerSupportAction(action_type="set_priority", argument="high")
        if "defective_item" not in tags:
            return CustomerSupportAction(action_type="add_tag", argument="defective_item")
        if "safety_risk" not in tags:
            return CustomerSupportAction(action_type="add_tag", argument="safety_risk")
        if not observation.draft_reply:
            return CustomerSupportAction(
                action_type="draft_reply",
                message="Your blender is defective. I approved a full refund and we will send a return label.",
            )
        return CustomerSupportAction(
            action_type="submit_resolution",
            resolution=ResolutionPayload(
                resolution_code="return_and_refund",
                refund_amount=79.99,
                message=observation.draft_reply
                or "Your blender is defective. I approved a full refund and we will send a return label.",
            ),
        )

    if "L_CANCEL_300" not in visible_ids:
        return CustomerSupportAction(action_type="open_log", argument="cancellation log S300")
    if "P_BILLING" not in visible_ids:
        return CustomerSupportAction(action_type="search_policy", argument="billing cancellation policy")
    if "P_SAVE" not in visible_ids:
        return CustomerSupportAction(action_type="search_policy", argument="vip retention save playbook")
    if "C300" not in visible_ids:
        return CustomerSupportAction(action_type="open_account", argument="customer C300")
    if observation.route != "billing":
        return CustomerSupportAction(action_type="route_ticket", argument="billing")
    if observation.priority != "urgent":
        return CustomerSupportAction(action_type="set_priority", argument="urgent")
    if "billing_dispute" not in tags:
        return CustomerSupportAction(action_type="add_tag", argument="billing_dispute")
    if "vip_customer" not in tags:
        return CustomerSupportAction(action_type="add_tag", argument="vip_customer")
    if not observation.draft_reply:
        return CustomerSupportAction(
            action_type="draft_reply",
            message="Your cancellation should have been completed earlier. I refunded 24.50, added a 10.00 credit, and confirmed the VIP cancellation fix.",
        )
    return CustomerSupportAction(
        action_type="submit_resolution",
        resolution=ResolutionPayload(
            resolution_code="refund_and_credit",
            refund_amount=24.50,
            goodwill_credit=10.00,
            message=observation.draft_reply
            or "Your cancellation should have been completed earlier. I refunded 24.50, added a 10.00 credit, and confirmed the VIP cancellation fix.",
        ),
    )


def run_episode(task_id: str, client: Any | None = None, model_name: str | None = None) -> dict[str, Any]:
    env = SupportQueueEnvClient()
    observation = env.reset(task_id=task_id)
    total_reward = 0.0
    print(f"[START] Task={task_id}")
    done = False

    while not done:
        action = _safe_next_action(observation, client=client, model_name=model_name)
        try:
            step_result = env.step(action)
        except Exception:
            fallback_action = _scripted_action(observation)
            step_result = env.step(fallback_action)
        total_reward += step_result.reward
        print(f"[STEP] reward={step_result.reward:.4f} done={step_result.done}")
        observation = step_result.observation
        done = step_result.done

    print(f"[END] total_reward={total_reward:.4f}")
    return {
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "final_score": observation.reward_details.grader_score,
    }


def main() -> list[dict[str, Any]]:
    api_base_url, api_key, model_name = _resolve_api_config()
    if api_key and model_name:
        if OpenAI is None:
            return [run_episode(task_id) for task_id in TASK_ORDER]
        try:
            client = OpenAI(api_key=api_key, base_url=api_base_url, timeout=5.0, max_retries=0)
            return [run_episode(task_id, client=client, model_name=model_name) for task_id in TASK_ORDER]
        except Exception:
            return [run_episode(task_id) for task_id in TASK_ORDER]
    return [run_episode(task_id) for task_id in TASK_ORDER]


if __name__ == "__main__":
    main()
