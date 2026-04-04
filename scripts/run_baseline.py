"""OpenAI baseline runner for the support queue environment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_queue_env.client import SupportQueueEnvClient
from support_queue_env.models import CustomerSupportAction
from support_queue_env.server.support_queue_environment import TASK_ORDER


SYSTEM_PROMPT = """
You are solving a customer support operations environment.
Return exactly one JSON object per turn with keys:
- action_type
- argument (optional string)
- message (optional string)
- resolution (optional object)

Choose from:
search_policy, open_order, open_account, open_log, set_priority, route_ticket, add_tag, draft_reply, submit_resolution

When enough evidence is available, submit_resolution with:
resolution_code, refund_amount, goodwill_credit, shipping_refund, message

Be concise and deterministic.
""".strip()


def make_prompt(result: Any) -> str:
    observation = result.observation.model_dump(mode="json")
    return json.dumps(
        {
            "instructions": observation["instructions"],
            "task": observation["task"],
            "visible_artifacts": observation["visible_artifacts"],
            "priority": observation["priority"],
            "route": observation["route"],
            "tags": observation["tags"],
            "draft_reply": observation["draft_reply"],
            "last_action_status": observation["last_action_status"],
            "remaining_steps": observation["remaining_steps"],
            "reward_details": observation["reward_details"],
        },
        indent=2,
        sort_keys=True,
    )


def model_action(client: OpenAI, model: str, result: Any) -> CustomerSupportAction:
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(result)},
        ],
    )
    payload = json.loads(response.output_text)
    return CustomerSupportAction.model_validate(payload)


def run_episode(api_client: OpenAI, model: str, task_id: str) -> dict[str, Any]:
    env = SupportQueueEnvClient()
    result = env.reset(task_id=task_id)
    trajectory_rewards = [result.reward]
    while not result.done:
        action = model_action(api_client, model, result)
        result = env.step(action)
        trajectory_rewards.append(result.reward)
    final_score = result.info["evaluation"]["final_score"]
    return {
        "task_id": task_id,
        "trajectory_rewards": trajectory_rewards,
        "cumulative_reward": sum(trajectory_rewards),
        "final_score": final_score,
    }


def main() -> None:
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    api_key = os.environ["OPENAI_API_KEY"]
    api_client = OpenAI(api_key=api_key)

    episodes = [run_episode(api_client, model, task_id) for task_id in TASK_ORDER]
    aggregate = sum(item["final_score"] for item in episodes) / len(episodes)
    output = {"model": model, "episodes": episodes, "average_final_score": round(aggregate, 4)}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
