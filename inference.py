"""Reproducible inference runner for the support queue OpenEnv."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from support_queue_env.client import SupportQueueEnvClient
from support_queue_env.models import CustomerSupportAction, CustomerSupportObservation
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
Submit a final resolution once the ticket is fully handled.
""".strip()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


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


def _model_action(client: OpenAI, model_name: str, observation: CustomerSupportObservation) -> CustomerSupportAction:
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


def run_episode(client: OpenAI, model_name: str, task_id: str) -> dict[str, Any]:
    env = SupportQueueEnvClient()
    observation = env.reset(task_id=task_id)
    total_reward = 0.0
    print(f"[START] Task={task_id}")
    done = False

    while not done:
        action = _model_action(client, model_name, observation)
        step_result = env.step(action)
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
    api_key = _require_env("OPENAI_API_KEY")
    model_name = _require_env("MODEL_NAME")
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    return [run_episode(client, model_name, task_id) for task_id in TASK_ORDER]


if __name__ == "__main__":
    main()
