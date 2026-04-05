"""FastAPI server for the customer support environment."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..models import CustomerSupportAction
from .support_queue_environment import SupportQueueEnvironment


env = SupportQueueEnvironment()
app = FastAPI(title="Support Queue OpenEnv", version="0.1.0")


class ResetRequest(BaseModel):
    task_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "support_queue_env",
        "description": "Customer support triage and resolution environment for OpenEnv.",
        "max_steps": 12,
        "tasks": [
            {"task_id": "delayed_shipping_refund", "difficulty": "easy"},
            {"task_id": "defective_return_window", "difficulty": "medium"},
            {"task_id": "subscription_cancellation_dispute", "difficulty": "hard"},
        ],
        "action_space": [
            "search_policy",
            "open_order",
            "open_account",
            "open_log",
            "set_priority",
            "route_ticket",
            "add_tag",
            "draft_reply",
            "submit_resolution",
        ],
    }


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> dict:
    try:
        result = env.reset(task_id=payload.task_id if payload else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"observation": result.model_dump(mode="json")}


@app.post("/step")
def step(action: CustomerSupportAction) -> dict:
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@app.get("/state")
def state() -> dict:
    try:
        return env.state().model_dump(mode="json")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
