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
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> dict:
    try:
        result = env.reset(task_id=payload.task_id if payload else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump(mode="json")


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
