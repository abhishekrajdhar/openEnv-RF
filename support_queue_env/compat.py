"""Compatibility helpers for local development without OpenEnv installed."""

from __future__ import annotations

from typing import Any, Dict, Generic, TypeVar

from pydantic import BaseModel, Field

try:
    from openenv.core.env import Environment as OpenEnvEnvironment
except Exception:  # pragma: no cover - exercised only when OpenEnv is available
    OpenEnvEnvironment = object


class OpenEnvModel(BaseModel):
    """Shared Pydantic base class."""

    model_config = {"extra": "forbid"}


class Action(OpenEnvModel):
    """Fallback action base model."""


class Observation(OpenEnvModel):
    """Fallback observation base model."""


class State(OpenEnvModel):
    """Fallback state base model."""

    episode_id: str
    step_count: int = 0
    done: bool = False


ObservationT = TypeVar("ObservationT", bound=Observation)


class StepResult(OpenEnvModel, Generic[ObservationT]):
    """Fallback step result matching the Gym-style contract."""

    observation: ObservationT
    reward: float = Field(gt=0.0001, lt=0.9999)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class Environment(OpenEnvEnvironment):
    """Fallback Environment base with the standard API."""

    def reset(self, *args: Any, **kwargs: Any) -> Observation:
        raise NotImplementedError

    def step(self, action: Action) -> StepResult[Any]:
        raise NotImplementedError

    def state(self) -> State:
        raise NotImplementedError


class LocalEnvClient:
    """Simple in-process client used for local testing and baseline runs."""

    def __init__(self, env: Environment):
        self._env = env

    def reset(self, **kwargs: Any) -> Observation:
        return self._env.reset(**kwargs)

    def step(self, action: Action) -> StepResult[Any]:
        return self._env.step(action)

    def state(self) -> State:
        return self._env.state()

    def close(self) -> None:
        return None
