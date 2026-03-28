"""Client wrapper for local use and OpenEnv-style workflows."""

from __future__ import annotations

from .compat import LocalEnvClient
from .server.support_queue_environment import SupportQueueEnvironment


class SupportQueueEnvClient(LocalEnvClient):
    """In-process client that exposes the standard reset/step/state API."""

    def __init__(self) -> None:
        super().__init__(SupportQueueEnvironment())
