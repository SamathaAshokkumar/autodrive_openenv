"""Compatibility layer for environments where OpenEnv is not installed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self) -> Dict[str, Any]:
            return self.__dict__.copy()

    def Field(default=None, **kwargs):
        return default

try:
    from openenv.core.env_server.types import Action, Observation, State
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.http_server import create_app
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        reward: Optional[float] = None
        done: bool = False

    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0

    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS: bool = False

    TAction = TypeVar("TAction")
    TObservation = TypeVar("TObservation")
    TState = TypeVar("TState")
    TResult = TypeVar("TResult")

    @dataclass
    class StepResult(Generic[TResult]):
        observation: TResult
        reward: Optional[float]
        done: bool

    class EnvClient(Generic[TAction, TObservation, TState]):
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenEnv is not installed. Install openenv-core to use the typed client.")

    class _FallbackApp:
        def get(self, *args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

    def create_app(*args, **kwargs):
        return _FallbackApp()
