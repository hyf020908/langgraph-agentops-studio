from __future__ import annotations

# JSON serialization helpers.
# Artifact export and state snapshots pass through here so Pydantic models,
# LangChain messages, dataclasses, and paths can be serialized consistently.

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


def json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, BaseMessage):
        # Message objects are reduced to the fields that matter for replay/debug.
        return {
            "type": value.type,
            "content": value.content,
            "name": getattr(value, "name", None),
            "tool_calls": getattr(value, "tool_calls", None),
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def dumps(payload: Any, *, indent: int = 2) -> str:
    return json.dumps(payload, default=json_default, ensure_ascii=False, indent=indent)


def to_jsonable(payload: Any) -> Any:
    # Round-tripping through JSON strips non-serializable runtime objects before
    # state is written to disk or passed into export helpers.
    return json.loads(dumps(payload))
