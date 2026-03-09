from __future__ import annotations

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
    return json.loads(dumps(payload))

