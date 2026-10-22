"""MCP configuration stores environment references, never credential values."""

import re
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    transport: Literal["streamable_http", "sse", "stdio"] = "streamable_http"
    url: str | None = Field(default=None, max_length=2048)
    command: str | None = Field(default=None, min_length=1, max_length=1024)
    args: list[Annotated[str, Field(max_length=4096)]] = Field(
        default_factory=list, max_length=64
    )
    env: dict[str, str] = Field(default_factory=dict, max_length=64)
    headers_env: dict[str, str] = Field(default_factory=dict, max_length=32)
    enabled: bool = True
    allowed_tools: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_transport(self):
        if self.transport == "stdio":
            if not self.command or self.url or self.headers_env:
                raise ValueError(
                    "stdio requires command and does not accept url or headers_env."
                )
        else:
            url = urlsplit(self.url or "")
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "HTTP transports require an http(s) URL without credentials, query or fragment."
                )
            if self.command or self.args or self.env:
                raise ValueError("HTTP transports do not accept command, args or env.")
        for reference in [*self.env.values(), *self.headers_env.values()]:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", reference):
                raise ValueError(
                    "Credential references must be environment variable names."
                )
        for key in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("Invalid environment key.")
        for key in self.headers_env:
            if not re.fullmatch(r"[A-Za-z0-9!#$%&'*+.^_`|~-]+", key):
                raise ValueError("Invalid header name.")
        return self


class MCPToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str = Field(min_length=1, max_length=2048)
