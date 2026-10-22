"""Public contracts for bounded specialist collaboration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpecialistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    role: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    mcp_servers: list[
        Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]
    ] = Field(default_factory=list, max_length=8)


class CollaborationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["parallel", "sequential"] = "parallel"
    agents: list[SpecialistConfig] = Field(min_length=1, max_length=8)
    max_tool_calls: int = Field(default=4, ge=0, le=16)

    @model_validator(mode="after")
    def unique_names(self):
        if len({agent.name for agent in self.agents}) != len(self.agents):
            raise ValueError("Specialist names must be unique.")
        return self


class SpecialistResult(BaseModel):
    agent: str
    role: str
    status: Literal["success", "error"]
    summary: str = ""
    source_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tool"]
    server: str
    tool: str
    arguments: dict = Field(default_factory=dict)


class FinalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["final"]
    summary: str = Field(min_length=1, max_length=12000, pattern=r"\S")
    source_ids: list[str] = Field(default_factory=list, max_length=128)
