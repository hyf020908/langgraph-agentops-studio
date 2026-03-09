from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from schemas.models import EvidenceRecord, FindingRecord, PlanStep, ReviewFeedback
from services.config import LLMSettings


class BaseLLMProvider(Protocol):
    name: str

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(slots=True)
class OpenAICompatibleProvider:
    settings: LLMSettings
    default_base_url: str | None = None
    name: str = "openai_compatible"
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for provider-backed LLM calls.") from exc

        if not self.settings.api_key:
            raise ValueError("LLM_API_KEY is required for the configured LLM provider.")

        base_url = self.settings.base_url or self.default_base_url
        client_kwargs: dict[str, Any] = {
            "api_key": self.settings.api_key,
            "timeout": self.settings.timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if self.settings.extra_headers:
            client_kwargs["default_headers"] = self.settings.extra_headers
        self._client = OpenAI(**client_kwargs)

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **self.settings.extra_kwargs,
        )
        content = response.choices[0].message.content or ""
        return content.strip()


@dataclass(slots=True)
class OpenAIProvider(OpenAICompatibleProvider):
    name: str = "openai"


@dataclass(slots=True)
class DeepSeekProvider(OpenAICompatibleProvider):
    name: str = "deepseek"

    def __post_init__(self) -> None:
        self.default_base_url = self.default_base_url or "https://api.deepseek.com/v1"
        super().__post_init__()


def build_llm_provider(settings: LLMSettings) -> BaseLLMProvider:
    provider = settings.provider
    if provider == "openai":
        return OpenAIProvider(settings=settings)
    if provider == "deepseek":
        return DeepSeekProvider(settings=settings)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(settings=settings)
    raise ValueError(f"Unsupported LLM provider: {provider}")


class BaseReasoningEngine(Protocol):
    def plan_task(self, user_request: str) -> tuple[list[PlanStep], list[str], list[str]]:
        ...

    def analyze_evidence(
        self,
        user_request: str,
        ranked_evidence: list[EvidenceRecord],
        revision_count: int,
    ) -> list[FindingRecord]:
        ...

    def review_report(
        self,
        draft_report: str,
        ranked_evidence: list[EvidenceRecord],
        revision_count: int,
        human_approval_required: bool,
    ) -> ReviewFeedback:
        ...


class ProviderReasoningEngine:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider
        self.logger = logging.getLogger("agentops")

    def plan_task(self, user_request: str) -> tuple[list[PlanStep], list[str], list[str]]:
        payload = {
            "task": user_request,
            "requirements": [
                "Return exactly 4 plan steps with IDs like P1/P2.",
                "Return 3 acceptance criteria.",
                "Return 3 retrieval-focused search queries.",
            ],
            "response_schema": {
                "plan": [
                    {
                        "step_id": "P1",
                        "objective": "...",
                        "owner": "planner_agent",
                        "done_definition": "...",
                        "dependencies": ["P0"],
                    }
                ],
                "acceptance_criteria": ["..."],
                "search_queries": ["..."],
            },
        }
        model_payload = self._complete_json(
            system_prompt="You are a workflow planner for a LangGraph multi-agent system. Return strict JSON only.",
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )

        plan_items = model_payload.get("plan", [])
        acceptance = [str(item).strip() for item in model_payload.get("acceptance_criteria", []) if str(item).strip()]
        queries = [str(item).strip() for item in model_payload.get("search_queries", []) if str(item).strip()]

        plan = [
            PlanStep(
                step_id=str(item.get("step_id", f"P{index}")),
                objective=str(item.get("objective", "")).strip(),
                owner=str(item.get("owner", "planner_agent")),
                done_definition=str(item.get("done_definition", "")).strip(),
                dependencies=[str(dep) for dep in item.get("dependencies", [])],
            )
            for index, item in enumerate(plan_items, start=1)
            if item.get("objective") and item.get("done_definition")
        ]

        if len(plan) < 2 or len(queries) < 1:
            raise RuntimeError("LLM planning response did not satisfy minimum schema constraints.")
        return plan[:4], acceptance[:4], queries[:4]

    def analyze_evidence(
        self,
        user_request: str,
        ranked_evidence: list[EvidenceRecord],
        revision_count: int,
    ) -> list[FindingRecord]:
        serialized_evidence = [
            {
                "evidence_id": item.evidence_id,
                "claim": item.claim,
                "confidence": item.confidence,
                "risk_flags": item.risk_flags,
                "summary": item.summary,
                "citations": item.citations,
            }
            for item in ranked_evidence[:12]
        ]
        payload = {
            "task": user_request,
            "revision_count": revision_count,
            "ranked_evidence": serialized_evidence,
            "response_schema": {
                "findings": [
                    {
                        "finding_id": "F1",
                        "theme": "...",
                        "insight": "...",
                        "rationale": "...",
                        "evidence_ids": ["EVD-01"],
                        "risk_level": "low|medium|high",
                    }
                ]
            },
        }
        model_payload = self._complete_json(
            system_prompt="You are an analyst agent. Return strict JSON findings grounded in the provided evidence.",
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )

        findings_payload = model_payload.get("findings", [])
        findings: list[FindingRecord] = []
        for index, item in enumerate(findings_payload, start=1):
            findings.append(
                FindingRecord(
                    finding_id=str(item.get("finding_id", f"F{index}")),
                    theme=str(item.get("theme", "Key Finding")).strip(),
                    insight=str(item.get("insight", "")).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                    evidence_ids=[str(eid) for eid in item.get("evidence_ids", [])],
                    risk_level=str(item.get("risk_level", "medium")).lower(),
                )
            )
        if not findings:
            raise RuntimeError("LLM analysis response did not include valid findings.")
        return findings[:6]

    def review_report(
        self,
        draft_report: str,
        ranked_evidence: list[EvidenceRecord],
        revision_count: int,
        human_approval_required: bool,
    ) -> ReviewFeedback:
        payload = {
            "revision_count": revision_count,
            "human_approval_required": human_approval_required,
            "ranked_evidence_count": len(ranked_evidence),
            "draft_excerpt": draft_report[:4000],
            "response_schema": {
                "verdict": "approve|revise|escalate",
                "score": 0.0,
                "summary": "...",
                "questions": ["..."],
                "revision_requests": ["..."],
                "major_risks": ["..."],
            },
        }
        model_payload = self._complete_json(
            system_prompt="You are a governance reviewer. Return strict JSON with verdict and actionable feedback.",
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )

        return ReviewFeedback(
            verdict=str(model_payload.get("verdict", "revise")).lower(),
            score=float(model_payload.get("score", 0.65)),
            summary=str(model_payload.get("summary", "Reviewer response was incomplete.")).strip(),
            questions=[str(item) for item in model_payload.get("questions", [])],
            revision_requests=[str(item) for item in model_payload.get("revision_requests", [])],
            major_risks=[str(item) for item in model_payload.get("major_risks", [])],
        )

    def _complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw = self.provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json(raw)
        if payload is None:
            self.logger.error("Could not parse strict JSON from provider response.")
            raise RuntimeError("Provider response is not valid JSON for the requested schema.")
        return payload


def build_reasoning_engine(provider: BaseLLMProvider) -> BaseReasoningEngine:
    return ProviderReasoningEngine(provider=provider)


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None
