from __future__ import annotations

from services.config import Settings


def determine_next_step(state, settings: Settings) -> str:
    governance = state.get("governance_evaluation")
    approval = state.get("approval_decision")
    requires_human_review = False
    if isinstance(governance, dict):
        requires_human_review = bool(governance.get("requires_human_review", False))
    elif governance is not None:
        requires_human_review = bool(getattr(governance, "requires_human_review", False))
    if requires_human_review and not approval:
        return "human_review"

    if state.get("artifacts"):
        return "end"
    if state.get("error_info") and not state["error_info"].recoverable:
        return "executor_agent" if not state.get("findings") else "analyst_agent"
    if state.get("error_info") and state.get("retry_count", 0) <= settings.max_retries:
        return "research_pipeline"
    if not state.get("ranked_evidence"):
        if state.get("retry_count", 0) > settings.max_retries:
            return "executor_agent"
        return "research_pipeline"
    if not state.get("findings") or not state.get("draft_report"):
        return "analyst_agent"
    return "executor_agent"


def supervisor_router(state) -> str:
    return state.get("next_step", "end")
