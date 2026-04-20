from __future__ import annotations

# Supervisor routing helpers.
# These functions centralize the top-level transition rules so the graph wiring
# stays declarative while the routing policy remains easy to inspect.

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
        # Governance can request a manual checkpoint before export proceeds.
        return "human_review"

    if state.get("artifacts"):
        # Artifact creation is the terminal signal for the outer graph.
        return "end"
    if state.get("error_info") and not state["error_info"].recoverable:
        # Unrecoverable failures skip back to the stage most likely to produce a
        # user-visible outcome: analysis if findings exist, otherwise export.
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
    # LangGraph conditional edges read the route that the supervisor already
    # stored in state, rather than recomputing policy in the router callback.
    return state.get("next_step", "end")
