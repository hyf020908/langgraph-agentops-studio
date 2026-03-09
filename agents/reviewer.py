from __future__ import annotations

from langgraph.types import Command, interrupt

from schemas.models import ApprovalDecision
from services.runtime import AgentRuntime
from tools.factory import ToolRegistry


def build_reviewer_node(runtime: AgentRuntime, tools: ToolRegistry):
    def reviewer_agent(state):
        governance = state.get("governance_evaluation")
        confidence_gate = (
            governance.recommendation_confidence if governance else state.get("recommendation").confidence_level
        ) if state.get("recommendation") else 0.0
        human_gate = bool(governance.requires_human_review) if governance else state.get("human_approval_required", False)

        feedback = runtime.reasoning.review_report(
            draft_report=state.get("draft_report", ""),
            ranked_evidence=state.get("ranked_evidence", []),
            revision_count=state.get("revision_count", 0),
            human_approval_required=human_gate,
        )
        tools.invoke(
            "review_formatter_tool",
            {
                "review_feedback": feedback.model_dump(),
                "task_id": state["task_id"],
            },
        )

        must_escalate = human_gate or confidence_gate < runtime.settings.risk_threshold_for_human_review

        trace = runtime.trace(
            node="reviewer_agent",
            status=feedback.verdict,
            message="Reviewer evaluated the draft report.",
            metadata={
                "score": feedback.score,
                "verdict": feedback.verdict,
                "must_escalate": must_escalate,
                "triggered_policies": governance.triggered_policies if governance else [],
            },
        )
        update = {
            "review_feedback": feedback,
            "reviewer_history": [feedback],
            "status": f"review_{feedback.verdict}",
            "sender": "reviewer_agent",
            "execution_trace": [trace],
        }
        if feedback.verdict == "revise" and state.get("revision_count", 0) < runtime.settings.max_revisions:
            update["revision_count"] = state.get("revision_count", 0) + 1
            return Command(update=update, goto="analyst_agent")
        if feedback.verdict == "escalate" or must_escalate:
            update["human_approval_required"] = True
            return Command(update=update, goto="human_review")
        return Command(update=update, goto="executor_agent")

    return reviewer_agent


def build_human_review_node(runtime: AgentRuntime):
    def human_review(state):
        governance = state.get("governance_evaluation")
        decision_payload = interrupt(
            {
                "task_id": state["task_id"],
                "status": state.get("status"),
                "review_summary": state.get("review_feedback").summary if state.get("review_feedback") else None,
                "major_risks": state.get("review_feedback").major_risks if state.get("review_feedback") else [],
                "triggered_policies": governance.triggered_policies if governance else [],
                "risk_summary": governance.risk_summary if governance else "",
                "evidence_gaps": governance.evidence_gaps if governance else [],
                "contradiction_summary": governance.contradiction_summary if governance else "",
                "recommendation_confidence": governance.recommendation_confidence if governance else None,
                "required_human_action": governance.required_human_action if governance else "Provide approval decision.",
            }
        )
        decision = ApprovalDecision(
            approved=bool(decision_payload.get("approved", False)),
            reviewer=decision_payload.get("reviewer", runtime.settings.default_reviewer),
            rationale=decision_payload.get("rationale", "Human approval decision recorded."),
        )
        trace = runtime.trace(
            node="human_review",
            status="approved" if decision.approved else "rejected",
            message="Human review resumed the graph from an interrupt.",
            metadata=decision.model_dump(),
        )
        update = {
            "approval_decision": decision,
            "human_approval_required": False,
            "sender": "human_review",
            "execution_trace": [trace],
        }
        if decision.approved:
            update["status"] = "approved_for_export"
            return Command(update=update, goto="executor_agent")
        update["status"] = "human_requested_revision"
        update["revision_count"] = state.get("revision_count", 0) + 1
        return Command(update=update, goto="analyst_agent")

    return human_review
