from __future__ import annotations

from langchain_core.messages import AIMessage

from schemas.models import FindingRecord
from services.runtime import AgentRuntime
from tools.factory import ToolRegistry


def build_analyst_node(runtime: AgentRuntime, tools: ToolRegistry):
    def analyst_agent(state):
        findings = runtime.reasoning.analyze_evidence(
            user_request=state["user_request"],
            ranked_evidence=state.get("ranked_evidence", []),
            revision_count=state.get("revision_count", 0),
        )

        recommendation = runtime.recommendation_service.synthesize(
            user_request=state["user_request"],
            findings=findings,
            ranked_evidence=state.get("ranked_evidence", []),
            evidence_assessments=state.get("evidence_assessments", []),
            coverage_record=state.get("coverage_record"),
        )

        governance = runtime.governance_service.evaluate(
            recommendation=recommendation,
            coverage_record=state.get("coverage_record"),
            evidence_assessments=state.get("evidence_assessments", []),
            conflicts=state.get("evidence_conflicts", []),
            task_type=state.get("task_type", "general"),
            user_request=state["user_request"],
        )

        report_payload = tools.invoke(
            "report_writer_tool",
            {
                "user_request": state["user_request"],
                "plan": [step.model_dump() for step in state.get("plan", [])],
                "findings": [finding.model_dump() for finding in findings],
                "ranked_evidence": [record.model_dump() for record in state.get("ranked_evidence", [])],
                "evidence_assessments": [record.model_dump() for record in state.get("evidence_assessments", [])],
                "evidence_conflicts": [record.model_dump() for record in state.get("evidence_conflicts", [])],
                "evidence_supports": [record.model_dump() for record in state.get("evidence_supports", [])],
                "coverage_record": state.get("coverage_record").model_dump() if state.get("coverage_record") else None,
                "acceptance_criteria": state.get("acceptance_criteria", []),
                "review_feedback_summary": state.get("review_feedback").summary if state.get("review_feedback") else None,
                "recommendation": recommendation.model_dump(),
                "governance_evaluation": governance.model_dump(),
            },
        )

        trace = runtime.trace(
            node="analyst_agent",
            status="completed",
            message="Synthesized findings, recommendation, and governance evaluation.",
            metadata={
                "finding_count": len(findings),
                "recommendation_type": recommendation.recommendation_type,
                "requires_human_review": governance.requires_human_review,
            },
        )
        finding_summary = "\n".join(f"- {finding.theme}: {finding.insight}" for finding in findings)
        return {
            "findings": [FindingRecord.model_validate(item) for item in findings],
            "recommendation": recommendation,
            "governance_evaluation": governance,
            "draft_report": report_payload["draft_report"],
            "human_approval_required": governance.requires_human_review,
            "status": "draft_ready",
            "sender": "analyst_agent",
            "messages": [AIMessage(content=f"Analysis complete.\n{finding_summary}")],
            "execution_trace": [trace],
            "error_info": None,
        }

    return analyst_agent
