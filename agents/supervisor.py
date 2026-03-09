from __future__ import annotations

from graph.routing import determine_next_step
from services.runtime import AgentRuntime


def build_supervisor_node(runtime: AgentRuntime):
    def supervisor(state):
        next_step = determine_next_step(state=state, settings=runtime.settings)
        status = state.get("status", "running") if next_step == "end" else f"routing_to_{next_step}"
        trace = runtime.trace(
            node="supervisor",
            status="routed",
            message="Supervisor evaluated current state and selected the next step.",
            metadata={
                "next_step": next_step,
                "retry_count": state.get("retry_count", 0),
                "revision_count": state.get("revision_count", 0),
            },
        )
        return {
            "next_step": next_step,
            "sender": "supervisor",
            "status": status,
            "execution_trace": [trace],
        }

    return supervisor
