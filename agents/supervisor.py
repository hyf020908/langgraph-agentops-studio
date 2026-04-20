from __future__ import annotations

# Supervisor node for the outer graph loop.
# This node does not inspect prompts or call providers; it reads accumulated
# state and delegates routing to the policy in `graph.routing`.

from graph.routing import determine_next_step
from services.runtime import AgentRuntime


def build_supervisor_node(runtime: AgentRuntime):
    def supervisor(state):
        # Routing is derived from current state only, which makes the control
        # policy deterministic and easy to test outside the graph runtime.
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
