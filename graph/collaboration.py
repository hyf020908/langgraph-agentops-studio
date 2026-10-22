"""Checkpointed fan-out/fan-in with optional sequential peer handoffs."""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from schemas.collaboration import CollaborationConfig, SpecialistResult
from schemas.state import AgentState, replace_records
from services.collaboration import run_specialist


def build_collaboration_subgraph(runtime):
    def dispatch(state):
        config = CollaborationConfig.model_validate(state["collaboration"])
        results = [
            SpecialistResult.model_validate(item)
            for item in state.get("collaboration_results", [])
        ]
        completed = {item.agent for item in results}
        pending = [agent for agent in config.agents if agent.name not in completed]
        if not pending:
            return "finish_collaboration"
        selected = pending[:1] if config.mode == "sequential" else pending
        return [
            Send(
                "specialist",
                {
                    "specialist": agent.model_dump(),
                    "user_request": state["user_request"],
                    "tool_budget": config.max_tool_calls,
                    "plan": [
                        step.model_dump() if hasattr(step, "model_dump") else step
                        for step in state.get("plan", [])
                    ],
                    "peer_results": [item.model_dump() for item in results]
                    if config.mode == "sequential"
                    else [],
                },
            )
            for agent in selected
        ]

    def finish(state):
        config = CollaborationConfig.model_validate(state["collaboration"])
        by_name = {
            SpecialistResult.model_validate(item).agent: item
            for item in state.get("collaboration_results", [])
        }
        return {
            "collaboration_complete": True,
            "collaboration_results": replace_records(
                [by_name[agent.name] for agent in config.agents]
            ),
            "execution_trace": [
                runtime.trace(
                    "collaboration",
                    "completed",
                    "Specialist results handed to research.",
                    {"mode": config.mode, "agents": len(config.agents)},
                )
            ],
        }

    graph = StateGraph(AgentState)
    graph.add_node("dispatch_specialists", lambda state: {})
    graph.add_node("specialist", lambda state: run_specialist(runtime, state))
    graph.add_node("finish_collaboration", finish)
    graph.add_edge(START, "dispatch_specialists")
    graph.add_conditional_edges("dispatch_specialists", dispatch)
    graph.add_edge("specialist", "dispatch_specialists")
    graph.add_edge("finish_collaboration", END)
    return graph.compile(name="specialist_collaboration")
