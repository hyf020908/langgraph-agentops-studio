from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.analyst import build_analyst_node
from agents.executor import build_executor_node
from agents.planner import build_initialize_node, build_planner_node
from agents.reviewer import build_human_review_node, build_reviewer_node
from agents.supervisor import build_supervisor_node
from graph.research_subgraph import build_research_subgraph
from graph.routing import supervisor_router
from schemas.state import AgentState
from services.checkpoint import build_checkpointer
from services.runtime import AgentRuntime
from tools.factory import build_tool_registry


def build_agent_graph(runtime: AgentRuntime):
    tools = build_tool_registry(runtime)
    research_subgraph = build_research_subgraph(runtime, tools)

    graph = StateGraph(AgentState)
    graph.add_node("initialize_task", build_initialize_node(runtime))
    graph.add_node("planner_agent", build_planner_node(runtime))
    graph.add_node("supervisor", build_supervisor_node(runtime))
    graph.add_node("research_pipeline", research_subgraph)
    graph.add_node("analyst_agent", build_analyst_node(runtime, tools))
    graph.add_node("reviewer_agent", build_reviewer_node(runtime, tools))
    graph.add_node("human_review", build_human_review_node(runtime))
    graph.add_node("executor_agent", build_executor_node(runtime, tools))

    graph.add_edge(START, "initialize_task")
    graph.add_edge("initialize_task", "planner_agent")
    graph.add_edge("planner_agent", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "research_pipeline": "research_pipeline",
            "analyst_agent": "analyst_agent",
            "human_review": "human_review",
            "executor_agent": "executor_agent",
            "end": END,
        },
    )
    graph.add_edge("research_pipeline", "supervisor")
    graph.add_edge("analyst_agent", "reviewer_agent")
    graph.add_edge("executor_agent", "supervisor")

    return graph.compile(checkpointer=build_checkpointer(runtime.settings), name="langgraph_agentops_studio")
