from __future__ import annotations

# Planning-stage nodes.
# The planner turns a free-form user request into an explicit execution plan,
# acceptance criteria, and research queries that seed the rest of the graph.

from langchain_core.messages import AIMessage

from services.runtime import AgentRuntime


def build_initialize_node(runtime: AgentRuntime):
    def initialize_task(state):
        # This node records the first trace event inside the compiled graph; the
        # initial state already exists before LangGraph starts executing.
        trace = runtime.trace(
            node="initialize_task",
            status="completed",
            message="Task initialized and ready for planning.",
            metadata={"task_id": state["task_id"]},
        )
        return {
            "status": "ready_for_planning",
            "sender": "initialize_task",
            "execution_trace": [trace],
        }

    return initialize_task


def build_planner_node(runtime: AgentRuntime):
    def planner_agent(state):
        # Planning is the only stage that directly expands the user's task into
        # a structured workflow contract for the downstream research loop.
        plan, acceptance_criteria, queries = runtime.reasoning.plan_task(state["user_request"])
        trace = runtime.trace(
            node="planner_agent",
            status="completed",
            message="Generated plan, acceptance criteria, and initial research queries.",
            metadata={"query_count": len(queries)},
        )
        plan_summary = "\n".join(f"- {step.step_id}: {step.objective}" for step in plan)
        return {
            "plan": plan,
            "acceptance_criteria": acceptance_criteria,
            "search_queries": queries,
            "status": "planned",
            "next_step": "research_pipeline",
            "sender": "planner_agent",
            "messages": [AIMessage(content=f"Planning complete.\n{plan_summary}")],
            "execution_trace": [trace],
            "error_info": None,
        }

    return planner_agent
