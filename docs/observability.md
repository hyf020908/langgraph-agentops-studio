# Runtime Observability

The API exposes checkpoint-backed progress without bypassing the compiled
LangGraph workflow. `POST /runs` remains synchronous and backward compatible.
Clients that generate a `task_id` before starting the POST may poll
`GET /runs/{task_id}` concurrently. The GET handler calls
`graph.get_state({"configurable": {"thread_id": task_id}})` and never advances
the graph.

Before the first checkpoint exists, and for unknown thread IDs, GET returns
`404`. Clients should treat a 404 received while their own POST is still
pending as a short-lived `waiting` state.

## Run response ledgers

Every create, inspect, and continue response uses the same additive fields:

- `execution_trace`: node-level state transitions emitted by the workflow.
- `tool_call_history`: actual `ToolNode` message exchanges and direct registry
  invocations, including bounded inputs, output previews, status, and duration.
- `model_call_history`: provider-boundary model invocations with node,
  provider/model, input/output previews, status, duration, and truncation
  metadata. Model output previews are bounded at 12,000 characters.
- `next_nodes`: node names from the latest LangGraph `StateSnapshot.next`.
- `draft_report`: the current report during analysis, revision, or human review.
- `final_report`: complete markdown only after the workflow reaches
  `status=completed`.

The histories are stored in `AgentState` with append reducers, so they are part
of checkpoints and survive interrupt/resume. The model recorder uses a
`ContextVar` to isolate concurrent runs that share the process-wide runtime.
Research tool records are derived from the real `AIMessage.tool_calls` and
`ToolMessage` outputs produced by LangGraph `ToolNode`; synchronous report,
review, and export tools are recorded at the registry invocation boundary.

## Detailed conclusion controls

The recommendation remains deterministic and governance-safe. Its rationale is
organized into key findings, evidence basis, alternatives/trade-offs, risk
mitigations, and next actions. Reporting limits only control this presentation
projection; the complete `residual_risks` ledger remains unchanged for
governance evaluation.

Configuration:

```env
LLM_MAX_TOKENS=2200
REPORT_DETAIL_LEVEL=detailed
REPORT_MAX_FINDINGS_IN_CONCLUSION=5
REPORT_MAX_EVIDENCE_IN_CONCLUSION=6
REPORT_MAX_RISKS_IN_CONCLUSION=5
REPORT_INCLUDE_NEXT_ACTIONS=true
```

## LangGraph references

- [Persistence and `graph.get_state`](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Streaming graph state, custom data, and subgraphs](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Interrupt and checkpoint semantics](https://docs.langchain.com/oss/python/langgraph/interrupts)

