import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelCallRecord, RunResponse, ToolCallRecord, TraceEvent } from "../api/client";

type ExecutionView = "nodes" | "tools" | "models";

interface ExecutionPanelProps {
  run: RunResponse | null;
  isLoading: boolean;
  isOpen: boolean;
  onClose: () => void;
}

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value || "time unavailable";
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function statusTone(status: string) {
  const normalized = status.toLowerCase();
  if (normalized.includes("error") || normalized.includes("fail") || normalized.includes("reject")) {
    return "danger";
  }
  if (normalized.includes("wait") || normalized.includes("interrupt") || normalized.includes("review")) {
    return "warning";
  }
  if (
    normalized.includes("complete") ||
    normalized.includes("success") ||
    normalized.includes("approve") ||
    normalized.includes("observed")
  ) {
    return "success";
  }
  return "neutral";
}

function hasMetadata(metadata: Record<string, unknown>) {
  return Object.keys(metadata).length > 0;
}

function EmptyFeed({ isLoading, label }: { isLoading: boolean; label: string }) {
  return (
    <div className="execution-empty" role="status">
      {isLoading && <span className="execution-empty__pulse" aria-hidden="true" />}
      <strong>{isLoading ? "Waiting for the next checkpoint" : `No ${label} returned`}</strong>
      <p>
        {isLoading
          ? "The workflow is still running. New backend events will appear here automatically."
          : `This run did not report any ${label}; no placeholder activity has been generated.`}
      </p>
    </div>
  );
}

function MetadataDetails({ value, label = "Metadata" }: { value: Record<string, unknown>; label?: string }) {
  if (!hasMetadata(value)) {
    return null;
  }

  return (
    <details className="trace-details">
      <summary>{label}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

function NodeTimeline({
  events,
  nextNodes,
  isLoading,
}: {
  events: TraceEvent[];
  nextNodes: string[];
  isLoading: boolean;
}) {
  if (events.length === 0) {
    return <EmptyFeed isLoading={isLoading} label="node events" />;
  }

  return (
    <ol className="execution-timeline">
      {events.map((event, index) => (
        <li key={`${event.timestamp}-${event.node}-${index}`}>
          <span className={`timeline-marker timeline-marker--${statusTone(event.status)}`} aria-hidden="true" />
          <article className="timeline-card">
            <header>
              <div>
                <span className="trace-index">{String(index + 1).padStart(2, "0")}</span>
                <strong>{event.node}</strong>
              </div>
              <div className="trace-card__meta">
                <span className={`trace-status trace-status--${statusTone(event.status)}`}>{event.status}</span>
                <time dateTime={event.timestamp}>{formatTimestamp(event.timestamp)}</time>
              </div>
            </header>
            <p>{event.message}</p>
            <MetadataDetails value={event.metadata} />
          </article>
        </li>
      ))}
      {isLoading && (
        <li className="timeline-pending" aria-live="polite">
          <span className="timeline-marker timeline-marker--running" aria-hidden="true" />
          <div>
            <strong>{nextNodes.length > 0 ? `Scheduled: ${nextNodes.join(", ")}` : "Workflow in progress"}</strong>
            <span>Polling for the next LangGraph checkpoint…</span>
          </div>
        </li>
      )}
    </ol>
  );
}

function ToolCalls({ calls, isLoading }: { calls: ToolCallRecord[]; isLoading: boolean }) {
  if (calls.length === 0) {
    return <EmptyFeed isLoading={isLoading} label="tool calls" />;
  }

  return (
    <div className="call-list">
      {calls.map((call, index) => (
        <article className="call-card" key={`${call.timestamp}-${call.tool_name}-${index}`}>
          <header>
            <div>
              <span className="call-card__kind">Tool call</span>
              <h3>{call.tool_name}</h3>
            </div>
            <div className="trace-card__meta">
              <span className={`trace-status trace-status--${statusTone(call.status)}`}>{call.status}</span>
              <time dateTime={call.timestamp}>{formatTimestamp(call.timestamp)}</time>
            </div>
          </header>
          <p className="call-card__route">
            <span>Node</span>
            <strong>{call.node}</strong>
          </p>
          <details className="trace-details">
            <summary>Tool input</summary>
            <pre>{JSON.stringify(call.input_payload, null, 2)}</pre>
          </details>
          <div className="call-output">
            <span>{call.error ? "Error" : "Output preview"}</span>
            <pre>{call.error ?? (call.output_preview || "No output preview was recorded.")}</pre>
          </div>
          <MetadataDetails value={call.metadata} />
        </article>
      ))}
    </div>
  );
}

function ModelCalls({ calls, isLoading }: { calls: ModelCallRecord[]; isLoading: boolean }) {
  if (calls.length === 0) {
    return <EmptyFeed isLoading={isLoading} label="model calls" />;
  }

  return (
    <div className="call-list">
      {calls.map((call, index) => (
        <article className="call-card call-card--model" key={`${call.timestamp}-${call.node}-${index}`}>
          <header>
            <div>
              <span className="call-card__kind">Model call</span>
              <h3>{call.model || call.provider}</h3>
            </div>
            <div className="trace-card__meta">
              <span className={`trace-status trace-status--${statusTone(call.status)}`}>{call.status}</span>
              <time dateTime={call.timestamp}>{formatTimestamp(call.timestamp)}</time>
            </div>
          </header>
          <div className="call-card__facts">
            <p>
              <span>Node</span>
              <strong>{call.node}</strong>
            </p>
            <p>
              <span>Provider</span>
              <strong>{call.provider}</strong>
            </p>
          </div>
          <details className="trace-details">
            <summary>Model input preview</summary>
            <pre>{call.input_preview || "No input preview was recorded."}</pre>
          </details>
          <div className="call-output call-output--model">
            <span>{call.error ? "Error" : "Model output"}</span>
            <pre>{call.error ?? (call.output_preview || "No model output was recorded.")}</pre>
          </div>
          <MetadataDetails value={call.metadata} />
        </article>
      ))}
    </div>
  );
}

export function ExecutionPanel({ run, isLoading, isOpen, onClose }: ExecutionPanelProps) {
  const [activeView, setActiveView] = useState<ExecutionView>("nodes");
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const events = run?.execution_trace ?? [];
  const toolCalls = run?.tool_call_history ?? [];
  const modelCalls = run?.model_call_history ?? [];
  const views = useMemo(
    () => [
      { id: "nodes" as const, label: "Nodes", count: events.length },
      { id: "tools" as const, label: "Tools", count: toolCalls.length },
      { id: "models" as const, label: "Models", count: modelCalls.length },
    ],
    [events.length, modelCalls.length, toolCalls.length],
  );

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="execution-overlay"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) {
          onClose();
        }
      }}
    >
      <section className="execution-panel" role="dialog" aria-modal="true" aria-labelledby="execution-title">
        <header className="execution-panel__header">
          <div>
            <p className="eyebrow">LangGraph observability</p>
            <h2 id="execution-title">Execution process</h2>
            <p>
              <code>{run?.task_id ?? "Run not initialized"}</code>
              <span aria-hidden="true">·</span>
              <span>{isLoading ? "Live checkpoint polling" : "Recorded run history"}</span>
            </p>
          </div>
          <div className="execution-panel__actions">
            {isLoading && <span className="live-indicator">Live</span>}
            <button ref={closeButtonRef} type="button" className="icon-button" onClick={onClose}>
              <span aria-hidden="true">×</span>
              <span className="sr-only">Close execution process</span>
            </button>
          </div>
        </header>

        <nav className="execution-tabs" aria-label="Execution detail views">
          {views.map((view) => (
            <button
              key={view.id}
              type="button"
              className={activeView === view.id ? "is-active" : ""}
              aria-pressed={activeView === view.id}
              onClick={() => setActiveView(view.id)}
            >
              <span>{view.label}</span>
              <strong>{view.count}</strong>
            </button>
          ))}
        </nav>

        <div className="execution-panel__body" aria-live="polite">
          {activeView === "nodes" && (
            <NodeTimeline events={events} nextNodes={run?.next_nodes ?? []} isLoading={isLoading} />
          )}
          {activeView === "tools" && <ToolCalls calls={toolCalls} isLoading={isLoading} />}
          {activeView === "models" && <ModelCalls calls={modelCalls} isLoading={isLoading} />}
        </div>
      </section>
    </div>
  );
}
