import { useCallback, useState } from "react";
import type { ProvidersResponse, RunResponse } from "../api/client";
import { ExecutionPanel } from "./ExecutionPanel";
import { JsonView } from "./JsonView";
import { MarkdownReport } from "./MarkdownReport";
import { StatusBadge } from "./StatusBadge";

interface ResultPanelProps {
  run: RunResponse | null;
  providers: ProvidersResponse | null;
  isLoading: boolean;
}

function statusTone(run: RunResponse | null) {
  if (!run) {
    return "neutral";
  }
  if (run.approval_required) {
    return "warning";
  }
  const status = run.status.toLowerCase();
  if (status.includes("error") || status.includes("failed")) {
    return "danger";
  }
  if (status.includes("waiting") || status.includes("running") || status.includes("progress")) {
    return "neutral";
  }
  return "success";
}

function artifactName(path: string) {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

export function ResultPanel({ run, providers, isLoading }: ResultPanelProps) {
  const [isExecutionOpen, setIsExecutionOpen] = useState(false);
  const closeExecution = useCallback(() => setIsExecutionOpen(false), []);
  const executionEventCount =
    (run?.execution_trace?.length ?? 0) +
    (run?.tool_call_history?.length ?? 0) +
    (run?.model_call_history?.length ?? 0);
  const report = run?.final_report ?? run?.draft_report ?? run?.review_summary;
  const reportLabel = run?.final_report ? "Final report" : run?.draft_report ? "Draft report" : "Review summary";

  return (
    <section className="panel result-panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Run intelligence</p>
          <h2>Result snapshot</h2>
        </div>
        <div className="result-actions">
          <button
            type="button"
            className="execution-trigger"
            disabled={!run}
            aria-expanded={isExecutionOpen}
            onClick={() => setIsExecutionOpen(true)}
          >
            <span className="execution-trigger__icon" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <span>
              Execution process
              <small>{isLoading ? "tracking live" : `${executionEventCount} events`}</small>
            </span>
          </button>
          <StatusBadge label={run?.status ?? (isLoading ? "running" : "idle")} tone={statusTone(run)} />
        </div>
      </div>

      {!run && !isLoading && (
        <div className="empty-state">
          <div className="empty-state__glyph">LG</div>
          <h3>No run submitted</h3>
          <p>Launch a workflow to see status, review gates, artifacts, and the raw API response.</p>
        </div>
      )}

      {isLoading && (
        <div className="loading-state" role="status">
          <span />
          <div>
            <p>Running the workflow graph…</p>
            <button type="button" onClick={() => setIsExecutionOpen(true)} disabled={!run}>
              Open live execution process
            </button>
          </div>
        </div>
      )}

      {run && !isLoading && (
        <div className="result-stack">
          <div className="metric-grid">
            <div className="metric-tile">
              <span>Task ID</span>
              <strong>{run.task_id}</strong>
            </div>
            <div className="metric-tile">
              <span>Approval</span>
              <strong>{run.approval_required ? "Required" : "Clear"}</strong>
            </div>
            <div className="metric-tile">
              <span>Provider</span>
              <strong>{providers?.llm ?? "unknown"}</strong>
            </div>
          </div>

          <article className="report-block report-block--primary">
            <span>{reportLabel}</span>
            <MarkdownReport
              content={report ?? "The backend response did not include a report or review summary for this run."}
            />
          </article>

          {run.review_summary && run.review_summary !== report && (
            <article className="report-block">
              <span>Review summary</span>
              <p>{run.review_summary}</p>
            </article>
          )}

          <div className="artifact-list">
            <div className="artifact-list__header">
              <span>Artifacts</span>
              <strong>{run.artifact_paths.length}</strong>
            </div>
            {run.artifact_paths.length > 0 ? (
              <ul>
                {run.artifact_paths.map((path) => (
                  <li key={path}>
                    <span>{artifactName(path)}</span>
                    <code>{path}</code>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No artifacts were returned in this response.</p>
            )}
          </div>

          {run.approval_payload && <JsonView title="Approval payload" value={run.approval_payload} defaultOpen />}
          <JsonView title="Raw JSON" value={run} />
        </div>
      )}

      <ExecutionPanel run={run} isLoading={isLoading} isOpen={isExecutionOpen} onClose={closeExecution} />
    </section>
  );
}
