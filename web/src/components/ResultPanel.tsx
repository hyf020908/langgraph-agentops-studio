import type { ProvidersResponse, RunResponse } from "../api/client";
import { JsonView } from "./JsonView";
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
  if (run.status.toLowerCase().includes("error") || run.status.toLowerCase().includes("failed")) {
    return "danger";
  }
  return "success";
}

function artifactName(path: string) {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

export function ResultPanel({ run, providers, isLoading }: ResultPanelProps) {
  return (
    <section className="panel result-panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Run intelligence</p>
          <h2>Result snapshot</h2>
        </div>
        <StatusBadge label={run?.status ?? (isLoading ? "running" : "idle")} tone={statusTone(run)} />
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
          <p>Running the workflow graph...</p>
        </div>
      )}

      {run && (
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

          <article className="report-block">
            <span>Final report / summary</span>
            <p>{run.review_summary ?? "The backend response did not include a review summary for this run."}</p>
          </article>

          <article className="report-block">
            <span>Decision record</span>
            <p>
              Decision details are exported as artifacts when the workflow reaches export. The current API snapshot
              exposes approval gate data and artifact paths.
            </p>
          </article>

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
    </section>
  );
}
