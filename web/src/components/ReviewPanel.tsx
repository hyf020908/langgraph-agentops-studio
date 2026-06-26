import { useMemo, useState } from "react";
import type { ContinueRequest, RunResponse } from "../api/client";

interface ReviewPanelProps {
  run: RunResponse | null;
  isSubmitting: boolean;
  onContinue: (payload: ContinueRequest) => void;
}

export function ReviewPanel({ run, isSubmitting, onContinue }: ReviewPanelProps) {
  const [reviewer, setReviewer] = useState("ops-reviewer");
  const [rationale, setRationale] = useState("Policy checks reviewed and accepted.");
  const [approved, setApproved] = useState(true);

  const approvalRequired = run?.approval_required ?? false;
  const canContinue = Boolean(run?.task_id && reviewer.trim() && rationale.trim() && !isSubmitting);
  const approvalHint = useMemo(() => {
    if (!run) {
      return "Submit a run to enable human review controls.";
    }
    if (!approvalRequired) {
      return "No active approval gate in the latest response.";
    }
    return "A human review gate is waiting for a decision.";
  }, [approvalRequired, run]);

  return (
    <section className={`panel review-panel ${approvalRequired ? "review-panel--active" : ""}`}>
      <div className="panel__header">
        <div>
          <p className="eyebrow">Human gate</p>
          <h2>Continue review</h2>
        </div>
        {approvalRequired && <span className="pulse-dot" aria-label="Approval required" />}
      </div>

      <p className="muted-copy">{approvalHint}</p>

      <div className="segmented-control" aria-label="Review decision">
        <button
          type="button"
          className={approved ? "is-selected" : ""}
          disabled={!approvalRequired || isSubmitting}
          onClick={() => setApproved(true)}
        >
          Approve
        </button>
        <button
          type="button"
          className={!approved ? "is-selected" : ""}
          disabled={!approvalRequired || isSubmitting}
          onClick={() => setApproved(false)}
        >
          Reject
        </button>
      </div>

      <label className="field">
        <span>Reviewer</span>
        <input
          value={reviewer}
          onChange={(event) => setReviewer(event.target.value)}
          disabled={!approvalRequired || isSubmitting}
        />
      </label>

      <label className="field">
        <span>Rationale</span>
        <textarea
          value={rationale}
          onChange={(event) => setRationale(event.target.value)}
          rows={4}
          disabled={!approvalRequired || isSubmitting}
        />
      </label>

      <button
        className="secondary-button"
        type="button"
        disabled={!approvalRequired || !canContinue}
        onClick={() =>
          onContinue({
            approved,
            reviewer: reviewer.trim(),
            rationale: rationale.trim(),
          })
        }
      >
        {isSubmitting ? "Continuing..." : "Continue run"}
      </button>
    </section>
  );
}
