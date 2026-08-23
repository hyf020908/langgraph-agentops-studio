import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  apiClient,
  type ContinueRequest,
  type HealthResponse,
  type ProvidersResponse,
  type RunRequest,
  type RunResponse,
  type TaskType,
} from "./api/client";
import { ProviderStrip } from "./components/ProviderStrip";
import { ResultPanel } from "./components/ResultPanel";
import { ReviewPanel } from "./components/ReviewPanel";
import { TaskForm } from "./components/TaskForm";

const featurePills = ["Multi-Agent Workflow", "Evidence Grounding", "Human Review Gate", "Artifact Export"];
const RUN_POLL_INTERVAL_MS = 700;

function createTaskId() {
  const randomPart =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 10)
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  return `task-${randomPart}`;
}

function waitingRun(taskId: string): RunResponse {
  return {
    task_id: taskId,
    status: "waiting_for_checkpoint",
    next_nodes: [],
    approval_required: false,
    approval_payload: null,
    artifact_paths: [],
    review_summary: null,
    draft_report: null,
    final_report: null,
    execution_trace: [],
    tool_call_history: [],
    model_call_history: [],
  };
}

function messageFromError(error: unknown) {
  if (error instanceof ApiError) {
    return error.status ? `${error.message} (HTTP ${error.status})` : error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error.";
}

export default function App() {
  const [task, setTask] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("architecture");
  const [autoApprove, setAutoApprove] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [providers, setProviders] = useState<ProvidersResponse | null>(null);
  const [run, setRun] = useState<RunResponse | null>(null);
  const [isBootLoading, setIsBootLoading] = useState(true);
  const [isRunLoading, setIsRunLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingGeneration = useRef(0);

  useEffect(() => {
    let isActive = true;

    async function loadBackendStatus() {
      try {
        setIsBootLoading(true);
        const [healthResponse, providerResponse] = await Promise.all([apiClient.health(), apiClient.providers()]);
        if (!isActive) {
          return;
        }
        setHealth(healthResponse);
        setProviders(providerResponse);
      } catch (loadError) {
        if (isActive) {
          setError(messageFromError(loadError));
        }
      } finally {
        if (isActive) {
          setIsBootLoading(false);
        }
      }
    }

    loadBackendStatus();

    return () => {
      isActive = false;
    };
  }, []);

  useEffect(
    () => () => {
      pollingGeneration.current += 1;
    },
    [],
  );

  async function runWithPolling(taskId: string, operation: () => Promise<RunResponse>) {
    const generation = pollingGeneration.current + 1;
    pollingGeneration.current = generation;
    let requestInFlight = false;

    async function pollSnapshot() {
      if (requestInFlight || pollingGeneration.current !== generation) {
        return;
      }
      requestInFlight = true;
      try {
        const snapshot = await apiClient.getRun(taskId);
        if (pollingGeneration.current === generation) {
          setRun(snapshot);
        }
      } catch {
        // A 404 is expected before LangGraph creates its first checkpoint.
        // Other polling errors are also non-fatal; the authoritative POST or
        // continue response still determines the outcome shown to the user.
      } finally {
        requestInFlight = false;
      }
    }

    const intervalId = window.setInterval(() => void pollSnapshot(), RUN_POLL_INTERVAL_MS);
    void pollSnapshot();
    try {
      return await operation();
    } finally {
      window.clearInterval(intervalId);
      if (pollingGeneration.current === generation) {
        pollingGeneration.current += 1;
      }
    }
  }

  async function submitRun(payload: RunRequest) {
    const taskId = payload.task_id ?? createTaskId();
    try {
      setError(null);
      setIsRunLoading(true);
      setRun(waitingRun(taskId));
      const response = await runWithPolling(taskId, () => apiClient.createRun({ ...payload, task_id: taskId }));
      setRun(response);
    } catch (submitError) {
      setError(messageFromError(submitError));
    } finally {
      setIsRunLoading(false);
    }
  }

  async function continueRun(payload: ContinueRequest) {
    if (!run) {
      return;
    }

    try {
      setError(null);
      setIsRunLoading(true);
      const response = await runWithPolling(run.task_id, () => apiClient.continueRun(run.task_id, payload));
      setRun(response);
    } catch (continueError) {
      setError(messageFromError(continueError));
    } finally {
      setIsRunLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="hero__content">
          <p className="eyebrow">Governed agent operations</p>
          <h1>LangGraph AgentOps Studio</h1>
          <p className="hero__subtitle">
            Launch multi-agent workflows with RAG, web grounding, governance review, and artifact export from a
            focused control surface.
          </p>
          <div className="feature-pills" aria-label="Platform features">
            {featurePills.map((pill) => (
              <span key={pill}>{pill}</span>
            ))}
          </div>
        </div>
        <div className="hero__visual" aria-hidden="true">
          <div className="flow-line flow-line--one" />
          <div className="flow-line flow-line--two" />
          <div className="flow-node flow-node--planner">Plan</div>
          <div className="flow-node flow-node--research">Ground</div>
          <div className="flow-node flow-node--review">Review</div>
          <div className="flow-node flow-node--export">Export</div>
        </div>
      </section>

      <ProviderStrip health={health} providers={providers} isLoading={isBootLoading} />

      {error && (
        <div className="error-banner" role="alert">
          <strong>Request failed</strong>
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
            Dismiss
          </button>
        </div>
      )}

      <section className="workspace-grid" aria-label="Workflow controls and results">
        <div className="control-stack">
          <TaskForm
            task={task}
            taskType={taskType}
            autoApprove={autoApprove}
            isSubmitting={isRunLoading}
            onTaskChange={setTask}
            onTaskTypeChange={setTaskType}
            onAutoApproveChange={setAutoApprove}
            onSubmit={submitRun}
          />
          <ReviewPanel run={run} isSubmitting={isRunLoading} onContinue={continueRun} />
        </div>

        <ResultPanel run={run} providers={providers} isLoading={isRunLoading} />
      </section>

      <footer className="app-footer">
        <span>Backend: {apiClient.baseUrl}</span>
        <span>Live state follows LangGraph checkpoints; final state follows the create or continue response.</span>
      </footer>
    </main>
  );
}
