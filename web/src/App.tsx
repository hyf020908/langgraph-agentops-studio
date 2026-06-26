import { useEffect, useState } from "react";
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

  async function submitRun(payload: RunRequest) {
    try {
      setError(null);
      setIsRunLoading(true);
      const response = await apiClient.createRun(payload);
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
      const response = await apiClient.continueRun(run.task_id, payload);
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
        <span>Latest run data reflects the most recent create or continue response.</span>
      </footer>
    </main>
  );
}
