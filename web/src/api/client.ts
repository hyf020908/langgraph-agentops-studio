export type TaskType = "general" | "architecture" | "security" | "finance";

export interface HealthResponse {
  status: string;
}

export interface ProvidersResponse {
  llm: string;
  embedding: string;
  vector_db: string;
  web_search: string | null;
  web_reader: string | null;
}

export interface RunRequest {
  task: string;
  task_id?: string;
  task_type: TaskType;
  auto_approve: boolean;
}

export interface ContinueRequest {
  approved: boolean;
  reviewer: string;
  rationale: string;
}

export interface TraceEvent {
  timestamp: string;
  node: string;
  status: string;
  message: string;
  metadata: Record<string, unknown>;
}

export interface ToolCallRecord {
  timestamp: string;
  node: string;
  tool_name: string;
  status: "success" | "error";
  input_payload: Record<string, unknown>;
  output_preview: string;
  error: string | null;
  metadata: Record<string, unknown>;
}

export interface ModelCallRecord {
  timestamp: string;
  node: string;
  provider: string;
  model: string;
  status: "success" | "error";
  input_preview: string;
  output_preview: string;
  error: string | null;
  metadata: Record<string, unknown>;
}

export interface RunResponse {
  task_id: string;
  status: string;
  next_nodes: string[];
  approval_required: boolean;
  approval_payload: Record<string, unknown> | null;
  artifact_paths: string[];
  review_summary: string | null;
  draft_report: string | null;
  final_report: string | null;
  execution_trace: TraceEvent[];
  tool_call_history: ToolCallRecord[];
  model_call_history: ModelCallRecord[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        message = payload.detail;
      }
    } catch {
      message = response.statusText || message;
    }
    throw new ApiError(message, response.status);
  }

  return (await response.json()) as T;
}

export const apiClient = {
  baseUrl: API_BASE_URL,
  health: () => request<HealthResponse>("/health"),
  providers: () => request<ProvidersResponse>("/providers"),
  createRun: (payload: RunRequest) =>
    request<RunResponse>("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getRun: (taskId: string) => request<RunResponse>(`/runs/${encodeURIComponent(taskId)}`),
  continueRun: (taskId: string, payload: ContinueRequest) =>
    request<RunResponse>(`/runs/${encodeURIComponent(taskId)}/continue`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
