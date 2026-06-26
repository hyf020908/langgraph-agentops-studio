import type { RunRequest, TaskType } from "../api/client";

interface TaskFormProps {
  task: string;
  taskType: TaskType;
  autoApprove: boolean;
  isSubmitting: boolean;
  onTaskChange: (value: string) => void;
  onTaskTypeChange: (value: TaskType) => void;
  onAutoApproveChange: (value: boolean) => void;
  onSubmit: (payload: RunRequest) => void;
}

const taskTypes: TaskType[] = ["general", "architecture", "security", "finance"];

export function TaskForm({
  task,
  taskType,
  autoApprove,
  isSubmitting,
  onTaskChange,
  onTaskTypeChange,
  onAutoApproveChange,
  onSubmit,
}: TaskFormProps) {
  const canSubmit = task.trim().length > 0 && !isSubmitting;

  return (
    <form
      className="panel task-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSubmit) {
          return;
        }
        onSubmit({
          task: task.trim(),
          task_type: taskType,
          auto_approve: autoApprove,
        });
      }}
    >
      <div className="panel__header">
        <div>
          <p className="eyebrow">Workflow launch</p>
          <h2>New run</h2>
        </div>
      </div>

      <label className="field">
        <span>Task</span>
        <textarea
          value={task}
          onChange={(event) => onTaskChange(event.target.value)}
          placeholder="Evaluate orchestration patterns for a regulated platform migration plan."
          rows={8}
          required
        />
      </label>

      <div className="field-grid">
        <label className="field">
          <span>Task Type</span>
          <select value={taskType} onChange={(event) => onTaskTypeChange(event.target.value as TaskType)}>
            {taskTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </label>

        <label className="switch-control">
          <input
            type="checkbox"
            checked={autoApprove}
            onChange={(event) => onAutoApproveChange(event.target.checked)}
          />
          <span className="switch-control__track" aria-hidden="true" />
          <span>
            Auto approve
            <small>Resume approval gates automatically for demo runs.</small>
          </span>
        </label>
      </div>

      <button className="primary-button" type="submit" disabled={!canSubmit}>
        {isSubmitting ? "Running..." : "Submit run"}
      </button>
    </form>
  );
}
