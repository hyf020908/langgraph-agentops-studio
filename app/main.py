from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.runner import WorkflowRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph AgentOps Studio from the CLI.")
    parser.add_argument("--task", help="Task string to execute.")
    parser.add_argument("--task-file", help="Path to a text or JSON file containing a task.")
    parser.add_argument("--task-id", help="Optional task ID for checkpointed runs.")
    parser.add_argument("--task-type", default="general", help="Task type used by governance policy evaluation.")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically continue any human approval interrupt for demonstration runs.",
    )
    return parser.parse_args()


def resolve_task(args: argparse.Namespace) -> str:
    if args.task:
        return args.task
    if args.task_file:
        return WorkflowRunner.read_task_from_example(args.task_file)
    raise SystemExit("Provide --task or --task-file.")


def main() -> None:
    args = parse_args()
    task = resolve_task(args)
    try:
        runner = WorkflowRunner()
        state, interrupt_payload = runner.start(
            task=task,
            task_id=args.task_id,
            auto_approve=args.auto_approve,
            task_type=args.task_type,
        )
    except Exception as exc:
        raise SystemExit(f"Workflow initialization/execution failed: {exc}") from exc
    summary = runner.summarize(state, interrupt_payload)
    print(json.dumps(summary.model_dump(), indent=2))
    if interrupt_payload and not args.auto_approve:
        print("Run paused for human approval. Re-run with --auto-approve or continue via API/runner.")


if __name__ == "__main__":
    main()
