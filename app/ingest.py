from __future__ import annotations

# CLI helper for populating the retrieval corpus.
# This keeps vector-store ingestion separate from the main run command so
# knowledge base refreshes do not have to pass through the agent workflow.

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.runtime import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local knowledge documents into the configured vector store.")
    parser.add_argument("--source-dir", help="Directory containing source files for ingestion.")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the target collection before upserting chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = build_runtime()
    report = runtime.retrieval.ingest_directory(
        source_dir=args.source_dir,
        recreate_collection=args.recreate_collection,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
