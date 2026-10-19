#!/usr/bin/env python3
"""Reuse validated query-plan checkpoints when expanding a benchmark run.

Only per-query planner checkpoints are copied.  Aggregate files such as
``plans.jsonl`` and ``summary.json`` are deliberately left to the main runner,
which rebuilds them against the complete target QA file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", type=Path, required=True)
    parser.add_argument("--target_run", type=Path, required=True)
    parser.add_argument("--target_qa_file", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def query_id(index: int, query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"q_{index:06d}_{digest}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.count < 0:
        raise ValueError("--count must be non-negative")

    source_stage = args.source_run / "04_query_plans"
    target_stage = args.target_run / "04_query_plans"
    if not source_stage.exists():
        raise FileNotFoundError(f"Missing source planner stage: {source_stage}")

    qa_rows = read_jsonl(args.target_qa_file)
    requested = min(args.count, len(qa_rows))
    aggregate = {
        int(row.get("row_index", -1)): row
        for row in read_jsonl(source_stage / "plans.jsonl")
        if row.get("status", "ok") == "ok"
    }

    copied = 0
    already_present = 0
    unavailable = 0
    target_stage.mkdir(parents=True, exist_ok=True)

    for index, qa in enumerate(qa_rows[:requested]):
        query = str(qa.get("query", qa.get("question", "")))
        qid = query_id(index, query)
        source_path = source_stage / f"{qid}.json"
        target_path = target_stage / f"{qid}.json"

        if target_path.exists():
            existing = read_json(target_path)
            if existing.get("status") == "ok" and existing.get("query") == query:
                already_present += 1
                continue

        if source_path.exists():
            checkpoint = read_json(source_path)
            if checkpoint.get("status") != "ok":
                unavailable += 1
                continue
            if checkpoint.get("query") != query:
                raise ValueError(
                    f"Query mismatch at row {index}; refusing to reuse {source_path}"
                )
            shutil.copy2(source_path, target_path)
            copied += 1
            continue

        # Older runs may retain only plans.jsonl.  Reconstruct a resumable
        # checkpoint in that case.  It lacks request-usage metadata but its
        # validated plan remains reusable.
        plan_row = aggregate.get(index)
        if plan_row is None:
            unavailable += 1
            continue
        if str(plan_row.get("query", "")) != query:
            raise ValueError(
                f"Query mismatch at row {index}; refusing aggregate-plan reuse"
            )
        write_json(target_path, {
            "status": "ok",
            "query_id": qid,
            "row_index": index,
            "query": query,
            "gold_doc_ids": [str(x) for x in (qa.get("gold_doc_ids") or [])],
            "plan": plan_row.get("plan", {}),
            "reused_from": str(source_stage / "plans.jsonl"),
        })
        copied += 1

    print(json.dumps({
        "requested": requested,
        "copied": copied,
        "already_present": already_present,
        "unavailable_and_will_be_generated": unavailable,
        "source_run": str(args.source_run),
        "target_run": str(args.target_run),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
