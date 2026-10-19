#!/usr/bin/env python3
"""Aggregate an existing Ragas AR/CR run by HolisQA category without rerunning it."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFERRED_ORDER = ["medicine", "computer_science", "business", "biology", "psychology"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean_present(rows: list[dict[str, Any]], key: str) -> tuple[float | None, int]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return ((sum(values) / len(values)) if values else None, len(values))


def percent(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def successful_row(row: dict[str, Any]) -> bool:
    return (
        row.get("ragas_status") == "ok"
        or (row.get("answer_relevancy") is not None
            and row.get("context_recall") is not None)
    )


def display(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def category_name(qa: dict[str, Any]) -> str:
    value = str(qa.get("category") or qa.get("domain") or "").strip()
    if value:
        return value
    qa_id = str(qa.get("qa_id") or "")
    return qa_id.split(":", 1)[0] if ":" in qa_id else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=Path, required=True)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    per_query_path = args.evaluation_dir / "per_query.jsonl"
    if not per_query_path.exists():
        raise FileNotFoundError(per_query_path)
    qa_rows = read_jsonl(args.qa_file)
    score_rows = read_jsonl(per_query_path)
    if len(qa_rows) != len(score_rows):
        raise ValueError(
            f"QA/evaluation row mismatch: {len(qa_rows)} != {len(score_rows)}"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, score in enumerate(score_rows):
        row_index = int(score.get("row_index", position))
        if row_index < 0 or row_index >= len(qa_rows):
            raise ValueError(f"Invalid row_index {row_index} in {per_query_path}")
        category = category_name(qa_rows[row_index])
        grouped[category].append(score)

    order = [name for name in PREFERRED_ORDER if name in grouped]
    order.extend(sorted(set(grouped) - set(order)))
    by_category: dict[str, dict[str, Any]] = {}
    for category in order:
        rows = grouped[category]
        ar, ar_count = mean_present(rows, "answer_relevancy")
        cr, cr_count = mean_present(rows, "context_recall")
        cr_id, cr_id_count = mean_present(rows, "CR_ID")
        successful = sum(successful_row(row) for row in rows)
        by_category[category] = {
            "evaluated_queries": len(rows),
            "successful_ragas_queries": successful,
            "failed_ragas_queries": len(rows) - successful,
            "AR": percent(ar),
            "CR": percent(cr),
            "CR_ID": percent(cr_id),
            "successful_ar_queries": ar_count,
            "successful_cr_queries": cr_count,
            "queries_with_cr_id": cr_id_count,
        }

    all_rows = [row for category in order for row in grouped[category]]
    overall_ar, overall_ar_count = mean_present(all_rows, "answer_relevancy")
    overall_cr, overall_cr_count = mean_present(all_rows, "context_recall")
    overall_cr_id, overall_cr_id_count = mean_present(all_rows, "CR_ID")
    output = {
        "evaluation_dir": str(args.evaluation_dir),
        "qa_file": str(args.qa_file),
        "aggregation_policy": (
            "AR and CR are macro-averaged over rows with a non-null metric, matching "
            "evaluate_ar_cr.py; failed Ragas rows are reported separately and are not zero-filled."
        ),
        "overall_recomputed": {
            "evaluated_queries": len(all_rows),
            "successful_ragas_queries": sum(successful_row(row) for row in all_rows),
            "AR": percent(overall_ar),
            "CR": percent(overall_cr),
            "CR_ID": percent(overall_cr_id),
            "successful_ar_queries": overall_ar_count,
            "successful_cr_queries": overall_cr_count,
            "queries_with_cr_id": overall_cr_id_count,
        },
        "by_category": by_category,
    }

    output_json = args.output_json or args.evaluation_dir / "summary_by_category.json"
    output_csv = args.output_csv or args.evaluation_dir / "summary_by_category.csv"
    output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "queries", "successful", "failed", "AR", "CR", "CR_ID"])
        for category in order:
            row = by_category[category]
            writer.writerow([
                category, row["evaluated_queries"], row["successful_ragas_queries"],
                row["failed_ragas_queries"], row["AR"], row["CR"], row["CR_ID"],
            ])

    print("category\tqueries\tsuccess\tfail\tAR\tCR\tCR_ID")
    for category in order:
        row = by_category[category]
        print(
            f"{category}\t{row['evaluated_queries']}\t{row['successful_ragas_queries']}\t"
            f"{row['failed_ragas_queries']}\t{display(row['AR'])}\t{display(row['CR'])}\t"
            f"{display(row['CR_ID'])}"
        )
    print(f"[done] JSON: {output_json}")
    print(f"[done] CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
