#!/usr/bin/env python3
"""Collect QASC and MS MARCO metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasc_run", type=Path, required=True)
    parser.add_argument("--msmarco_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def first_number(value: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, (int, float)):
            return float(item)
    return None


def collect(name: str, run: Path) -> dict[str, Any]:
    ragas = read_json(run / "09_ar_cr" / "summary.json")
    cost = read_json(run / "10_cost" / "summary.json")

    online = cost.get("causalrag2_table_compatible", {}).get(
        "online_query_per_query", {}
    )
    return {
        "benchmark": name,
        "run_dir": str(run),
        "evaluated_queries": int(ragas.get("evaluated_queries", 0) or 0),
        "successful_ragas_queries": int(
            ragas.get("successful_ragas_queries", 0) or 0
        ),
        "AR": first_number(ragas, "AR"),
        "CR": first_number(ragas, "CR"),
        "CR_ID_diagnostic": first_number(ragas, "CR_ID"),
        "cr_context_mode": ragas.get("cr_context_mode"),
        "hop_top_k": read_json(run / "05_retrieval" / "summary.json").get(
            "hop_top_k"
        ),
        "online_tokens_per_query": first_number(online, "tokens"),
        "online_time_seconds_per_query": first_number(online, "time_seconds"),
        "online_cost_usd_per_query": first_number(online, "cost_usd"),
    }


def main() -> int:
    args = parse_args()
    rows = [
        collect("QASC", args.qasc_run),
        collect("MS MARCO", args.msmarco_run),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    tsv_path = args.output_dir / "summary.tsv"
    json_path.write_text(
        json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": rows}, ensure_ascii=False, indent=2))
    print(f"[done] combined JSON: {json_path}")
    print(f"[done] combined TSV:  {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
