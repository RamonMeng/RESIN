#!/usr/bin/env python3
"""Summarize RESIN token use and latency without double counting artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TOKEN_FIELDS = (
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "visible_completion_tokens",
    "total_tokens",
)

PREPROCESS_STAGES = (
    "00_data_preparation",
    "01_extraction",
    "02_normalization",
    "03_index",
)
ONLINE_STAGES = (
    "04_query_plans",
    "05_retrieval",
    "06_graph",
    "07_agent",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def json_values(path: Path) -> Iterable[Any]:
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        else:
            yield read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return


def nested_metrics(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        metric = value.get("_resin_request_metrics")
        if isinstance(metric, dict):
            yield metric
        for child in value.values():
            yield from nested_metrics(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_metrics(child)


def blank_stage() -> dict[str, Any]:
    return {
        "llm_requests": 0,
        "requests_with_usage": 0,
        "llm_api_latency_seconds_sum": 0.0,
        **{field: 0 for field in TOKEN_FIELDS},
    }


def add_metric(target: dict[str, Any], metric: dict[str, Any]) -> None:
    target["llm_requests"] += 1
    target["llm_api_latency_seconds_sum"] += float(metric.get("latency_seconds") or 0.0)
    if metric.get("total_tokens") is not None:
        target["requests_with_usage"] += 1
    for field in TOKEN_FIELDS:
        value = metric.get(field)
        if value is not None:
            target[field] += int(value)


def scan_stage(root: Path, stage_name: str, seen: set[str],
               target: dict[str, Any]) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if "10_cost" in path.parts or "legacy_colliding_raw" in path.parts:
            continue
        for value in json_values(path):
            for metric in nested_metrics(value):
                request_id = str(metric.get("request_id") or "")
                if not request_id:
                    request_id = json.dumps(metric, sort_keys=True, ensure_ascii=False)
                if request_id in seen:
                    continue
                seen.add(request_id)
                add_metric(target, metric)


def finalize_stage(value: dict[str, Any], queries: int) -> dict[str, Any]:
    requests = int(value.get("llm_requests", 0))
    value["llm_api_latency_seconds_sum"] = round(
        float(value.get("llm_api_latency_seconds_sum", 0.0)), 6
    )
    value["llm_api_latency_seconds_mean"] = round(
        value["llm_api_latency_seconds_sum"] / requests, 6
    ) if requests else 0.0
    wall = float(value.get("wall_time_seconds", 0.0))
    value["wall_time_seconds"] = round(wall, 6)
    value["tokens_per_query"] = round(value.get("total_tokens", 0) / queries, 3) if queries else None
    value["wall_time_seconds_per_query"] = round(wall / queries, 6) if queries else None
    return value


def aggregate(stages: dict[str, dict[str, Any]], names: Iterable[str],
              queries: int) -> dict[str, Any]:
    output = blank_stage()
    output["wall_time_seconds"] = 0.0
    included = []
    for name in names:
        if name not in stages:
            continue
        included.append(name)
        stage = stages[name]
        output["wall_time_seconds"] += float(stage.get("wall_time_seconds", 0.0))
        output["llm_requests"] += int(stage.get("llm_requests", 0))
        output["requests_with_usage"] += int(stage.get("requests_with_usage", 0))
        output["llm_api_latency_seconds_sum"] += float(
            stage.get("llm_api_latency_seconds_sum", 0.0)
        )
        for field in TOKEN_FIELDS:
            output[field] += int(stage.get(field, 0))
    output["included_stages"] = included
    return finalize_stage(output, queries)


def add_estimated_cost(value: dict[str, Any], input_price: float,
                       cached_input_price: float, output_price: float) -> None:
    """Attach list-price estimates; prices are USD per one million tokens."""
    prompt = int(value.get("prompt_tokens", 0))
    cached = min(prompt, int(value.get("cached_prompt_tokens", 0)))
    uncached = max(0, prompt - cached)
    completion = int(value.get("completion_tokens", 0))
    cost = (
        uncached * input_price
        + cached * cached_input_price
        + completion * output_price
    ) / 1_000_000
    no_cache_cost = (
        prompt * input_price + completion * output_price
    ) / 1_000_000
    value["uncached_prompt_tokens"] = uncached
    value["estimated_cost_usd"] = round(cost, 9)
    value["estimated_cost_usd_no_cache"] = round(no_cache_cost, 9)


def source_corpus_tokens(data_dir: Path, explicit_tokens: int | None,
                         tokenizer_name: str) -> tuple[int | None, dict[str, Any]]:
    """Count each selected original HolisQA article once for /1K normalization."""
    if explicit_tokens is not None:
        return explicit_tokens, {
            "status": "provided", "tokens": explicit_tokens,
            "tokenizer": "user_provided",
        }
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        return None, {"status": "unavailable", "reason": "metadata.json missing"}
    metadata = read_json(metadata_path)
    source_dir = Path(str(metadata.get("source_dir", "")))
    domains = [str(x) for x in metadata.get("domains", []) if str(x)]
    selected = {str(x) for x in metadata.get("selected_document_ids", []) if str(x)}
    if not source_dir.exists() or not domains:
        return None, {
            "status": "unavailable",
            "reason": "official source_dir/domains unavailable; pass --corpus_tokens",
            "source_dir": str(source_dir),
        }
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(tokenizer_name)
    except Exception as exc:
        return None, {
            "status": "unavailable",
            "reason": f"cannot load tiktoken tokenizer {tokenizer_name}: {exc!r}",
        }

    texts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for domain in domains:
        path = source_dir / domain / "sentences.jsonl"
        if not path.exists():
            return None, {"status": "unavailable", "reason": f"missing {path}"}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                article_id = f"{domain}__{row.get('article_id', '')}"
                if selected and article_id not in selected:
                    continue
                texts[article_id].append((
                    int(row.get("sent_idx", 0)), str(row.get("text", ""))
                ))
    total = 0
    for rows in texts.values():
        body = " ".join(text for _, text in sorted(rows) if text.strip())
        total += len(encoding.encode(body))
    return total, {
        "status": "computed_from_original_articles",
        "tokens": total,
        "tokenizer": tokenizer_name,
        "documents": len(texts),
        "source_dir": str(source_dir),
        "includes_each_selected_article_once": True,
    }


def count_queries(qa_file: Path) -> int:
    with qa_file.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--qa_file", type=Path, default=None)
    parser.add_argument("--evaluation_wall_seconds", type=float, default=None)
    parser.add_argument("--corpus_tokens", type=int, default=None)
    parser.add_argument("--corpus_tokenizer", default="o200k_base")
    parser.add_argument("--input_price_per_million", type=float, default=0.05)
    parser.add_argument("--cached_input_price_per_million", type=float, default=0.005)
    parser.add_argument("--output_price_per_million", type=float, default=0.40)
    args = parser.parse_args()

    qa_file = args.qa_file or args.data_dir / "qa" / "holisqa.jsonl"
    queries = count_queries(qa_file)
    stages: dict[str, dict[str, Any]] = defaultdict(blank_stage)
    seen: set[str] = set()

    scan_stage(
        args.data_dir / "01_passage_construction",
        "00_data_preparation",
        seen,
        stages["00_data_preparation"],
    )
    for name in (
        "01_extraction", "02_normalization", "03_index", "04_query_plans",
        "05_retrieval", "06_graph", "07_agent", "08_evaluation",
    ):
        scan_stage(args.run_dir / name, name, seen, stages[name])

    preprocessing_timing = args.data_dir / "preprocessing_timing.json"
    if preprocessing_timing.exists():
        timing = read_json(preprocessing_timing)
        stage = stages["00_data_preparation"]
        stage["wall_time_seconds"] = float(timing.get("total_wall_time_seconds", 0.0))
        stage["selection_wall_time_seconds"] = float(
            timing.get("selection_wall_time_seconds", 0.0)
        )
        stage["passage_construction_wall_time_seconds"] = float(
            timing.get("passage_construction_wall_time_seconds", 0.0)
        )
    else:
        passage_diagnostics = (
            args.data_dir / "01_passage_construction" / "diagnostics.json"
        )
        if passage_diagnostics.exists():
            timing = read_json(passage_diagnostics)
            stage = stages["00_data_preparation"]
            stage["wall_time_seconds"] = float(timing.get("wall_time_seconds", 0.0))
            stage["passage_construction_wall_time_seconds"] = float(
                timing.get("wall_time_seconds", 0.0)
            )
            stage["timing_scope"] = (
                "latest recorded passage-construction invocation; a resumed/migrated "
                "run may not represent clean from-scratch construction latency"
            )

    wall_path = args.run_dir / "10_cost" / "stage_wall_times.json"
    if wall_path.exists():
        for name, timing in read_json(wall_path).get("stages", {}).items():
            if isinstance(timing, dict):
                stages[name]["wall_time_seconds"] = float(
                    timing.get("cumulative_wall_time_seconds",
                               timing.get("wall_time_seconds", 0.0))
                )

    finalized = {
        name: finalize_stage(dict(value), queries)
        for name, value in sorted(stages.items())
        if value.get("llm_requests") or value.get("wall_time_seconds")
    }
    for value in finalized.values():
        add_estimated_cost(
            value, args.input_price_per_million,
            args.cached_input_price_per_million, args.output_price_per_million,
        )
    preprocessing = aggregate(finalized, PREPROCESS_STAGES, queries)
    online = aggregate(finalized, ONLINE_STAGES, queries)
    add_estimated_cost(
        preprocessing, args.input_price_per_million,
        args.cached_input_price_per_million, args.output_price_per_million,
    )
    add_estimated_cost(
        online, args.input_price_per_million,
        args.cached_input_price_per_million, args.output_price_per_million,
    )
    corpus_tokens, corpus_accounting = source_corpus_tokens(
        args.data_dir, args.corpus_tokens, args.corpus_tokenizer
    )
    construction_per_1k = None
    if corpus_tokens and corpus_tokens > 0:
        scale = 1000.0 / corpus_tokens
        construction_per_1k = {
            "tokens": round(preprocessing.get("total_tokens", 0) * scale, 2),
            "time_seconds": round(preprocessing.get("wall_time_seconds", 0.0) * scale, 6),
            "cost_usd": round(preprocessing.get("estimated_cost_usd", 0.0) * scale, 9),
            "cost_usd_no_cache": round(
                preprocessing.get("estimated_cost_usd_no_cache", 0.0) * scale, 9
            ),
        }
    online_per_query = {
        "tokens": round(online.get("total_tokens", 0) / queries, 2) if queries else None,
        "time_seconds": round(online.get("wall_time_seconds", 0.0) / queries, 6)
        if queries else None,
        "cost_usd": round(online.get("estimated_cost_usd", 0.0) / queries, 9)
        if queries else None,
        "cost_usd_no_cache": round(
            online.get("estimated_cost_usd_no_cache", 0.0) / queries, 9
        ) if queries else None,
    }
    report = {
        "queries": queries,
        "preprocessing": preprocessing,
        "online": online,
        "stages": finalized,
        "pricing": {
            "model": "gpt-5-nano",
            "currency": "USD",
            "unit": "per_1_million_tokens",
            "input": args.input_price_per_million,
            "cached_input": args.cached_input_price_per_million,
            "output_including_reasoning": args.output_price_per_million,
            "source": "OpenAI list pricing; configurable through CLI arguments",
        },
        "corpus_token_accounting": corpus_accounting,
        "causalrag2_table_compatible": {
            "construction_per_1k_corpus_tokens": construction_per_1k,
            "online_query_per_query": online_per_query,
            "latency_note": (
                "time_seconds is total stage wall time divided by corpus-token/query count. "
                "Because this run uses concurrent workers, it is a throughput-normalized "
                "wall-time measure, not mean serial per-item response latency."
            ),
            "reference_causalrag2": {
                "construction_per_1k_corpus_tokens": {
                    "tokens": 9864.49, "time_seconds": 9.622, "cost_usd": 0.0015,
                },
                "online_query_per_query": {
                    "tokens": 5075.65, "time_seconds": 0.801, "cost_usd": 0.00049,
                },
            },
        },
        "evaluation": {
            "wall_time_seconds": args.evaluation_wall_seconds,
            "token_usage": "not exposed by the standalone Ragas evaluator",
            "included_in_online": False,
        },
        "definitions": {
            "preprocessing": (
                "query/article selection, semantic passage construction, reasoning-unit "
                "extraction, normalization, and indexing"
            ),
            "online": "query planning, hop-wise retrieval, and hop/final answer generation",
            "wall_time": "elapsed user-visible time; sequential stage wall times are summed",
            "llm_api_latency_sum": (
                "sum of individual request latencies; may exceed wall time under concurrency"
            ),
            "completion_tokens": "provider-reported output tokens, including reasoning tokens",
            "visible_completion_tokens": "completion tokens minus reasoning tokens",
        },
        "deduplicated_llm_requests": len(seen),
    }
    output = args.run_dir / "10_cost" / "summary.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[done] cost report:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
