#!/usr/bin/env python3
"""Prepare a reproducible MS MARCO v1.1 passage-unit experiment.

The adapter streams the official ``microsoft/ms_marco`` validation split from
Hugging Face (or reads an equivalent local JSON/JSONL export).  Every candidate
web passage is one RESIN reasoning unit.  Passages marked
``is_selected == 1`` are retained only as gold provenance for evaluation; the
flag and reference answer are never included in corpus text or model prompts.

Selection is deterministic: take the first N validation rows in official order
that have a non-empty human answer and at least one selected passage.  All
selected questions share the union of their candidate passages as one corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source_file", type=Path)
    source.add_argument(
        "--from_huggingface", action="store_true",
        help="Stream MS MARCO from Hugging Face (default without --source_file).",
    )
    parser.add_argument("--hf_dataset", default="microsoft/ms_marco")
    parser.add_argument("--hf_config", default="v1.1")
    parser.add_argument("--hf_split", default="validation")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--max_queries", type=int, default=100)
    return parser.parse_args()


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def digest(value: str, size: int = 20) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:size]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def local_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield dict(value)
        return
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        for key in ("data", "rows", "records"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for row in value:
        if isinstance(row, dict):
            yield dict(row)


def huggingface_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face streaming requires datasets:\n"
            "  python3 -m pip install -U datasets"
        ) from exc
    stream = load_dataset(
        args.hf_dataset,
        args.hf_config,
        split=args.hf_split,
        streaming=True,
    )
    for row in stream:
        yield dict(row)


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_space(x) for x in value if normalize_space(x)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped == "[]":
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [normalize_space(stripped)]
        return string_list(decoded)
    return []


def passage_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("passages", {})
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        texts = list(value.get("passage_text", []))
        selected = list(value.get("is_selected", []))
        urls = list(value.get("url", []))
        for index, text in enumerate(texts):
            try:
                is_selected = int(selected[index]) if index < len(selected) else -1
            except (TypeError, ValueError):
                is_selected = -1
            result.append({
                "text": normalize_space(text),
                "url": normalize_space(urls[index]) if index < len(urls) else "",
                "is_selected": is_selected,
                "position": index,
            })
        return [x for x in result if x["text"]]
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            try:
                is_selected = int(item.get("is_selected", -1))
            except (TypeError, ValueError):
                is_selected = -1
            text = normalize_space(item.get("passage_text", item.get("text", "")))
            if text:
                result.append({
                    "text": text,
                    "url": normalize_space(item.get("url", "")),
                    "is_selected": is_selected,
                    "position": index,
                })
    return result


def passage_id(text: str) -> str:
    return "msmarco_passage_" + digest(normalize_space(text).lower())


def document_id(text: str) -> str:
    return "msmarco_doc_" + digest(normalize_space(text).lower())


def main() -> int:
    args = parse_args()
    if args.max_queries <= 0:
        raise ValueError("--max_queries must be positive")
    if args.source_file is None:
        rows = huggingface_rows(args)
        source_description = (
            f"huggingface://{args.hf_dataset}/{args.hf_config}/{args.hf_split}"
        )
    else:
        rows = local_rows(args.source_file)
        source_description = str(args.source_file)

    selected: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []
    scanned = 0
    skipped_no_answer = 0
    skipped_no_gold_context = 0
    for row in rows:
        scanned += 1
        answers = string_list(row.get("answers", []))
        passages = passage_rows(row)
        if not answers:
            skipped_no_answer += 1
            continue
        if not any(item["is_selected"] == 1 for item in passages):
            skipped_no_gold_context += 1
            continue
        selected.append((row, passages, answers[0]))
        if len(selected) >= args.max_queries:
            break
    if len(selected) < args.max_queries:
        raise ValueError(
            f"Only {len(selected)} eligible rows found after scanning {scanned}; "
            f"requested {args.max_queries}."
        )

    passage_store: dict[str, dict[str, Any]] = {}
    for _, passages, _ in selected:
        for item in passages:
            source_id = passage_id(item["text"])
            passage_store.setdefault(source_id, {
                "text": item["text"], "url": item["url"],
            })

    documents: list[dict[str, Any]] = []
    raw_units: list[dict[str, Any]] = []
    for source_id, item in passage_store.items():
        text, url = item["text"], item["url"]
        doc_id = document_id(text)
        documents.append({
            "document_id": doc_id,
            "id": doc_id,
            "title": source_id,
            "body": text,
            "text": text,
            "url": url,
            "source": "MS MARCO v1.1 validation passage",
            "source_sentence_ids": [source_id],
            "sentence_ids": [source_id],
            "sentences": [text],
        })
        raw_units.append({
            "kind": "relation",
            "subject": "web passage",
            "subject_kind": "concept",
            "predicate": "contains_information",
            "object": text,
            "conditions": [],
            "conclusion": {},
            "aliases": [],
            "scope": "passage",
            "modality": "asserted",
            "polarity": "positive",
            "confidence": 1.0,
            "evidence_quote": "",
            "doc_id": doc_id,
            "chunk_id": 0,
            "extraction_unit_index": 0,
            "document": {
                "document_id": doc_id,
                "title": source_id,
                "url": url,
                "source": "MS MARCO v1.1 validation passage",
                "source_sentence_ids": [source_id],
                "sentence_ids": [source_id],
            },
        })

    qa_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for row, passages, answer in selected:
        query = normalize_space(row.get("query"))
        query_id = str(row.get("query_id", len(qa_rows)))
        gold_passages = [x for x in passages if x["is_selected"] == 1]
        context_ids = list(dict.fromkeys(passage_id(x["text"]) for x in gold_passages))
        gold_doc_ids = list(dict.fromkeys(document_id(x["text"]) for x in gold_passages))
        gold_contexts = list(dict.fromkeys(x["text"] for x in gold_passages))
        evidence = [{
            "source": "MS MARCO v1.1 validation",
            "url": item["url"],
            "sentence_id": passage_id(item["text"]),
            "fact": item["text"],
        } for item in gold_passages]
        qa = {
            "qa_id": query_id,
            "query": query,
            "question": query,
            "answer": answer,
            "all_answers": string_list(row.get("answers", [])),
            "well_formed_answers": string_list(row.get("wellFormedAnswers", [])),
            "question_type": normalize_space(row.get("query_type")),
            "gold_doc_ids": gold_doc_ids,
            "context_sentence_ids": context_ids,
            "evidence_list": evidence,
            "gold_context": gold_contexts,
        }
        qa_rows.append(qa)
        gold_rows.append({
            "qa_id": query_id,
            "query": query,
            "answer": answer,
            "context_sentence_ids": context_ids,
            "gold_contexts": gold_contexts,
        })

    write_json(args.data_dir / "corpus" / "corpus.json", documents)
    write_jsonl(args.data_dir / "qa" / "msmarco.jsonl", qa_rows)
    write_jsonl(args.data_dir / "gold_contexts.jsonl", gold_rows)
    write_json(args.data_dir / "metadata.json", {
        "dataset": "MS MARCO",
        "config": args.hf_config,
        "split": args.hf_split,
        "source": source_description,
        "selection": "first_n_labeled_with_selected_passage_in_official_order",
        "requested_queries": args.max_queries,
        "queries": len(qa_rows),
        "rows_scanned": scanned,
        "skipped_no_answer": skipped_no_answer,
        "skipped_no_gold_context": skipped_no_gold_context,
        "reasoning_unit_definition": "one unique candidate web passage",
        "documents": len(documents),
        "reasoning_units": len(raw_units),
        "corpus_characters": sum(len(row["body"]) for row in documents),
        "gold_labels_excluded_from_corpus": True,
    })

    extraction = args.run_dir / "01_extraction"
    write_jsonl(extraction / "units.jsonl", raw_units)
    write_jsonl(extraction / "errors.jsonl", [])
    write_json(extraction / "summary.json", {
        "documents": len(documents),
        "chunks": len(documents),
        "units": len(raw_units),
        "errors": 0,
        "model": "none (MS MARCO candidate-passage-as-unit adapter)",
        "reasoning_unit_definition": "one unique candidate web passage",
    })

    print(f"[done] rows scanned: {scanned}")
    print(f"[done] eligible queries: {len(qa_rows)}")
    print(f"[done] unique passage units: {len(raw_units)}")
    print(f"[done] data: {args.data_dir}")
    print(f"[done] extraction checkpoint: {extraction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
