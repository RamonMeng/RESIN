#!/usr/bin/env python3
"""Prepare a reproducible 100-query QASC experiment for RESIN.

The adapter reads the official ``allenai/qasc`` validation split from Hugging
Face (or an equivalent local JSON/JSONL export).  QASC already supplies the two
atomic supporting facts required for sentence composition, so each unique
``fact1``/``fact2`` sentence becomes one reasoning unit.  No LLM preprocessing
is needed and ``combinedfact`` is deliberately excluded from the corpus because
it states the composed conclusion and would leak the reasoning result.

All selected questions share the union of their supporting facts as one global
corpus.  This preserves retrieval competition while keeping gold provenance
exact and makes Context Recall auditable at the fact-ID level.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source_file", type=Path)
    source.add_argument(
        "--from_huggingface", action="store_true",
        help="Stream QASC from Hugging Face (the default when no file is supplied).",
    )
    parser.add_argument("--hf_dataset", default="allenai/qasc")
    parser.add_argument("--hf_split", default="validation")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--max_queries", type=int, default=100)
    return parser.parse_args()


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


def load_local_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            values = (json.loads(line) for line in handle if line.strip())
            return [dict(row) for row in itertools.islice(values, limit)]
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        for key in ("data", "rows", "records"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [dict(row) for row in value[:limit] if isinstance(row, dict)]


def load_huggingface_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face streaming requires datasets:\n"
            "  python3 -m pip install -U datasets"
        ) from exc
    stream = load_dataset(
        args.hf_dataset,
        split=args.hf_split,
        streaming=True,
    )
    return [dict(row) for row in itertools.islice(stream, args.max_queries)]


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def choices(row: dict[str, Any]) -> list[tuple[str, str]]:
    value = row.get("choices", {})
    if isinstance(value, dict):
        labels = list(value.get("label", []))
        texts = list(value.get("text", []))
        return [(normalize_space(label), normalize_space(text))
                for label, text in zip(labels, texts)]
    result: list[tuple[str, str]] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                label = item.get("label", chr(ord("A") + index))
                text = item.get("text", item.get("value", ""))
            else:
                label, text = chr(ord("A") + index), item
            result.append((normalize_space(label), normalize_space(text)))
    return result


def formatted_question(row: dict[str, Any], options: list[tuple[str, str]]) -> str:
    supplied = normalize_space(row.get("formatted_question"))
    if supplied:
        return supplied
    question = normalize_space(row.get("question"))
    suffix = " ".join(f"({label}) {text}" for label, text in options)
    return f"{question} {suffix}".strip()


def fact_id(text: str) -> str:
    return "qasc_fact_" + digest(normalize_space(text).lower())


def document_id(text: str) -> str:
    return "qasc_doc_" + digest(normalize_space(text).lower())


def main() -> int:
    args = parse_args()
    if args.max_queries <= 0:
        raise ValueError("--max_queries must be positive")
    if args.source_file is None:
        selected = load_huggingface_rows(args)
        source_description = f"huggingface://{args.hf_dataset}/{args.hf_split}"
    else:
        selected = load_local_rows(args.source_file, args.max_queries)
        source_description = str(args.source_file)
    if not selected:
        raise ValueError("No QASC examples selected")

    # Build one shared corpus from the unique gold support facts of all selected
    # questions.  Exact-text deduplication preserves stable fact provenance.
    facts: dict[str, str] = {}
    for row in selected:
        for key in ("fact1", "fact2"):
            text = normalize_space(row.get(key))
            if text:
                facts.setdefault(fact_id(text), text)

    documents: list[dict[str, Any]] = []
    raw_units: list[dict[str, Any]] = []
    for source_id, text in facts.items():
        doc_id = document_id(text)
        document = {
            "document_id": doc_id,
            "id": doc_id,
            "title": source_id,
            "body": text,
            "text": text,
            "source": "QASC validation supporting fact",
            "source_sentence_ids": [source_id],
            "sentence_ids": [source_id],
            "sentences": [text],
        }
        documents.append(document)
        raw_units.append({
            "kind": "relation",
            "subject": "science fact",
            "subject_kind": "concept",
            "predicate": "states",
            "object": text,
            "conditions": [],
            "conclusion": {},
            "aliases": [],
            "scope": "sentence",
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
                "source": "QASC validation supporting fact",
                "source_sentence_ids": [source_id],
                "sentence_ids": [source_id],
            },
        })

    qa_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        options = choices(row)
        option_by_label = {label.upper(): text for label, text in options}
        answer_key = normalize_space(row.get("answerKey")).upper()
        answer = option_by_label.get(answer_key, "")
        fact_texts = [normalize_space(row.get("fact1")),
                      normalize_space(row.get("fact2"))]
        fact_texts = [text for text in fact_texts if text]
        if not answer or len(fact_texts) != 2:
            skipped.append({
                "row_index": index,
                "id": row.get("id", ""),
                "reason": "missing answer choice or one of fact1/fact2",
            })
            continue
        context_ids = [fact_id(text) for text in fact_texts]
        gold_doc_ids = [document_id(text) for text in fact_texts]
        question = formatted_question(row, options)
        qa_id = str(row.get("id") or f"qasc_{index:06d}")
        evidence = [{
            "source": "QASC validation",
            "sentence_id": source_id,
            "fact": text,
        } for source_id, text in zip(context_ids, fact_texts)]
        qa = {
            "qa_id": qa_id,
            "query": question,
            "question": question,
            "raw_question": normalize_space(row.get("question")),
            "choices": [{"label": label, "text": text}
                        for label, text in options],
            "answer": answer,
            "answer_key": answer_key,
            "combined_fact": normalize_space(row.get("combinedfact")),
            "question_type": "multiple_choice_sentence_composition",
            "gold_doc_ids": gold_doc_ids,
            "context_sentence_ids": context_ids,
            "evidence_list": evidence,
            "gold_context": fact_texts,
        }
        qa_rows.append(qa)
        gold_rows.append({
            "qa_id": qa_id,
            "query": question,
            "answer": answer,
            "answer_key": answer_key,
            "context_sentence_ids": context_ids,
            "gold_contexts": fact_texts,
        })

    if not qa_rows:
        raise ValueError("No valid labeled QASC examples were prepared")

    write_json(args.data_dir / "corpus" / "corpus.json", documents)
    write_jsonl(args.data_dir / "qa" / "qasc.jsonl", qa_rows)
    write_jsonl(args.data_dir / "gold_contexts.jsonl", gold_rows)
    write_json(args.data_dir / "metadata.json", {
        "dataset": "QASC",
        "source": source_description,
        "split": args.hf_split,
        "selection": "first_n_in_official_order",
        "requested_queries": args.max_queries,
        "queries": len(qa_rows),
        "skipped_queries": skipped,
        "reasoning_unit_definition": "one unique QASC fact1/fact2 sentence",
        "combinedfact_in_corpus": False,
        "documents": len(documents),
        "reasoning_units": len(raw_units),
        "corpus_characters": sum(len(row["body"]) for row in documents),
    })

    extraction = args.run_dir / "01_extraction"
    write_jsonl(extraction / "units.jsonl", raw_units)
    write_jsonl(extraction / "errors.jsonl", [])
    write_json(extraction / "summary.json", {
        "documents": len(documents),
        "chunks": len(documents),
        "units": len(raw_units),
        "errors": 0,
        "model": "none (QASC annotated-fact-as-unit adapter)",
        "reasoning_unit_definition": "one unique QASC fact1/fact2 sentence",
    })

    print(f"[done] queries: {len(qa_rows)}")
    print(f"[done] unique fact units: {len(raw_units)}")
    print(f"[done] skipped queries: {len(skipped)}")
    print(f"[done] data: {args.data_dir}")
    print(f"[done] extraction checkpoint: {extraction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
