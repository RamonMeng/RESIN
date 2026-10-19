#!/usr/bin/env python3
"""Evaluate an existing RESIN run with the Ragas metrics.

This script does not rerun extraction, retrieval, graph construction, or answer
generation. It reads the old-pipeline artifacts and evaluates the exact Ragas
metric implementations:

  * AnswerRelevancy (Ragas AR)
  * LLM-based ContextRecall (Ragas CR)

The evaluator LLM is supplied through an OpenAI-compatible API endpoint. The
default is Gemini-3.5-Flash-Lite through Google's OpenAI-compatible Gemini
endpoint. The Ragas metrics
themselves are unchanged.

The script also writes CR_ID, a deterministic sentence-ID overlap diagnostic.
CR_ID is not the main CR reported by this script and should not be confused
with Ragas' LLM-based ContextRecall.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--qa_file", type=Path, required=True)
    p.add_argument(
        "--corpus_file", type=Path, required=True,
        help=("Aligned corpus used for CR_ID diagnostics and, when requested, "
              "for mapping retrieved units back to complete source passages."),
    )
    p.add_argument("--answers_file", type=Path, default=None)
    p.add_argument("--retrieval_file", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument(
        "--judge_provider", choices=["gemini", "openai"],
        default=os.environ.get("RESIN_JUDGE_PROVIDER", "gemini"),
    )
    p.add_argument(
        "--judge_model",
        default=os.environ.get("RESIN_JUDGE_MODEL", "gemini-3.5-flash-lite"),
    )
    p.add_argument(
        "--base_url",
        default=os.environ.get("RESIN_JUDGE_BASE_URL", ""),
    )
    p.add_argument("--api_key", default=None)
    p.add_argument(
        "--embedding_model", default="all-MiniLM-L6-v2",
        help="HuggingFace/Sentence-Transformers model used by Ragas AR.",
    )
    p.add_argument("--strictness", type=int, default=3)
    p.add_argument(
        "--gold_ar_diagnostic", action="store_true",
        help=("Also score each gold answer with the same Ragas AnswerRelevancy "
              "metric. This diagnoses short-answer/answer-style bias and is not "
              "the system AR reported for comparison."),
    )
    p.add_argument(
        "--max_contexts", type=int, default=25,
        help=("Maximum top-ranked retrieved reasoning units considered per query. "
              "In source_passage mode, these units are mapped to passages and "
              "duplicate passages are removed after this cutoff."),
    )
    p.add_argument(
        "--max_context_chars", type=int, default=0,
        help="Optional total character cap for retrieved contexts; 0 means no extra cap.",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument(
        "--cr_only", action="store_true",
        help=("Compute only official Ragas LLM ContextRecall. This mode needs "
              "retrieval outputs and reference answers, but no agent answers."),
    )
    p.add_argument(
        "--cr_context_mode",
        choices=["reasoning_unit", "source_passage"],
        default="source_passage",
        help=("Text representation passed to Ragas ContextRecall. reasoning_unit "
              "uses the extracted structured unit (legacy behavior); "
              "source_passage maps every retrieved unit back to its complete "
              "corpus passage and removes duplicate passages while preserving "
              "the retrieval order."),
    )
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def query_id(index: int, query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"q_{index:06d}_{digest}"


def flatten_ids(value: Any) -> list[str]:
    """Extract sentence IDs from common HolisQA representations."""
    found: list[str] = []
    if isinstance(value, str):
        found.extend(x for x in re.split(r"[,;\s]+", value.strip()) if x)
    elif isinstance(value, list):
        for item in value:
            found.extend(flatten_ids(item))
    elif isinstance(value, dict):
        for key in (
            "context_sentence_ids", "gold_context_sentence_ids", "sentence_ids",
            "source_sentence_ids", "sentence_id", "id",
        ):
            if key in value:
                found.extend(flatten_ids(value[key]))
    return list(dict.fromkeys(found))


def gold_sentence_ids(row: dict[str, Any]) -> list[str]:
    for key in (
        "context_sentence_ids", "gold_context_sentence_ids", "evidence_sentence_ids",
        "evidence_list", "gold_context", "contexts",
    ):
        if key in row:
            ids = flatten_ids(row[key])
            if ids:
                return ids
    return []


def load_source_map(corpus_file: Path) -> dict[str, list[str]]:
    """Map semantic-passage IDs and document IDs to source sentence IDs."""
    raw = read_json(corpus_file)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("corpus", []))
    if not isinstance(raw, list):
        return {}
    result: dict[str, list[str]] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        ids = flatten_ids(
            row.get("source_sentence_ids") or row.get("sentence_ids") or row.get("sentence_id")
        )
        keys = [
            row.get("passage_id"), row.get("id"), row.get("document_id"),
            row.get("article_id"), row.get("parent_document_id"),
        ]
        if ids:
            for key in keys:
                if key:
                    result.setdefault(str(key), [])
                    result[str(key)].extend(ids)
    return {key: list(dict.fromkeys(value)) for key, value in result.items()}


def load_passage_text_map(corpus_file: Path) -> dict[str, str]:
    """Map passage/document IDs to the complete source text used by retrieval.

    Only IDs that identify the concrete corpus row are indexed.  In particular,
    parent/article IDs are not mapped because one article can contain multiple
    semantic passages and silently choosing one would corrupt the evaluation.
    """
    raw = read_json(corpus_file)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("corpus", []))
    if not isinstance(raw, list):
        return {}
    result: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        body = str(row.get("body") or row.get("text") or row.get("content") or "").strip()
        if not body:
            continue
        title = str(row.get("title") or "").strip()
        text = f"Title: {title}\n{body}" if title else body
        for key in (row.get("passage_id"), row.get("id"), row.get("document_id")):
            if key not in (None, ""):
                result[str(key)] = text
    return result


def answer_text(row: dict[str, Any]) -> str:
    value = row.get("answer", "")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def candidate_unit(candidate: dict[str, Any]) -> dict[str, Any]:
    value = candidate.get("unit", candidate)
    return value if isinstance(value, dict) else {}


def retrieved_sentence_ids(
    retrieval: dict[str, Any], source_map: dict[str, list[str]]
) -> list[str]:
    found: list[str] = []
    for candidate in retrieval.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        unit = candidate_unit(candidate)
        direct = (
            unit.get("source_sentence_ids") or unit.get("sentence_ids")
            or unit.get("source_sentence_id")
        )
        ids = flatten_ids(direct)
        if not ids:
            for key in (
                unit.get("passage_id"), unit.get("document_id"),
                unit.get("doc_id"), unit.get("article_id"),
            ):
                if key is not None and str(key) in source_map:
                    ids.extend(source_map[str(key)])
        found.extend(ids)
    return list(dict.fromkeys(found))


def unit_to_context(unit: dict[str, Any]) -> str:
    """Serialize the exact reasoning unit shown to the RESIN agent."""
    fields: list[str] = []
    for label, key in (
        ("kind", "kind"), ("subject", "subject"), ("subject_kind", "subject_kind"),
        ("predicate", "predicate"), ("object", "object"),
        ("conditions", "conditions"), ("conclusion", "conclusion"),
        ("evidence", "evidence_quote"),
    ):
        value = unit.get(key)
        if value not in (None, "", [], {}):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            fields.append(f"{label}: {value}")
    return "\n".join(fields).strip()


def unit_passage_id(unit: dict[str, Any], passage_map: dict[str, str]) -> str | None:
    """Resolve a normalized reasoning unit to its concrete source passage."""
    document = unit.get("document", {})
    if not isinstance(document, dict):
        document = {}
    for value in (
        unit.get("passage_id"), unit.get("doc_id"), unit.get("document_id"),
        document.get("passage_id"), document.get("document_id"),
    ):
        if value not in (None, "") and str(value) in passage_map:
            return str(value)
    return None


def retrieved_contexts(
    retrieval: dict[str, Any], max_contexts: int, max_chars: int,
    context_mode: str, passage_map: dict[str, str],
) -> list[str]:
    contexts: list[str] = []
    seen_passages: set[str] = set()
    for candidate in retrieval.get("candidates", [])[:max(0, max_contexts)]:
        if not isinstance(candidate, dict):
            continue
        unit = candidate_unit(candidate)
        if context_mode == "source_passage":
            passage_id = unit_passage_id(unit, passage_map)
            if passage_id is not None:
                if passage_id in seen_passages:
                    continue
                seen_passages.add(passage_id)
                text = passage_map[passage_id]
            else:
                # Keep the query evaluable for older artifacts that do not
                # retain a resolvable source ID; record-level diagnostics below
                # make these fallbacks visible.
                text = unit_to_context(unit)
        else:
            text = unit_to_context(unit)
        if text:
            contexts.append(text)
    if max_chars > 0:
        clipped: list[str] = []
        used = 0
        for text in contexts:
            remaining = max_chars - used
            if remaining <= 0:
                break
            clipped.append(text[:remaining])
            used += min(len(text), remaining)
        contexts = clipped
    return contexts


def context_mapping_diagnostics(
    retrieval: dict[str, Any], max_contexts: int, passage_map: dict[str, str]
) -> dict[str, int]:
    candidates = [
        candidate for candidate in retrieval.get("candidates", [])[:max(0, max_contexts)]
        if isinstance(candidate, dict)
    ]
    passage_ids = [
        unit_passage_id(candidate_unit(candidate), passage_map)
        for candidate in candidates
    ]
    resolved = [value for value in passage_ids if value is not None]
    return {
        "candidate_units_considered": len(candidates),
        "resolved_candidate_units": len(resolved),
        "unresolved_candidate_units": len(candidates) - len(resolved),
        "unique_source_passages": len(set(resolved)),
    }


def install_ragas_optional_import_shims() -> None:
    """Bridge Ragas' legacy optional Vertex-AI import for OpenAI-only runs.

    Some Ragas releases import these classes at module import time even when
    the evaluator uses OpenAI. Newer langchain-community releases moved the
    Vertex-AI integration to langchain-google-vertexai. The classes are only
    used for optional isinstance checks on this path, so a compatibility module
    is sufficient and does not alter the Ragas metrics or prompts.
    """
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        try:
            from langchain_google_vertexai import ChatVertexAI  # type: ignore
        except Exception:
            ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore
        shim = types.ModuleType(module_name)
        shim.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        sys.modules[module_name] = shim
        try:
            import langchain_community.chat_models as chat_models
            chat_models.vertexai = shim  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        from langchain_community import llms as community_llms
        if not hasattr(community_llms, "VertexAI"):
            try:
                from langchain_google_vertexai import VertexAI  # type: ignore
            except Exception:
                VertexAI = type("VertexAI", (), {})  # type: ignore
            community_llms.VertexAI = VertexAI  # type: ignore[attr-defined]
    except Exception:
        pass


def load_ragas(args: argparse.Namespace) -> dict[str, Any]:
    """Construct Ragas metrics without reimplementing their prompts or scores."""
    install_ragas_optional_import_shims()
    try:
        import ragas
    except Exception as exc:
        raise SystemExit(
            "Ragas could not be imported in the current Python environment.\n"
            f"Actual import error: {type(exc).__name__}: {exc}\n"
            "Check with:\n"
            "  python3 -m pip show ragas\n"
            "  python3 -c 'import ragas; print(ragas.__version__)'"
        ) from exc
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI SDK with: pip install -U openai") from exc
    try:
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise SystemExit(
            "This evaluator requires a Ragas release exposing ragas.llms.llm_factory. "
            "Upgrade with: pip install -U ragas"
        ) from exc
    try:
        from ragas.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from ragas.embeddings.base import HuggingFaceEmbeddings
        except ImportError:
            try:
                from ragas.embeddings import HuggingfaceEmbeddings as HuggingFaceEmbeddings
            except ImportError as exc:
                raise SystemExit(
                    "This Ragas installation has no HuggingFace embedding wrapper. "
                    "Upgrade with: pip install -U ragas"
                ) from exc
    try:
        from ragas.metrics.collections import AnswerRelevancy
    except ImportError:
        try:
            from ragas.metrics import AnswerRelevancy
        except ImportError as exc:
            raise SystemExit("Cannot import Ragas AnswerRelevancy.") from exc
    try:
        from ragas.metrics.collections import ContextRecall
    except ImportError:
        try:
            from ragas.metrics import ContextRecall
        except ImportError:
            try:
                from ragas.metrics import LLMContextRecall as ContextRecall
            except ImportError as exc:
                raise SystemExit("Cannot import Ragas LLM ContextRecall.") from exc
    try:
        from ragas import SingleTurnSample
    except ImportError:
        try:
            from ragas.dataset_schema import SingleTurnSample
        except ImportError as exc:
            raise SystemExit("Cannot import Ragas SingleTurnSample.") from exc

    if not args.base_url:
        if args.judge_provider == "gemini":
            # Gemini's OpenAI-compatible endpoint lets Ragas keep using its
            # official OpenAI client adapter without reimplementing metrics.
            args.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        else:
            args.base_url = "https://api.openai.com/v1"
    if not args.api_key:
        env_name = "GEMINI_API_KEY" if args.judge_provider == "gemini" else "OPENAI_API_KEY"
        args.api_key = os.environ.get(env_name, "")
    if not args.api_key:
        env_name = "GEMINI_API_KEY" if args.judge_provider == "gemini" else "OPENAI_API_KEY"
        raise SystemExit(f"{env_name} is empty. Export it or pass --api_key.")
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url.rstrip("/"))
    # The Ragas implementation is unchanged. Gemini is selected through its
    # OpenAI-compatible endpoint, so Ragas still uses its normal OpenAI adapter.
    try:
        evaluator_llm = llm_factory(args.judge_model, provider="openai", client=client)
    except TypeError:
        evaluator_llm = llm_factory(args.judge_model, client=client)

    embedding_model = None
    ar_metric = None
    if not args.cr_only:
        embedding_model = args.embedding_model
        if embedding_model == "all-MiniLM-L6-v2":
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            evaluator_embeddings = HuggingFaceEmbeddings(model=embedding_model)
        except TypeError:
            evaluator_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        try:
            ar_metric = AnswerRelevancy(
                llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=args.strictness
            )
        except TypeError:
            ar_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
            if hasattr(ar_metric, "strictness"):
                ar_metric.strictness = args.strictness
    cr_metric = ContextRecall(llm=evaluator_llm)
    try:
        ragas_version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        ragas_version = getattr(ragas, "__version__", "unknown")
    return {
        "SingleTurnSample": SingleTurnSample,
        "answer_relevancy": ar_metric,
        "context_recall": cr_metric,
        "ragas_version": ragas_version,
        "embedding_model_resolved": embedding_model,
    }


async def metric_ascore(metric: Any, sample: Any, kind: str) -> float:
    """Call modern collection or legacy single-turn Ragas APIs."""
    single_turn = getattr(metric, "single_turn_ascore", None)
    if single_turn is not None:
        value = single_turn(sample)
    else:
        method = getattr(metric, "ascore", None)
        if method is None:
            raise RuntimeError(f"Ragas metric {type(metric).__name__} has no async score method")
        if kind == "ar":
            value = method(user_input=sample.user_input, response=sample.response)
        else:
            value = method(
                user_input=sample.user_input,
                retrieved_contexts=sample.retrieved_contexts,
                reference=sample.reference,
            )
    if hasattr(value, "__await__"):
        value = await value
    if hasattr(value, "value"):
        value = value.value
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Ragas returned a non-finite score: {result}")
    return result


async def score_one(
    item: dict[str, Any], runtime: dict[str, Any], output: Path,
    semaphore: asyncio.Semaphore, resume: bool, gold_ar_diagnostic: bool,
    cr_only: bool,
) -> dict[str, Any]:
    path = output / "ragas" / f"{item['query_id']}.json"
    if resume and path.exists():
        cached = read_json(path)
        diagnostic_ready = (cr_only or not gold_ar_diagnostic or
                            cached.get("gold_answer_relevancy") is not None or
                            cached.get("gold_ar_error"))
        context_ready = cached.get("context_signature") == item.get("context_signature")
        if cached.get("status") == "ok" and diagnostic_ready and context_ready:
            return cached
    record: dict[str, Any] = {
        "status": "error", "query_id": item["query_id"], "row_index": item["row_index"],
        "query": item["query"], "answer": item["answer"],
        "gold_answer": item["gold_answer"],
        "retrieved_contexts": item["retrieved_contexts"],
        "cr_context_mode": item["cr_context_mode"],
        "context_signature": item["context_signature"],
        "context_mapping": item["context_mapping"],
        "metric": ("Ragas LLMContextRecall" if cr_only
                   else "Ragas AnswerRelevancy + LLMContextRecall"),
    }
    try:
        async with semaphore:
            sample = runtime["SingleTurnSample"](
                user_input=item["query"], response=item["answer"],
                retrieved_contexts=item["retrieved_contexts"], reference=item["gold_answer"],
            )
            cr = await metric_ascore(runtime["context_recall"], sample, "cr")
            ar = (None if cr_only else
                  await metric_ascore(runtime["answer_relevancy"], sample, "ar"))
        record.update({"status": "ok", "context_recall": cr})
        if not cr_only:
            record["answer_relevancy"] = ar
        if not cr_only and gold_ar_diagnostic and item["gold_answer"].strip():
            try:
                async with semaphore:
                    gold_sample = runtime["SingleTurnSample"](
                        user_input=item["query"], response=item["gold_answer"],
                    )
                    gold_ar = await metric_ascore(
                        runtime["answer_relevancy"], gold_sample, "ar"
                    )
                record["gold_answer_relevancy"] = gold_ar
            except Exception as exc:
                # The diagnostic must never invalidate an otherwise successful
                # official AR/CR evaluation.
                record["gold_ar_error"] = repr(exc)
    except Exception as exc:
        record["error"] = repr(exc)
    write_json(path, record)
    return record


async def score_all(
    items: list[dict[str, Any]], runtime: dict[str, Any], output: Path,
    workers: int, resume: bool, gold_ar_diagnostic: bool, cr_only: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, workers))
    tasks = [
        asyncio.create_task(score_one(
            item, runtime, output, semaphore, resume, gold_ar_diagnostic, cr_only
        ))
        for item in items
    ]
    results: list[dict[str, Any]] = []
    for done, task in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await task)
        if done % 10 == 0 or done == len(tasks):
            print(f"[Ragas] completed {done} / {len(tasks)}")
    return sorted(results, key=lambda x: x["query_id"])


def mean_score(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    answers_file = args.answers_file or args.run_dir / "07_agent" / "answers.jsonl"
    retrieval_file = args.retrieval_file or args.run_dir / "05_retrieval" / "retrieval.jsonl"
    output = args.output_dir or args.run_dir / ("09_cr_only" if args.cr_only else "09_ar_cr")
    answers = ({
        row["query_id"]: row for row in read_jsonl(answers_file) if row.get("query_id")
    } if not args.cr_only else {})
    retrieval = {
        row["query_id"]: row for row in read_jsonl(retrieval_file) if row.get("query_id")
    }
    qa_rows = read_jsonl(args.qa_file)
    source_map = load_source_map(args.corpus_file)
    passage_map = load_passage_text_map(args.corpus_file)

    items: list[dict[str, Any]] = []
    for index, qa in enumerate(qa_rows):
        query = str(qa.get("query", ""))
        qid = query_id(index, query)
        answer = answers.get(qid, {})
        ret = retrieval.get(qid, {})
        gold_ids = gold_sentence_ids(qa)
        retrieved_ids = retrieved_sentence_ids(ret, source_map)
        covered = sorted(set(gold_ids) & set(retrieved_ids))
        id_cr = len(covered) / len(gold_ids) if gold_ids else None
        contexts = retrieved_contexts(
            ret, args.max_contexts, args.max_context_chars,
            args.cr_context_mode, passage_map,
        )
        context_mapping = context_mapping_diagnostics(
            ret, args.max_contexts, passage_map
        )
        context_signature = hashlib.sha1(json.dumps(
            {"mode": args.cr_context_mode, "contexts": contexts},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        items.append({
            "query_id": qid, "row_index": index, "query": query,
            "answer": answer_text(answer), "gold_answer": str(qa.get("answer", "") or ""),
            "agent_status": answer.get("status", ""),
            "retrieved_contexts": contexts,
            "cr_context_mode": args.cr_context_mode,
            "context_signature": context_signature,
            "context_mapping": context_mapping,
            "gold_context_sentence_ids": gold_ids,
            "retrieved_source_sentence_ids": retrieved_ids,
            "covered_sentence_ids": covered, "CR_ID": id_cr,
        })

    runtime = load_ragas(args)
    write_json(output / "config.json", {
        "run_dir": str(args.run_dir), "qa_file": str(args.qa_file),
        "corpus_file": str(args.corpus_file), "answers_file": str(answers_file),
        "retrieval_file": str(retrieval_file),
        "judge_provider": args.judge_provider, "judge_model": args.judge_model,
        "base_url": args.base_url,
        "embedding_model_requested": args.embedding_model,
        "embedding_model_resolved": runtime["embedding_model_resolved"],
        "ragas_version": runtime["ragas_version"], "strictness": args.strictness,
        "gold_ar_diagnostic": args.gold_ar_diagnostic,
        "max_contexts": args.max_contexts, "max_context_chars": args.max_context_chars,
        "cr_context_mode": args.cr_context_mode,
        "source_passages_loaded": len(passage_map),
        "workers": args.workers, "cr_only": args.cr_only,
        "metric_definition": {
            "AR": "Ragas AnswerRelevancy",
            "CR": "Ragas LLM-based ContextRecall using the reference answer",
            "CR_ID": "diagnostic overlap of gold and retrieved sentence IDs; not the main CR",
            "Gold_AR": ("optional Ragas AnswerRelevancy of the gold answer; "
                        "diagnostic only, not a system score"),
        },
    })

    results = asyncio.run(score_all(
        items, runtime, output, args.workers, not args.no_resume,
        args.gold_ar_diagnostic, args.cr_only,
    ))
    write_jsonl(output / "ragas_per_query.jsonl", results)
    successful = [row for row in results if row.get("status") == "ok"]
    cr_id_values = [row["CR_ID"] for row in items if row.get("CR_ID") is not None]
    ar = None if args.cr_only else mean_score(successful, "answer_relevancy")
    gold_ar = mean_score(successful, "gold_answer_relevancy")
    cr = mean_score(successful, "context_recall")
    cr_id = sum(cr_id_values) / len(cr_id_values) if cr_id_values else 0.0
    context_counts = [len(item["retrieved_contexts"]) for item in items]
    context_chars = [sum(len(text) for text in item["retrieved_contexts"]) for item in items]
    unresolved = [item["context_mapping"]["unresolved_candidate_units"] for item in items]
    combined = {
        "evaluated_queries": len(items), "successful_ragas_queries": len(successful),
        "AR": None if ar is None else 100.0 * ar,
        "CR": 100.0 * cr, "CR_ID": 100.0 * cr_id,
        "ar_0_to_1": ar, "cr_0_to_1": cr, "cr_id_0_to_1": cr_id,
        "successful_ar_queries": sum(row.get("answer_relevancy") is not None for row in results),
        "successful_cr_queries": sum(row.get("context_recall") is not None for row in results),
        "queries_with_gold_context": len(cr_id_values),
        "judge_provider": args.judge_provider, "judge_model": args.judge_model,
        "base_url": args.base_url,
        "embedding_model": runtime["embedding_model_resolved"],
        "ragas_version": runtime["ragas_version"], "strictness": args.strictness,
        "max_contexts": args.max_contexts,
        "cr_context_mode": args.cr_context_mode,
        "average_context_items": (
            sum(context_counts) / len(context_counts) if context_counts else 0.0
        ),
        "average_context_characters": (
            sum(context_chars) / len(context_chars) if context_chars else 0.0
        ),
        "unresolved_candidate_units": sum(unresolved),
        "cr_only": args.cr_only,
        "note": ("CR is produced directly by Ragas without running the agent; "
                 "CR_ID is only a diagnostic." if args.cr_only else
                 "AR and CR are produced by Ragas directly; CR_ID is only a diagnostic."),
    }
    if args.gold_ar_diagnostic:
        combined.update({
            "Gold_AR_diagnostic": 100.0 * gold_ar,
            "gold_ar_0_to_1": gold_ar,
            "successful_gold_ar_queries": sum(
                row.get("gold_answer_relevancy") is not None for row in results
            ),
            "gold_ar_note": (
                "Gold_AR is the same Ragas AR applied to reference answers. "
                "It measures answer-style bias and is not a system score."
            ),
        })
    write_json(output / "summary.json", combined)
    by_id = {row["query_id"]: row for row in results}
    write_jsonl(output / "per_query.jsonl", [
        {**item,
         "answer_relevancy": by_id.get(item["query_id"], {}).get("answer_relevancy"),
         "gold_answer_relevancy": by_id.get(item["query_id"], {}).get(
             "gold_answer_relevancy"
         ),
         "context_recall": by_id.get(item["query_id"], {}).get("context_recall"),
         "ragas_status": by_id.get(item["query_id"], {}).get("status", "error"),
         "ragas_error": by_id.get(item["query_id"], {}).get("error")}
        for item in items
    ])
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    print("[done] artifacts:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
