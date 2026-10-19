#!/usr/bin/env python3
"""Prepare HolisQA with EnSI-RAG-style entity-centered passages.

The official HolisQA articles are first represented as temporary sentence
coordinates. An LLM then selects contiguous, semantically self-contained
spans centered on one entity or evidence-bearing topic. The selected passage
text is reconstructed from the original sentences, so no paraphrasing or
fixed-size passage boundary is introduced.

Gold evidence is retained only for post-hoc evaluation. It is never included
in the passage-construction prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "holisqa"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "holisqa_resin_semantic"


PASSAGE_SYSTEM = """You are the offline passage-construction module of EnSI-RAG.

Construct query-independent passages from a scientific article. Do not answer
any question and do not summarize the whole article.

EnSI-RAG passage contract:
- Each passage has one primary entity, mechanism, intervention, finding,
  method, outcome, or other evidence-bearing topic.
- Keep adjacent sentences together when they jointly define the entity or
  explain its properties, relations, mechanism, or result.
- Split when the main evidence center changes.
- Passage lengths are semantic and variable. Do not use a fixed number of
  sentences, tokens, or characters as the passage definition.
- Return only contiguous ranges of the supplied source sentences.
- Do not paraphrase, reorder, or add outside knowledge.
- Unit IDs are 1-based integers. Copy start_unit and end_unit from the
provided unit labels. Valid IDs range from 1 to N.

Return JSON only in this form:
{"passages":[{"start_unit":1,"end_unit":3,"main_entity":"...",
"entity_type":"...","semantic_focus":"...",
"evidence_role":"..."}]}
"""


PASSAGE_USER = """Construct EnSI-RAG passages for this source article.

The numbered units are temporary addressing coordinates. The program will
recover the exact passage text from the original source sentences.

Rules:
1. Use inclusive contiguous ranges: start_unit <= end_unit.
2. Cover substantive evidence throughout the article.
3. Use the smallest range that remains semantically self-contained.
4. Do not use questions, answers, gold evidence, or qrels; they are not
   provided and passage construction must remain query-independent.
5. Metadata must be supported by the selected sentences.

ARTICLE_ID: {article_id}
ARTICLE_TITLE: {title}
SOURCE SENTENCE UNITS:
{units}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_hash(value: Any, length: int = 12) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def safe(value: Any) -> str:
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:140].strip("._") or "item")


def scoped_holisqa_id(domain: str, raw_id: Any) -> str:
    """Make an official HolisQA ID globally unique across its five domains.

    HolisQA restarts article IDs (A000001, ...) and sentence IDs (S00000001,
    ...) inside every domain.  Using those raw IDs as global keys silently
    overwrites checkpoints and can attach one domain's passage spans to a
    different domain's article.  Keep the original IDs in separate metadata
    fields and use this scoped form everywhere that acts as a key.
    """
    return f"{domain}__{str(raw_id)}"


def parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [x for x in (cleaned.find("{"), cleaned.find("[")) if x >= 0]
        if not starts:
            raise ValueError("No JSON found in LLM response: " + cleaned[:500])
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        value = json.loads(cleaned[start:end + 1])
    if isinstance(value, list):
        return {"passages": value}
    return value if isinstance(value, dict) else {"passages": []}


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout: float, retries: int, provider: str = "openai",
                 reasoning_effort: str = "minimal"):
        self.base = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        if self.provider == "openai" and not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    def call_json(self, user_prompt: str, max_tokens: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
        request_started = time.perf_counter()
        request_id = uuid.uuid4().hex
        messages = [
            {"role": "system", "content": PASSAGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            (
                "max_tokens"
                if self.provider == "vllm"
                else "max_completion_tokens"
            ): max_tokens,
        }

        # GPT-5 reasoning models only accept their default temperature. Do
        # not send temperature=0.0 when a GPT-5 model is selected, including
        # when the request is routed through a vLLM-compatible endpoint.
        if self.provider == "vllm" and not self.model.lower().startswith("gpt-5"):
            payload["temperature"] = 0.0

        if self.provider == "vllm":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["response_format"] = {"type": "json_object"}
            if (self.model.lower().startswith("gpt-5")
                    and self.reasoning_effort != "default"):
                payload["reasoning_effort"] = self.reasoning_effort
        variants = [payload]
        fallback = dict(payload)
        fallback.pop("response_format", None)
        fallback.pop("chat_template_kwargs", None)
        variants.append(fallback)
        last_error = ""
        for attempt in range(self.retries):
            for variant_index, body in enumerate(variants):
                try:
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = "Bearer " + self.api_key
                    response = requests.post(
                        self.base + "/chat/completions",
                        headers=headers,
                        json=body,
                        timeout=self.timeout,
                    )
                    if response.status_code >= 400:
                        last_error = f"HTTP {response.status_code}: {response.text[:800]}"
                        continue
                    data = response.json()
                    content = data["choices"][0]["message"].get("content", "")
                    if isinstance(content, list):
                        content = "".join(str(x.get("text", x)) for x in content)
                    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                    prompt_details = (usage.get("prompt_tokens_details")
                                      if isinstance(usage.get("prompt_tokens_details"), dict) else {})
                    completion_details = (usage.get("completion_tokens_details")
                                          if isinstance(usage.get("completion_tokens_details"), dict) else {})
                    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                    reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                    metrics = {
                        "request_id": request_id,
                        "provider": self.provider,
                        "requested_model": self.model,
                        "response_model": data.get("model", self.model),
                        "latency_seconds": round(time.perf_counter() - request_started, 6),
                        "attempts_used": attempt + 1,
                        "successful_variant": variant_index,
                        "prompt_tokens": prompt_tokens,
                        "cached_prompt_tokens": prompt_details.get("cached_tokens", 0),
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "visible_completion_tokens": (
                            max(0, int(completion_tokens) - int(reasoning_tokens or 0))
                            if completion_tokens is not None else None
                        ),
                        "total_tokens": usage.get("total_tokens"),
                    }
                    return parse_json(str(content)), str(content), metrics
                except Exception as exc:
                    last_error = repr(exc)
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Passage construction failed: " + last_error)


def load_domain(source_dir: Path, domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    domain_dir = source_dir / domain
    articles = {str(x["article_id"]): dict(x)
                for x in read_jsonl(domain_dir / "articles.jsonl")}
    sentences = read_jsonl(domain_dir / "sentences.jsonl")
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sentence in sentences:
        by_article[str(sentence["article_id"])].append(sentence)
    for rows in by_article.values():
        rows.sort(key=lambda x: int(x.get("sent_idx", 0)))

    docs: list[dict[str, Any]] = []
    for article_id, article in articles.items():
        scoped_article_id = scoped_holisqa_id(domain, article_id)
        source_sentences = by_article.get(article_id, [])
        units = []
        for index, sentence in enumerate(source_sentences, start=1):
            text = norm_space(sentence.get("text", ""))
            if text:
                raw_sentence_id = str(sentence.get("sentence_id", ""))
                units.append({
                    "unit_id": index,
                    "sentence_id": scoped_holisqa_id(domain, raw_sentence_id),
                    "original_sentence_id": raw_sentence_id,
                    "text": text,
                    "sent_idx": sentence.get("sent_idx", index - 1),
                })
        docs.append({
            "document_id": scoped_article_id,
            "title": article.get("title", ""),
            "source": "HolisQA",
            "author": "; ".join(article.get("authors", [])),
            "published_at": article.get("publication_date", ""),
            "url": article.get("openalex_id", ""),
            "category": domain,
            "article_id": scoped_article_id,
            "original_article_id": article_id,
            "units": units,
            "body": " ".join(x["text"] for x in units),
        })

    sentence_by_id = {str(x["sentence_id"]): x for x in sentences}
    qas: list[dict[str, Any]] = []
    for qa in read_jsonl(domain_dir / "qa.jsonl"):
        evidence = []
        scoped_context_sentence_ids = []
        for sentence_id in qa.get("context_sentence_ids", []):
            raw_sentence_id = str(sentence_id)
            sentence = sentence_by_id.get(raw_sentence_id)
            if sentence is None:
                continue
            raw_article_id = str(sentence["article_id"])
            scoped_article_id = scoped_holisqa_id(domain, raw_article_id)
            scoped_sentence_id = scoped_holisqa_id(domain, raw_sentence_id)
            scoped_context_sentence_ids.append(scoped_sentence_id)
            article = articles.get(raw_article_id, {})
            evidence.append({
                "sentence_id": scoped_sentence_id,
                "original_sentence_id": raw_sentence_id,
                "article_id": scoped_article_id,
                "original_article_id": raw_article_id,
                "fact": sentence.get("text", ""),
                "source": "HolisQA",
                "title": article.get("title", ""),
                "url": article.get("openalex_id", ""),
                "category": domain,
            })
        qas.append({
            "qa_id": f"{domain}:{qa.get('qa_id', '')}",
            "original_qa_id": qa.get("qa_id", ""),
            "category": domain,
            "slice_id": scoped_holisqa_id(domain, qa.get("slice_id", "")),
            "original_slice_id": qa.get("slice_id", ""),
            "query": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "context_sentence_ids": scoped_context_sentence_ids,
            "original_context_sentence_ids": qa.get("context_sentence_ids", []),
            "context_text": qa.get("context_text", ""),
            "evidence_list": evidence,
        })
    return docs, qas


SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])(?:[\"'”’\)\]]+)?(?=\s+(?:[A-Z0-9\[(\"'“‘])|$)"
)


def split_body_units(text: str) -> list[dict[str, Any]]:
    """Create temporary sentence-like coordinates for an already prepared corpus."""
    text = norm_space(text)
    if not text:
        return []
    boundaries = [match.end() for match in SENTENCE_BOUNDARY.finditer(text)]
    if not boundaries or boundaries[-1] < len(text):
        boundaries.append(len(text))
    units = []
    start = 1
    for end in boundaries:
        value = text[start:end].strip()
        if value:
            units.append({
                "unit_id": len(units) + 1,
                "sentence_id": f"unit_{len(units) + 1:04d}",
                "text": value,
                "sent_idx": len(units),
            })
        start = end
    return units or [{"unit_id": 1, "sentence_id": "unit_0001", "text": text, "sent_idx": 0}]


def load_prepared_bundle(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the corpus/QA bundle produced by prepare_holisqa_resin_semantic.py."""
    corpus_path = source_dir / "corpus" / "corpus.json"
    qa_candidates = [source_dir / "qa" / "holisqa.jsonl",
                     source_dir / "qa" / "train.jsonl"]
    qa_path = next((path for path in qa_candidates if path.exists()), None)
    if not corpus_path.exists() or qa_path is None:
        raise FileNotFoundError(
            "Expected either official HolisQA files or a prepared bundle with "
            "corpus/corpus.json and qa/holisqa.jsonl under " + str(source_dir)
        )
    raw_corpus = read_json(corpus_path)
    if isinstance(raw_corpus, dict):
        raw_corpus = raw_corpus.get("data", raw_corpus.get("corpus", []))
    if not isinstance(raw_corpus, list):
        raise ValueError("No corpus record list found in " + str(corpus_path))

    docs = []
    for index, row in enumerate(raw_corpus):
        if not isinstance(row, dict):
            continue
        document_id = str(row.get("document_id") or row.get("id") or f"doc_{index:06d}")
        body = str(row.get("body") or row.get("text") or "")
        docs.append({
            "document_id": document_id,
            "title": row.get("title", ""),
            "source": row.get("source", "HolisQA"),
            "author": row.get("author", ""),
            "published_at": row.get("published_at", ""),
            "url": row.get("url", ""),
            "category": row.get("category", ""),
            "article_id": row.get("article_id") or row.get("parent_document_id") or document_id,
            "units": split_body_units(body),
            "body": body,
        })
    return docs, read_jsonl(qa_path)


def make_windows(units: list[dict[str, Any]], size: int, overlap: int) -> list[list[dict[str, Any]]]:
    if size <= 0 or len(units) <= size:
        return [units] if units else []
    overlap = max(0, min(overlap, size - 1))
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(units), step):
        window = units[start:start + size]
        if window:
            windows.append(window)
        if start + size >= len(units):
            break
    return windows


def qa_evidence_article_ids(qa: dict[str, Any]) -> set[str]:
    """Return the source article IDs needed to answer one HolisQA item."""
    article_ids: set[str] = set()
    evidence = qa.get("evidence_list", [])
    if not isinstance(evidence, list):
        return article_ids
    for item in evidence:
        if not isinstance(item, dict):
            continue
        article_id = (item.get("article_id") or item.get("document_id")
                      or item.get("parent_document_id"))
        if article_id:
            article_ids.add(str(article_id))
    return article_ids


def clean_spans(value: dict[str, Any], window_start: int, window_end: int,
                num_units: int) -> list[dict[str, Any]]:
    raw_passages = value.get("passages", []) if isinstance(value, dict) else []
    if not isinstance(raw_passages, list):
        return []
    result = []
    for raw in raw_passages:
        if not isinstance(raw, dict):
            continue
        try:
            start = int(raw.get("start_unit"))
            end = int(raw.get("end_unit"))
        except (TypeError, ValueError):
            continue
        if start < 1 or end < start or end > num_units:
            continue
        if start < window_start or end > window_end:
            continue
        entity = norm_space(raw.get("main_entity"))
        if not entity:
            continue
        result.append({
            "start_unit": start,
            "end_unit": end,
            "main_entity": entity,
            "entity_type": norm_space(raw.get("entity_type")),
            "semantic_focus": norm_space(raw.get("semantic_focus")),
            "evidence_role": norm_space(raw.get("evidence_role")),
        })
    return result


def construct_document(doc: dict[str, Any], client: LLMClient, args: argparse.Namespace) -> dict[str, Any]:
    units = doc.get("units", [])
    if not units or not any(norm_space(x.get("text")) for x in units):
        return {
            "status": "error",
            "document_id": doc["document_id"],
            "spans": [],
            "calls": [],
            "errors": ["empty_source_units"],
            "diagnostics": {
                "units": len(units), "windows": 0,
                "passages": 0, "errors": 1,
            },
        }
    windows = make_windows(units, args.chunk_units, args.overlap_units)
    spans: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for window_index, window in enumerate(windows):
        start = int(window[0]["unit_id"])
        end = int(window[-1]["unit_id"])
        prompt = PASSAGE_USER.format(
            article_id=doc["document_id"],
            title=doc.get("title", ""),
            units="\n".join(f"[{x['unit_id']}] {x['text']}" for x in window),
        )
        record = {"window_index": window_index, "start_unit": start,
                  "end_unit": end, "prompt": prompt}
        try:
            parsed, raw, request_metrics = client.call_json(prompt, args.max_tokens)
            valid = clean_spans(parsed, start, end, len(units))
            spans.extend(valid)
            record.update({"status": "ok", "raw_response": raw,
                           "parsed_response": parsed,
                           "_resin_request_metrics": request_metrics,
                           "valid_passages": len(valid)})
        except Exception as exc:
            record.update({"status": "error", "error": repr(exc)})
            errors.append(repr(exc))
        calls.append(record)
    unique = {}
    for span in spans:
        key = (span["start_unit"], span["end_unit"])
        if key not in unique:
            unique[key] = span
    return {"status": "ok" if not errors and spans else "error",
            "document_id": doc["document_id"], "spans": list(unique.values()),
            "calls": calls, "errors": errors,
            "diagnostics": {"units": len(units), "windows": len(windows),
                             "passages": len(unique), "errors": len(errors)}}


def build_passages(docs: list[dict[str, Any]], constructions: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    output = []
    by_article: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        article_id = doc["document_id"]
        units = {int(x["unit_id"]): x for x in doc.get("units", [])}
        construction = constructions.get(article_id, {})
        for index, span in enumerate(construction.get("spans", [])):
            start, end = int(span["start_unit"]), int(span["end_unit"])
            selected = [units[x] for x in range(start, end + 1) if x in units]
            if not selected:
                continue
            passage_id = f"{article_id}__p{index:04d}_{stable_hash([article_id, start, end])}"
            by_article[article_id].append(passage_id)
            output.append({
                "document_id": passage_id,
                "parent_document_id": article_id,
                "article_id": article_id,
                "title": doc.get("title", ""),
                "source": doc.get("source", "HolisQA"),
                "author": doc.get("author", ""),
                "published_at": doc.get("published_at", ""),
                "url": doc.get("url", ""),
                "category": doc.get("category", ""),
                "passage_index": index,
                "body": " ".join(x["text"] for x in selected),
                "main_entity": span.get("main_entity", ""),
                "entity_type": span.get("entity_type", ""),
                "semantic_focus": span.get("semantic_focus", ""),
                "evidence_role": span.get("evidence_role", ""),
                "source_sentence_ids": [x["sentence_id"] for x in selected],
                "source_start_unit": start,
                "source_end_unit": end,
                "passage_construction": "llm_semantic_entity_centered_exact_source_span",
            })
    return output, dict(by_article)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--domains", default="biology,business,computer_science,medicine,psychology")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    # Keep semantic passage construction aligned with the CausalRAG2 main table.
    # An explicit --model still overrides this default.
    parser.add_argument("--model", default=os.environ.get("RESIN_MODEL", "gpt-5-nano"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--provider", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk_units", type=int, default=0,
                        help="Temporary LLM input-window size; 0 means one full-article call.")
    parser.add_argument("--overlap_units", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=6000)
    parser.add_argument(
        "--reasoning_effort",
        choices=["default", "minimal", "low", "medium", "high"],
        default=os.environ.get("RESIN_REASONING_EFFORT", "minimal"),
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    docs: list[dict[str, Any]] = []
    qas: list[dict[str, Any]] = []
    counts = {}
    prepared_bundle = (args.source_dir / "corpus" / "corpus.json").exists()
    if prepared_bundle:
        docs, qas = load_prepared_bundle(args.source_dir)
        counts["prepared_bundle"] = {"documents": len(docs), "queries": len(qas)}
    else:
        for domain in domains:
            domain_docs, domain_qas = load_domain(args.source_dir, domain)
            docs.extend(domain_docs)
            qas.extend(domain_qas)
            counts[domain] = {"documents": len(domain_docs), "queries": len(domain_qas)}

        document_ids = [str(doc["document_id"]) for doc in docs]
        duplicate_document_ids = sorted(
            item for item, count in Counter(document_ids).items() if count > 1
        )
        if duplicate_document_ids:
            raise ValueError(
                "HolisQA document IDs are not globally unique after domain scoping: "
                + ", ".join(duplicate_document_ids[:20])
            )
        qa_ids = [str(qa["qa_id"]) for qa in qas]
        duplicate_qa_ids = sorted(
            item for item, count in Counter(qa_ids).items() if count > 1
        )
        if duplicate_qa_ids:
            raise ValueError(
                "HolisQA query IDs are not globally unique after domain scoping: "
                + ", ".join(duplicate_qa_ids[:20])
            )
    # For a query-limited smoke test, select the queries first and then retain
    # every source article needed by their gold evidence. Selecting documents
    # first can silently produce a tiny query set whose evidence is absent.
    if args.max_queries is not None:
        selected_qas = qas[:args.max_queries]
        required_doc_ids = set()
        for qa in selected_qas:
            required_doc_ids.update(qa_evidence_article_ids(qa))
        docs_by_id = {str(doc["document_id"]): doc for doc in docs}
        missing_doc_ids = sorted(required_doc_ids - set(docs_by_id))
        if missing_doc_ids:
            raise ValueError(
                "Selected queries refer to source documents that were not loaded: "
                + ", ".join(missing_doc_ids[:20])
            )
        if args.max_docs is not None and len(required_doc_ids) > args.max_docs:
            print(
                f"[query-selection] ignoring --max_docs={args.max_docs}; "
                f"{len(required_doc_ids)} documents are required for "
                f"{len(selected_qas)} selected queries"
            )
        docs = [doc for doc in docs if str(doc["document_id"]) in required_doc_ids]
        eligible_qas = selected_qas
    else:
        if args.max_docs is not None:
            docs = docs[:args.max_docs]
        selected_doc_ids = {str(doc["document_id"]) for doc in docs}
        eligible_qas = [
            qa for qa in qas
            if (qa_evidence_article_ids(qa)
                and qa_evidence_article_ids(qa).issubset(selected_doc_ids))
        ]
        selected_qas = eligible_qas

    selected_doc_ids = {str(doc["document_id"]) for doc in docs}

    out = args.out_dir
    raw_dir = out / "01_passage_construction" / "raw"
    passage_started_wall = time.time()
    passage_started_perf = time.perf_counter()
    client = LLMClient(args.base_url, args.model, args.api_key, args.timeout,
                       args.retries, args.provider, args.reasoning_effort)
    constructions = {}
    print(f"[passage-construction] documents: {len(docs)} model: {args.model}")
    print(f"[query-selection] eligible queries: {len(eligible_qas)} "
          f"selected: {len(selected_qas)} "
          f"(evidence fully covered by selected documents)")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for doc in docs:
            path = raw_dir / (safe(doc["document_id"]) + ".json")
            if not args.no_resume and path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                saved_diagnostics = saved.get("diagnostics", {})
                if (saved.get("status") == "ok"
                        and int(saved_diagnostics.get("passages", 0)) > 0):
                    constructions[doc["document_id"]] = saved
                    continue
            futures[pool.submit(construct_document, doc, client, args)] = (doc, path)
        for done, future in enumerate(as_completed(futures), 1):
            doc, path = futures[future]
            result = future.result()
            write_json(path, result)
            constructions[doc["document_id"]] = result
            message = (f"[passage-construction] {doc['document_id']}: "
                       f"{result['diagnostics']['passages']} passages from "
                       f"{result['diagnostics']['windows']} LLM call(s) "
                       f"({done}/{len(futures)})")
            if result.get("errors"):
                message += f" errors={result['errors'][:2]}"
            print(message)

    passages, passage_ids_by_article = build_passages(docs, constructions)
    construction_dir = out / "01_passage_construction"
    write_jsonl(construction_dir / "passage_spans.jsonl", [
        {"document_id": doc["document_id"],
         "spans": constructions.get(doc["document_id"], {}).get("spans", []),
         "diagnostics": constructions.get(doc["document_id"], {}).get("diagnostics", {})}
        for doc in docs
    ])
    write_jsonl(construction_dir / "llm_calls.jsonl", [
        {"document_id": doc["document_id"], "calls": constructions.get(doc["document_id"], {}).get("calls", [])}
        for doc in docs
    ])
    write_json(construction_dir / "diagnostics.json", {
        "model": args.model, "domains": domains, "per_domain": counts,
        "documents": len(docs), "passages": len(passages),
        "reasoning_effort": args.reasoning_effort,
        "started_unix": passage_started_wall,
        "finished_unix": time.time(),
        "wall_time_seconds": round(time.perf_counter() - passage_started_perf, 6),
        "policy": "LLM-selected contiguous entity-centered source spans",
    })

    out_qas = []
    gold_contexts = []
    doc_by_title = {norm_space(doc.get("title")): doc["document_id"] for doc in docs
                    if norm_space(doc.get("title"))}
    for qa in selected_qas:
        article_ids = set()
        for evidence in qa.get("evidence_list", []) if isinstance(qa.get("evidence_list"), list) else []:
            evidence_id = (evidence.get("article_id") or evidence.get("document_id") or
                           evidence.get("parent_document_id"))
            if evidence_id:
                article_ids.add(str(evidence_id))
                continue
            title_id = doc_by_title.get(norm_space(evidence.get("title")))
            if title_id:
                article_ids.add(title_id)
        article_ids = sorted(article_ids)
        gold_passages = [pid for aid in article_ids for pid in passage_ids_by_article.get(aid, [])]
        out_qas.append(qa)
        gold_contexts.append({"qa_id": qa["qa_id"], "query": qa["query"],
                              "gold_doc_ids": article_ids, "gold_passage_ids": gold_passages})

    write_json(out / "corpus" / "corpus.json", passages)
    write_jsonl(out / "qa" / "holisqa.jsonl", out_qas)
    write_jsonl(out / "gold_contexts.jsonl", gold_contexts)
    write_json(out / "metadata.json", {
        "source_dir": str(args.source_dir.resolve()), "domains": domains,
        "per_domain_source_counts": counts, "documents_written": len(docs),
        "queries_written": len(out_qas), "eligible_queries": len(eligible_qas),
        "selected_document_ids": sorted(selected_doc_ids),
        "passages_written": len(passages),
        "passage_policy": "EnSI-RAG semantic entity-centered exact source spans",
        "gold_evidence_policy": "retained only for post-hoc evaluation; never sent to LLM prompts",
    })
    print("[done] documents:", len(docs))
    print("[done] queries:", len(out_qas))
    print("[done] semantic passages:", len(passages))
    print("[done] output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
