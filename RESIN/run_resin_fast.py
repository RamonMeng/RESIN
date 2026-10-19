#!/usr/bin/env python3
"""Checkpointed RESIN pipeline with benchmark-specific answer generation.

The generation styles reuse existing query plans and retrieval checkpoints.
Reference answers are used only after generation for F1 diagnostics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "multihoprag"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "resin_runs"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "extract", "normalize", "index", "plan",
                                       "retrieve", "graph", "agent", "evaluate"], default="all")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--run_dir", type=Path, default=None)
    p.add_argument("--corpus_file", type=Path, default=None)
    p.add_argument("--qa_file", type=Path, default=None)
    p.add_argument("--provider", choices=["openai", "vllm"], default="openai")
    # Keep the main generator aligned with the CausalRAG2 main table.
    # An explicit --model still overrides this default.
    p.add_argument("--model", default=os.environ.get("RESIN_MODEL", "gpt-5-nano"))
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--hop_workers", type=int, default=2,
        help=("Concurrent hop-answer requests within one query. Hop retrieval "
              "and evidence are unchanged; use 1 to restore serial execution."),
    )
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--chunk_chars", type=int, default=12000)
    p.add_argument("--chunk_overlap", type=int, default=500)
    p.add_argument("--top_k", type=int, default=25)
    p.add_argument("--top_chains", type=int, default=5)
    p.add_argument("--max_hops", type=int, default=4)
    p.add_argument(
        "--reasoning_mode", choices=["graph", "hopwise_candidates"], default="graph",
        help="Online reasoning backend. hopwise_candidates decomposes the query, retrieves top-k per hop, and aggregates.",
    )
    p.add_argument(
        "--hop_top_k", type=int, default=5,
        help="Number of embedding-matched reasoning units retained for each hop.",
    )
    p.add_argument(
        "--reasoning_embedding_model",
        default=os.environ.get("RESIN_REASONING_EMBEDDING", "all-MiniLM-L6-v2"),
        help="Sentence-Transformers model used by hopwise_candidates retrieval.",
    )
    p.add_argument(
        "--reasoning_embedding_device", default="cpu",
        help="Embedding device for hopwise_candidates, e.g. cpu or cuda.",
    )
    p.add_argument("--reasoning_embedding_batch_size", type=int, default=32)
    p.add_argument("--max_tokens", type=int, default=1800)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument(
        "--reasoning_effort",
        choices=["default", "minimal", "low", "medium", "high"],
        default=os.environ.get("RESIN_REASONING_EFFORT", "minimal"),
        help=("GPT-5 reasoning effort. 'minimal' reduces reasoning-token latency; "
              "'default' omits the API field for a baseline-compatible run."),
    )
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--skip_extract", action="store_true",
                   help="Reuse an existing 01_extraction checkpoint when running --stage all.")
    p.add_argument(
        "--reuse_preprocess_run", type=Path, default=None,
        help="Copy 01_extraction/02_normalization/03_index from an existing run and only rerun online stages.",
    )
    p.add_argument("--no_repair", action="store_true")
    p.add_argument(
        "--generation_style", choices=["concise", "holisqa", "qasc", "msmarco"], default="concise",
        help=("Answer composition style for --stage agent. 'holisqa' covers "
              "each supported part of a holistic question in a short paragraph; "
              "'qasc' composes a multiple-choice answer from existing retrieved facts; "
              "'msmarco' composes a direct answer from existing retrieved passages."),
    )
    p.add_argument(
        "--generation_domain", default=None,
        help="For --stage agent, generate only this QA category (e.g. business).",
    )
    p.add_argument(
        "--generation_limit", type=int, default=None,
        help="For --stage agent, generate the first N selected QA rows.",
    )
    p.add_argument(
        "--generation_output_dir", type=Path, default=None,
        help="Write agent checkpoints and answers here instead of run_dir/07_agent.",
    )
    # The anonymous release intentionally exposes no oracle-retrieval switch.
    # Main-table runs always retrieve from the complete prepared corpus.
    p.set_defaults(oracle_gold_docs=False)
    # CausalRAG2-compatible post-hoc evaluation settings.  The generator
    # remains controlled by --provider/--model; these settings are only for
    # the evaluator used by Answer Relevancy.
    p.add_argument("--judge_provider", choices=["gemini", "openai"],
                   default=os.environ.get("RESIN_JUDGE_PROVIDER", "gemini"))
    p.add_argument("--judge_model",
                   default=os.environ.get("RESIN_JUDGE_MODEL", "gemini-3.5-flash-lite"))
    p.add_argument("--judge_base_url",
                   default=os.environ.get(
                       "RESIN_JUDGE_BASE_URL",
                       "https://generativelanguage.googleapis.com/v1beta"))
    p.add_argument("--judge_api_key",
                   default=os.environ.get("GEMINI_API_KEY", ""))
    p.add_argument("--judge_workers", type=int, default=8)
    p.add_argument("--judge_timeout", type=float, default=120)
    p.add_argument("--judge_retries", type=int, default=3)
    p.add_argument("--judge_max_tokens", type=int, default=256)
    p.add_argument("--judge_strictness", type=int, default=3,
                   help="Ragas Answer Relevancy generations per answer; CausalRAG2 uses the Ragas default of 3.")
    p.add_argument("--embedding_model",
                   default=os.environ.get("RESIN_EVAL_EMBEDDING", "all-MiniLM-L6-v2"))
    p.add_argument("--skip_answer_relevancy", action="store_true",
                   help="Deprecated compatibility flag. RAGAS AR/CR are evaluated by the standalone evaluator.")
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_stage_timing(run: Path, stage_name: str, started_wall: float,
                        started_perf: float) -> None:
    """Persist wall-clock timing after each stage, including separate CLI runs."""
    path = run / "10_cost" / "stage_wall_times.json"
    value = read_json(path) if path.exists() else {"stages": {}}
    stages = value.setdefault("stages", {})
    finished_wall = time.time()
    previous = stages.get(stage_name, {}) if isinstance(stages.get(stage_name), dict) else {}
    history = previous.get("history", []) if isinstance(previous.get("history"), list) else []
    record = {
        "started_unix": started_wall,
        "finished_unix": finished_wall,
        "wall_time_seconds": round(time.perf_counter() - started_perf, 6),
    }
    history.append(record)
    stages[stage_name] = {
        **record,
        "cumulative_wall_time_seconds": round(
            sum(float(item.get("wall_time_seconds", 0.0)) for item in history), 6
        ),
        "history": history,
    }
    write_json(path, value)


def run_timed_stage(run: Path, stage_name: str, function: Any, *stage_args: Any) -> None:
    started_wall = time.time()
    started_perf = time.perf_counter()
    try:
        function(*stage_args)
    finally:
        record_stage_timing(run, stage_name, started_wall, started_perf)


def h(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode()).hexdigest()


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", " ", text)


def tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", norm(value))


# The old prototype treated every subject--predicate--object statement as an
# interchangeable graph edge.  Long-chain reasoning needs a small amount of
# relation typing so that a causal path is not completed with edges such as
# ``diagnosed_by`` or ``is_associated_with``.  These labels are deliberately
# conservative; unknown predicates remain ``other`` instead of being promoted
# to causal relations.
CAUSAL_FORWARD_PATTERNS = (
    "causes", "cause", "leads to", "lead to", "results in", "result in",
    "increases risk", "increase risk", "raises risk", "promotes", "induces",
    "triggers", "produces", "impairs", "inhibits", "reduces", "prevents",
    "contributes to", "drives", "facilitates", "enables", "accumulates",
    "builds up", "buildup", "breaks down", "degrades", "cleaves",
    "determines", "influences", "affects", "interferes with",
)
CAUSAL_REVERSE_PATTERNS = (
    "is caused by", "are caused by", "was caused by", "were caused by",
    "caused by", "results from", "resulting from", "due to", "arises from",
    "stems from", "derived from", "is a consequence of", "is due to",
)
TYPE_PATTERNS = (
    "instance_of", "instanceof", "is_a", "is a", "type_of", "type of",
    "belongs to", "classified as", "subclass of", "specialization of",
)
ASSOCIATION_PATTERNS = (
    "associated with", "is associated with", "correlates with", "linked with",
    "linked to", "related to", "co-occurs with",
)


def relation_info(predicate: Any) -> tuple[str, bool, str]:
    """Return (relation_family, reverse, canonical_relation).

    ``reverse`` means the surface statement is written effect -> cause, e.g.
    ``disease is caused by enzyme deficiency``.  The graph can then preserve a
    single cause -> effect direction without losing the original predicate.
    """
    p = norm(predicate)
    if not p:
        return "other", False, "related_to"
    if any(pattern in p for pattern in CAUSAL_REVERSE_PATTERNS):
        return "causal", True, "causes"
    if any(pattern in p for pattern in TYPE_PATTERNS):
        return "type", False, "is_a"
    if any(pattern in p for pattern in ASSOCIATION_PATTERNS):
        return "association", False, "associated_with"
    if any(pattern in p for pattern in CAUSAL_FORWARD_PATTERNS):
        return "causal", False, p
    return "other", False, p


def atom_signature(value: Any) -> str:
    """Stable key for a reasoning state or an extraction atom."""
    if not isinstance(value, dict):
        return norm(value)
    subject = norm(value.get("canonical_subject") or value.get("subject") or value.get("entity"))
    predicate = norm(value.get("canonical_predicate") or value.get("predicate") or value.get("property"))
    obj = norm(value.get("canonical_object") or value.get("object") or value.get("value"))
    return "|".join(x for x in (subject, predicate, obj) if x)


GENERIC_ENTITY_TOKENS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "from", "with",
    "and", "or", "is", "are", "was", "were", "disease", "disorder",
    "patient", "patients", "people", "person", "type", "types", "form",
    "forms", "gene", "genes", "enzyme", "enzymes", "activity", "level",
    "levels", "deficiency", "mutation", "mutations", "treatment", "therapy",
    "cell", "cells", "risk", "symptoms", "study", "studies",
}


STATE_REWRITE_PATTERNS = (
    (r"\bdeficiency\s+of\s+(?:the\s+)?", "low activity of "),
    (r"\binsufficient\s+activity\s+of\s+(?:the\s+)?", "low activity of "),
    (r"\breduced\s+activity\s+of\s+(?:the\s+)?", "low activity of "),
    (r"\bdecreased\s+activity\s+of\s+(?:the\s+)?", "low activity of "),
    (r"\b(low|reduced|decreased|deficient|impaired)\b", "low"),
    (r"\baccumulation\s+of\b", "accumulation"),
    (r"\bbuildup\s+of\b", "accumulation"),
    (r"\bbuild[- ]up\s+of\b", "accumulation"),
)


class EntityCanonicalizer:
    """Conservative alias and state canonicalizer.

    Raw LLM aliases can be document-level labels (for example ``GD``) and
    should not automatically merge every unit in that document.  An alias is
    accepted only when it maps to one subject, or when multiple subjects share
    a non-generic core such as ``glucocerebrosidase``.
    """

    def __init__(self, units: list[dict[str, Any]]):
        alias_subjects: dict[str, set[str]] = defaultdict(set)
        subject_labels: set[str] = set()
        for unit in units:
            subjects = [unit.get("subject", "")]
            for atom_key in ("source_state", "target_state"):
                atom_value = unit.get(atom_key) or {}
                if isinstance(atom_value, dict):
                    subjects.append(atom_value.get("subject", ""))
            for subject in subjects:
                if subject and norm(subject):
                    subject_labels.add(norm(subject))
            for alias in unit.get("aliases", []) or []:
                alias_key = norm(alias)
                if alias_key and unit.get("subject"):
                    alias_subjects[alias_key].add(norm(unit["subject"]))

        self.alias_to_canonical: dict[str, str] = {}
        for alias, subjects in alias_subjects.items():
            subjects = {x for x in subjects if x}
            if len(subjects) == 1:
                self.alias_to_canonical[alias] = next(iter(subjects))
                continue
            # A short acronym is safe only when its candidate subjects share a
            # meaningful core.  This keeps GCase useful while rejecting the
            # noisy document-wide GD alias.
            token_sets = [set(tokens(x)) - GENERIC_ENTITY_TOKENS for x in subjects]
            common = set.intersection(*token_sets) if token_sets else set()
            if common:
                representative = min(subjects, key=lambda x: (len(tokens(x)), x))
                self.alias_to_canonical[alias] = representative

        self.subject_labels = subject_labels
        self.single_aliases = {k: v for k, v in self.alias_to_canonical.items() if " " not in k}
        self.phrase_aliases = sorted(
            ((k, v) for k, v in self.alias_to_canonical.items() if " " in k),
            key=lambda item: (-len(item[0]), item[0]),
        )
        self._cache: dict[str, str] = {}

    def canonicalize(self, value: Any) -> str:
        text = norm(value)
        if not text:
            return ""
        if text in self._cache:
            return self._cache[text]
        if text in self.alias_to_canonical:
            result = self.alias_to_canonical[text]
        else:
            words = text.split()
            words = [self.single_aliases.get(word, word) for word in words]
            result = " ".join(words)
            for alias, canonical in self.phrase_aliases:
                if alias in result:
                    result = re.sub(r"(?<![a-z0-9])" + re.escape(alias) +
                                    r"(?![a-z0-9])", canonical, result)
        result = re.sub(r"\s+", " ", result).strip()
        self._cache[text] = result
        return result


def state_signature(value: Any, resolver: EntityCanonicalizer | None = None) -> str:
    """Canonical state key used to join causal edge endpoints.

    New extraction outputs use [entity][property][value].  Older causal units
    often contain a free-form subject phrase with predicates such as ``causes``
    and a target phrase with predicate ``state``.  We normalize those legacy
    forms conservatively so the new graph can still join them.
    """
    if not isinstance(value, dict):
        text = norm(value)
        predicate = ""
        obj = ""
    else:
        subject = value.get("canonical_subject") or value.get("subject") or value.get("entity")
        predicate = value.get("canonical_predicate") or value.get("predicate") or value.get("property")
        obj = value.get("canonical_object") or value.get("object") or value.get("value")
        text = norm(" ".join(str(x) for x in (subject, obj) if x))
        predicate = norm(predicate)
    if resolver:
        text = resolver.canonicalize(text)
    for pattern, replacement in STATE_REWRITE_PATTERNS:
        text = re.sub(pattern, replacement, text)
    family, _, _ = relation_info(predicate)
    if predicate in {"state", "causes", "cause", "caused_by", "leads to"} or family == "causal":
        # These are edge labels, not state properties, for legacy units.
        predicate = ""
    parts = [x for x in tokens(text) if x not in {"the", "a", "an", "of", "in", "on", "to", "from", "with"}]
    if predicate and predicate not in {"state"}:
        parts.extend(tokens(predicate))
    return "|".join(sorted(set(parts)))


STATE_JOIN_IGNORED = {
    "accumulation", "accumulates", "accumulate", "buildup", "builds", "build",
    "storage", "stored", "state", "present", "true", "false", "low", "reduced",
    "decreased", "deficient", "activity", "activities", "enzyme", "enzymes",
    "level", "levels",
}


def state_join_signature(value: Any, resolver: EntityCanonicalizer | None = None) -> str:
    """A conservative join key for equivalent causal endpoints.

    The full state signature remains in the node label.  This secondary key
    removes only generic state words so ``glucocerebroside`` can join
    ``glucocerebroside accumulation`` and ``GCase activity reduced`` can join
    ``deficiency of glucocerebrosidase``.  It does not remove the domain entity
    itself, so unrelated disease mechanisms are not collapsed.
    """
    full = state_signature(value, resolver)
    parts = [x for x in full.split("|") if x and x not in STATE_JOIN_IGNORED]
    return "|".join(sorted(set(parts))) or full


def safe(value: str) -> str:
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:150].strip("._") or "item")


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    raw = read_json(path)
    if isinstance(raw, list):
        return [dict(x) for x in raw]
    if isinstance(raw, dict):
        for key in ("data", "records", "queries", "corpus"):
            if isinstance(raw.get(key), list):
                return [dict(x) for x in raw[key]]
    raise ValueError("No record list found in " + str(path))


def resolve_files(a: argparse.Namespace) -> tuple[Path, Path]:
    corpus = a.corpus_file or a.data_dir / "corpus" / "corpus.json"
    qa_candidates = [a.qa_file,
                     a.data_dir / "qa" / "holisqa.jsonl",
                     a.data_dir / "qa" / "qasc.jsonl",
                     a.data_dir / "qa" / "msmarco.jsonl",
                     a.data_dir / "qa" / "train.jsonl",
                     a.data_dir / "qa" / "train.json",
                     a.data_dir / "corpus" / "MultiHopRAG.json",
                     a.data_dir / "corpus" / "multihopRAG.json",
                     a.data_dir / "MultiHopRAG.json",
                     a.data_dir / "multihopRAG.json"]
    qa = next((x for x in qa_candidates if x is not None and x.exists()), None)
    if qa is None:
        qa_dir = a.data_dir / "qa"
        qa = next(iter(sorted(qa_dir.glob("*.jsonl"))), None) if qa_dir.exists() else None
    if not corpus.exists() or qa is None:
        missing = []
        if not corpus.exists():
            missing.append("corpus=" + str(corpus))
        if qa is None:
            missing.append("qa JSON/JSONL under " + str(a.data_dir / "qa"))
        raise FileNotFoundError(
            "Dataset bundle is incomplete (" + ", ".join(missing) + ")"
        )
    return corpus, qa


class LLMClient:
    def __init__(self, base_url: str, model: str, key: str, timeout: float,
                 retries: int, temperature: float, provider: str = "openai",
                 reasoning_effort: str = "default"):
        self.base = base_url.rstrip("/")
        self.model = model
        self.key = key
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        if self.provider == "openai" and not self.key:
            raise ValueError("OPENAI_API_KEY is not set. Export it or pass --api_key.")

    def call_json(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
        request_started = time.perf_counter()
        request_id = uuid.uuid4().hex
        payload = {
            "model": self.model, "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        # GPT-5 reasoning models currently accept only their default
        # temperature. Omitting the field is required; sending 0.0 or 0.1
        # causes an HTTP 400 from the OpenAI API.
        if not (self.provider == "openai" and self.model.lower().startswith("gpt-5")):
            payload["temperature"] = self.temperature
        if (self.provider == "openai" and self.model.lower().startswith("gpt-5")
                and self.reasoning_effort != "default"):
            payload["reasoning_effort"] = self.reasoning_effort
        if self.provider == "vllm":
            payload["max_tokens"] = max_tokens
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["max_completion_tokens"] = max_tokens
        variants = [payload]
        fallback = dict(payload)
        fallback.pop("response_format", None)
        variants.append(fallback)
        fallback2 = dict(fallback)
        fallback2.pop("chat_template_kwargs", None)
        variants.append(fallback2)
        last = ""
        for attempt in range(self.retries):
            for variant_index, body in enumerate(variants):
                try:
                    response = requests.post(
                        self.base + "/chat/completions",
                        headers={"Authorization": "Bearer " + self.key,
                                 "Content-Type": "application/json"},
                        json=body, timeout=self.timeout)
                    if response.status_code >= 400:
                        last = "HTTP " + str(response.status_code) + ": " + response.text[:800]
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
                    total_tokens = usage.get("total_tokens")
                    reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                    cached_tokens = prompt_details.get("cached_tokens", 0)
                    data["_resin_request_metrics"] = {
                        "request_id": request_id,
                        "provider": self.provider,
                        "requested_model": self.model,
                        "response_model": data.get("model", self.model),
                        "latency_seconds": round(time.perf_counter() - request_started, 6),
                        "attempts_used": attempt + 1,
                        "successful_variant": variant_index,
                        "prompt_tokens": prompt_tokens,
                        "cached_prompt_tokens": cached_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "visible_completion_tokens": (
                            max(0, int(completion_tokens) - int(reasoning_tokens or 0))
                            if completion_tokens is not None else None
                        ),
                        "total_tokens": total_tokens,
                    }
                    return data, str(content), parse_json(str(content))
                except Exception as exc:
                    last = repr(exc)
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("LLM request failed: " + last)


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        starts = [x for x in (text.find("{"), text.find("[")) if x >= 0]
        if not starts:
            raise ValueError("No JSON found: " + text[:500])
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        result = json.loads(text[start:end + 1])
    if isinstance(result, list):
        return {"items": result}
    return result if isinstance(result, dict) else {"value": result}


# This is the Answer Relevancy prompt documented by CausalRAG2 (Ragas
# ResponseRelevancy).  Ragas calls this prompt ``strictness`` times, then
# embeds the generated questions and averages their cosine similarity with
# the original user question.  Keep the prompt in one constant so every
# evaluation artifact contains the exact text used for judging.
CAUSALRAG2_AR_SCHEMA = (
    '{"properties": {"question": {"title": "Question", "type": "string"}, '
    '"noncommittal": {"title": "Noncommittal", "type": "integer"}}, '
    '"required": ["question", "noncommittal"], '
    '"title": "ResponseRelevanceOutput", "type": "object"}'
)

CAUSALRAG2_AR_INSTRUCTION = (
    "Generate a question for the given answer and identify if the answer is noncommittal.\n"
    "Give noncommittal as 1 if the answer is noncommittal (evasive, vague, or ambiguous) "
    "and 0 if the answer is substantive.\n"
    'Examples of noncommittal answers: "I don\'t know", "I\'m not sure", "It depends".'
)

CAUSALRAG2_AR_EXAMPLES = r'''### Examples

Input: {"response": "Albert Einstein was born in Germany."}
Output: {"question": "Where was Albert Einstein born?", "noncommittal": 0}

Input: {"response": "The capital of France is Paris, a city known for its architecture and culture."}
Output: {"question": "What is the capital of France?", "noncommittal": 0}

Input: {"response": "I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022."}
Output: {"question": "What was the groundbreaking feature of the smartphone invented in 2023?", "noncommittal": 1}'''

CAUSALRAG2_AR_TEMPLATE = """{instruction}

Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:

{output_schema}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

{examples}

-----------------------------

Now perform the same with the following input

input: {input_json}
Output:
"""


def causalrag2_ar_prompt(answer: Any) -> str:
    return CAUSALRAG2_AR_TEMPLATE.format(
        instruction=CAUSALRAG2_AR_INSTRUCTION,
        output_schema=CAUSALRAG2_AR_SCHEMA,
        examples=CAUSALRAG2_AR_EXAMPLES,
        input_json=json.dumps({"response": str(answer or "")}, ensure_ascii=False),
    )


class JudgeClient:
    """Small evaluator client for the CausalRAG2/Ragas AR protocol.

    The published CausalRAG2 setup uses Gemini-2.5-Flash-Lite as the
    evaluator.  Gemini is called through its native generateContent REST API;
    an explicit OpenAI-compatible evaluator can be selected when Gemini is
    unavailable.  The evaluator remains independent of the RESIN generator,
    which may be Qwen served by vLLM.
    """

    def __init__(self, provider: str, model: str, base_url: str, key: str,
                 timeout: float, retries: int, temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        if not self.key:
            env_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
            raise ValueError(f"{env_name} is not set; required for Answer Relevancy evaluation.")

    def call_json(self, prompt: str, max_tokens: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
        last = ""
        for attempt in range(self.retries):
            try:
                if self.provider == "gemini":
                    url = f"{self.base_url}/models/{self.model}:generateContent"
                    payload = {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": self.temperature,
                            "maxOutputTokens": max_tokens,
                        },
                    }
                    response = requests.post(
                        url,
                        params={"key": self.key},
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout,
                    )
                    if response.status_code >= 400:
                        last = "HTTP " + str(response.status_code) + ": " + response.text[:800]
                        raise RuntimeError(last)
                    data = response.json()
                    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                    content = "".join(str(x.get("text", "")) for x in parts if isinstance(x, dict))
                    if not content:
                        raise ValueError("Gemini returned no text content: " + json.dumps(data)[:800])
                    return data, content, parse_json(content)

                # Explicit debugging fallback.  It is not the CausalRAG2
                # protocol because the evaluator model/provider changes.
                client = LLMClient(self.base_url, self.model, self.key,
                                   self.timeout, self.retries, self.temperature, "openai")
                return client.call_json([{"role": "user", "content": prompt}], max_tokens)
            except Exception as exc:
                last = repr(exc)
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Answer Relevancy judge failed: " + last)


def get_doc_id(doc: dict[str, Any], i: int) -> str:
    return str(doc.get("document_id") or doc.get("id") or
               "doc_" + h({"title": doc.get("title", ""), "url": doc.get("url", ""), "i": i})[:16])


def chunks(text: str, limit: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    step = max(1, limit - overlap)
    return [text[i:i + limit] for i in range(0, len(text), step) if text[i:i + limit].strip()]


def extraction_inputs(doc: dict[str, Any], chunk_chars: int, chunk_overlap: int) -> list[tuple[int, str]]:
    """Return extraction inputs while preserving EnSI-RAG semantic passages.

    ``chunk_chars`` is only a compatibility fallback for corpora that were
    not produced by a semantic passage constructor.  EnSI-RAG-style passages
    are already the intended retrieval units, so re-splitting them here would
    violate passage provenance and semantic containment.
    """
    text = str(doc.get("body") or doc.get("text") or "")
    if doc.get("passage_construction") == "llm_semantic_entity_centered_exact_source_span":
        return [(0, text)] if text.strip() else []
    return [
        (j, value)
        for j, value in enumerate(chunks(text, chunk_chars, chunk_overlap))
    ]


EXTRACT_SYSTEM = """You are the offline extractor for RESIN.
Return JSON only: {"units":[...]}.

RESIN represents long-chain reasoning as typed states and directed
transitions. Extract the smallest evidence-backed units that can be composed.
Each unit must have:
kind: entity_fact, type_property, entity_type, causal_edge, relation, or rule;
subject; subject_kind: entity, type, variable, event, state, or unknown;
predicate; object; conditions; conclusion; source_state; target_state;
aliases; scope; modality; polarity; confidence; evidence_quote.

Use these conventions:
- entity_fact: an ordinary fact about a specific entity.
- type_property: a generic property of a type, such as Humans are mortal.
- entity_type: an explicit membership assertion, such as Socrates is a human.
- causal_edge: a directed transition, such as enzyme deficiency causes substrate
  accumulation. Use source_state and target_state when the sentence expresses
  property-value states. Preserve the direction and any modality such as may,
  can, or is thought to.
- rule: only a conditional statement with non-empty conditions and a conclusion,
  such as If people drink more water, health improves. Do not use rule for a
  simple one-edge causal fact.

For a sentence containing multiple causal steps, emit multiple causal_edge
units, one per supported transition, while keeping the same evidence_quote.
Do not infer a transition merely because two entities occur in the same text.
Do not merge a statue with the person it represents. Keep entity type, context,
polarity, and uncertainty explicit. Every unit needs an exact short
evidence_quote.
"""


def extract_messages(doc: dict[str, Any], chunk_id: int, text: str) -> list[dict[str, str]]:
    meta = {k: doc.get(k, "") for k in ("title", "source", "author", "published_at", "url")}
    meta["chunk_id"] = chunk_id
    return [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": "ARTICLE_METADATA:\n" +
             json.dumps(meta, ensure_ascii=False, indent=2) +
             "\nARTICLE_CHUNK:\n<<<\n" + text + "\n>>>"}]


def extraction_task(task: dict[str, Any], client: LLMClient, raw_dir: Path,
                    resume: bool, max_tokens: int) -> Path:
    path = raw_dir / (safe(task["doc_id"]) + "_chunk_" + str(task["chunk_id"]).zfill(4) + ".json")
    if resume and path.exists() and read_json(path).get("status") == "ok":
        return path
    messages = extract_messages(task["doc"], task["chunk_id"], task["text"])
    output = {"status": "error", "doc_id": task["doc_id"], "chunk_id": task["chunk_id"],
              "chunk_text": task["text"], "request_messages": messages}
    try:
        response, content, parsed = client.call_json(messages, max_tokens)
        units = parsed.get("units", parsed.get("items", []))
        output.update({"status": "ok", "response": response, "response_content": content,
                       "parsed": parsed, "units": units if isinstance(units, list) else []})
    except Exception as exc:
        output["error"] = repr(exc)
    write_json(path, output)
    return path


def extraction_stage(a: argparse.Namespace, run: Path, corpus_file: Path) -> None:
    stage = run / "01_extraction"
    raw_dir = stage / "raw"
    corpus = load_records(corpus_file)
    if a.max_docs is not None:
        corpus = corpus[:a.max_docs]
    tasks, lookup = [], {}
    preserved_semantic_passages = 0
    for i, doc in enumerate(corpus):
        did = get_doc_id(doc, i)
        lookup[did] = doc
        inputs = extraction_inputs(doc, a.chunk_chars, a.chunk_overlap)
        if doc.get("passage_construction") == "llm_semantic_entity_centered_exact_source_span":
            preserved_semantic_passages += 1
        for j, text in inputs:
            tasks.append({"doc_id": did, "doc": doc, "chunk_id": j, "text": text})
    write_json(stage / "task_manifest.json", {"documents": len(corpus), "chunks": len(tasks),
                                              "corpus_file": str(corpus_file),
                                              "preserved_semantic_passages": preserved_semantic_passages,
                                              "chunk_chars_fallback": a.chunk_chars})
    client = LLMClient(a.base_url, a.model, a.api_key, a.timeout, a.retries,
                       a.temperature, a.provider, a.reasoning_effort)
    print("[extract] chunks:", len(tasks), "model:", client.model, "provider:", a.provider)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = [pool.submit(extraction_task, task, client, raw_dir,
                                not a.no_resume, a.max_tokens) for task in tasks]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == len(futures):
                print("[extract] completed", done, "/", len(futures))
    extracted, errors = [], []
    for path in sorted(raw_dir.glob("*.json")):
        item = read_json(path)
        if item.get("status") != "ok":
            errors.append({"path": str(path), "error": item.get("error", "")})
            continue
        doc = lookup.get(item.get("doc_id"), {})
        for j, unit in enumerate(item.get("units", [])):
            if not isinstance(unit, dict):
                continue
            record = dict(unit)
            record.update({
                "doc_id": item.get("doc_id"), "chunk_id": item.get("chunk_id"),
                "chunk_text": item.get("chunk_text", ""),
                "document": {k: doc.get(k, "") for k in (
                    "document_id", "parent_document_id", "title", "source", "author",
                    "published_at", "url", "category", "passage_index",
                    "source_sentence_ids", "source_start_unit", "source_end_unit",
                    "main_entity", "entity_type", "semantic_focus", "evidence_role")},
                "extraction_unit_index": j,
            })
            extracted.append(record)
    write_jsonl(stage / "units.jsonl", extracted)
    write_jsonl(stage / "errors.jsonl", errors)
    write_json(stage / "summary.json", {"documents": len(corpus), "chunks": len(tasks),
                                        "units": len(extracted), "errors": len(errors),
                                        "model": client.model})


def atom(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {"subject": "", "predicate": "", "object": str(value or ""),
                "canonical_subject": "", "canonical_predicate": "",
                "canonical_object": ""}
    subject = str(value.get("subject", value.get("entity", ""))).strip()
    predicate = norm(value.get("predicate", value.get("property", "")))
    obj = str(value.get("object", value.get("value", ""))).strip()
    return {"subject": subject, "predicate": predicate, "object": obj,
            "canonical_subject": norm(value.get("canonical_subject", "")),
            "canonical_predicate": norm(value.get("canonical_predicate", "")),
            "canonical_object": norm(value.get("canonical_object", ""))}


def confidence_value(value: Any) -> float:
    """Normalize numeric and qualitative confidence values from LLM JSON."""
    if isinstance(value, str):
        label = norm(value)
        qualitative = {"very low": 0.2, "low": 0.35, "medium": 0.6,
                       "high": 0.85, "very high": 0.95}
        if label in qualitative:
            return qualitative[label]
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def normalize_unit(raw: dict[str, Any], i: int) -> dict[str, Any] | None:
    kind = norm(raw.get("kind") or "entity_fact").replace(" ", "_")
    if kind not in {"entity_fact", "type_property", "entity_type", "causal_edge",
                    "relation", "rule"}:
        kind = "relation"
    subject = str(raw.get("subject") or "").strip()
    predicate = norm(raw.get("predicate") or raw.get("property") or "related_to")
    if not subject:
        return None
    document = raw.get("document", {})
    raw_conditions = raw.get("conditions") or []
    if isinstance(raw_conditions, dict):
        raw_conditions = [raw_conditions]
    conditions = [atom(x) for x in raw_conditions if isinstance(x, dict)]
    conclusion = atom(raw.get("conclusion", {}))
    if not conclusion["subject"]:
        conclusion = {"subject": subject, "predicate": predicate, "object": str(raw.get("object", ""))}
    source_state = atom(raw.get("source_state") or raw.get("cause_state") or {})
    target_state = atom(raw.get("target_state") or raw.get("effect_state") or {})
    # Backward compatibility: old extraction outputs only subject/predicate/
    # object.  Causal indexing can still use that representation.
    if kind == "causal_edge" and not source_state["subject"]:
        source_state = {"subject": subject, "predicate": predicate, "object": ""}
    if kind == "causal_edge" and not target_state["subject"]:
        target_state = {"subject": str(raw.get("object") or "").strip(),
                        "predicate": "state", "object": ""}
    relation_family, reverse, canonical_relation = relation_info(predicate)
    result = {
        "kind": kind, "subject": subject,
        "subject_kind": norm(raw.get("subject_kind") or ("type" if kind == "type_property" else "entity")),
        "predicate": predicate, "object": raw.get("object", ""),
        "conditions": conditions, "conclusion": conclusion,
        "source_state": source_state, "target_state": target_state,
        "relation_family": relation_family, "canonical_relation": canonical_relation,
        "reverse_surface": reverse,
        "aliases": [str(x).strip() for x in (raw.get("aliases") or []) if str(x).strip()],
        "scope": norm(raw.get("scope") or "unknown"),
        "modality": norm(raw.get("modality") or "asserted"),
        "polarity": norm(raw.get("polarity") or "positive"),
        "confidence": confidence_value(raw.get("confidence", 0.5)),
        "evidence_quote": str(raw.get("evidence_quote") or raw.get("chunk_text", "")[:500]).strip(),
        "doc_id": raw.get("doc_id", ""), "chunk_id": raw.get("chunk_id", 0),
        "document": document, "extraction_unit_index": raw.get("extraction_unit_index", i),
    }
    result["unit_id"] = "ru_" + h({k: result[k] for k in
                                   ("doc_id", "chunk_id", "kind", "subject", "predicate",
                                    "object", "conditions", "conclusion", "source_state",
                                    "target_state")})[:20]
    result["search_text"] = " ".join(str(x) for x in
                                     (result["kind"], result["subject"], *result["aliases"],
                                     result["predicate"], result["object"], result["evidence_quote"],
                                     result["relation_family"], result["canonical_relation"],
                                     result["modality"], result["polarity"],
                                     document.get("title", ""), document.get("source", ""),
                                     json.dumps(conditions), json.dumps(conclusion)) if x)
    return result


def normalize_stage(run: Path) -> None:
    stage = run / "02_normalization"
    raw = read_jsonl(run / "01_extraction" / "units.jsonl")
    units = [x for i, r in enumerate(raw) if (x := normalize_unit(r, i)) is not None]
    write_jsonl(stage / "units.jsonl", units)
    write_json(stage / "summary.json", {"input": len(raw), "output": len(units),
                                        "kinds": dict(Counter(x["kind"] for x in units))})


def add_idx(index: dict[str, list[str]], key: Any, uid: str) -> None:
    key = norm(key)
    if key and uid not in index.setdefault(key, []):
        index[key].append(uid)


def index_stage(run: Path) -> None:
    stage = run / "03_index"
    units = read_jsonl(run / "02_normalization" / "units.jsonl")
    resolver = EntityCanonicalizer(units)
    indexes = {name: {} for name in ("entity_index", "type_index", "property_index",
                                     "rule_index", "token_index", "source_index",
                                     "subject_index", "object_index", "alias_index",
                                     "state_index", "causal_out_index", "causal_in_index",
                                     "relation_family_index", "entity_type_index")}
    rules_with_conditions = 0
    causal_units = 0
    typed_units = 0
    for unit in units:
        uid = unit["unit_id"]
        canonical_subject = resolver.canonicalize(unit.get("subject", ""))
        canonical_object = resolver.canonicalize(unit.get("object", ""))
        if unit.get("subject_kind") in {"entity", "variable"} or unit["kind"] in {"entity_fact", "relation"}:
            add_idx(indexes["entity_index"], unit["subject"], uid)
            add_idx(indexes["entity_index"], canonical_subject, uid)
            for alias in unit.get("aliases", []) or []:
                alias_key = norm(alias)
                if resolver.alias_to_canonical.get(alias_key) == canonical_subject:
                    add_idx(indexes["entity_index"], alias, uid)
                    add_idx(indexes["alias_index"], alias, uid)
        if unit["kind"] == "type_property" or unit.get("subject_kind") == "type":
            add_idx(indexes["type_index"], unit["subject"], uid)
            add_idx(indexes["type_index"], canonical_subject, uid)
        if unit["kind"] == "entity_type" or norm(unit.get("predicate")) in {
                "instance_of", "instanceof", "is_a", "type_of", "type of"}:
            typed_units += 1
            add_idx(indexes["entity_type_index"], unit.get("subject"), uid)
            add_idx(indexes["entity_type_index"], unit.get("object"), uid)
        add_idx(indexes["property_index"], unit["predicate"], uid)
        family = unit.get("relation_family") or relation_info(unit.get("predicate", ""))[0]
        add_idx(indexes["relation_family_index"], family, uid)
        add_idx(indexes["subject_index"], unit.get("subject"), uid)
        add_idx(indexes["subject_index"], canonical_subject, uid)
        add_idx(indexes["object_index"], unit.get("object"), uid)
        add_idx(indexes["object_index"], canonical_object, uid)

        source_state = unit.get("source_state") or {}
        target_state = unit.get("target_state") or {}
        if family == "causal" and not source_state.get("subject"):
            source_state = {"subject": unit.get("subject", ""),
                            "predicate": unit.get("predicate", ""), "object": ""}
        if family == "causal" and not target_state.get("subject"):
            target_state = {"subject": unit.get("object", ""),
                            "predicate": "state", "object": ""}
        source_key = state_signature(source_state, resolver)
        target_key = state_signature(target_state, resolver)
        source_join_key = state_join_signature(source_state, resolver)
        target_join_key = state_join_signature(target_state, resolver)
        for key in {source_key, source_join_key, target_key, target_join_key}:
            if key:
                add_idx(indexes["state_index"], key, uid)
        if family == "causal" or unit.get("kind") == "causal_edge":
            causal_units += 1
            if source_key and target_key:
                for key in {source_key, source_join_key}:
                    if key:
                        add_idx(indexes["causal_out_index"], key, uid)
                for key in {target_key, target_join_key}:
                    if key:
                        add_idx(indexes["causal_in_index"], key, uid)
        if unit["kind"] == "rule":
            if unit.get("conditions"):
                rules_with_conditions += 1
                for condition in unit.get("conditions", []):
                    add_idx(indexes["rule_index"], "condition:" + atom_signature(condition), uid)
            else:
                # Keep malformed/legacy rules auditable without pretending
                # that they are conditional rules.
                add_idx(indexes["rule_index"], "unconditional:" + uid, uid)
        add_idx(indexes["source_index"], unit.get("document", {}).get("source", ""), uid)
        for word in set(tokens(unit.get("search_text", ""))):
            add_idx(indexes["token_index"], word, uid)
    for name, value in indexes.items():
        write_json(stage / (name + ".json"), value)
    write_jsonl(stage / "unit_store.jsonl", units)
    write_json(stage / "summary.json", {"units": len(units),
                                        "keys": {k: len(v) for k, v in indexes.items()},
                                        "causal_units": causal_units,
                                        "typed_units": typed_units,
                                        "rules_with_conditions": rules_with_conditions})


PLAN_SYSTEM = """You are the RESIN query planner. Do not answer the question.
Decompose the question into an ordered list of minimal one-hop subquestions.
Each subquestion must be answerable from a small set of evidence-backed
reasoning units and should expose the bridge needed by the next hop. Do not
write a multi-hop subquestion. Use at most the requested MAX_HOPS.
Return JSON only:
{"reasoning_mode":"entity_join | type_inheritance | conditional_rule | causal_chain | temporal | comparison | mixed",
 "answer_type":"entity | boolean | date | explanation | unknown",
 "target_description":"","entity_mentions":[],"type_hints":[],"properties":[],
 "constraints":[],"source_constraints":[],"time_constraints":[],"bridge_terms":[],
 "subquestions":[{"hop":1,"question":"","target":"","depends_on":[],"bridge_terms":[],"reasoning_type":""}]}
"""


def plan_one(row: dict[str, Any], i: int, client: LLMClient, stage: Path,
             resume: bool, max_tokens: int, max_hops: int) -> Path:
    query = str(row.get("query", ""))
    query_id = "q_" + str(i).zfill(6) + "_" + h(query)[:8]
    path = stage / (query_id + ".json")
    if resume and path.exists() and read_json(path).get("status") == "ok":
        return path
    messages = [{"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": f"MAX_HOPS: {max_hops}\nQUESTION:\n{query}"}]
    output = {"status": "error", "query_id": query_id, "row_index": i,
              "query": query,
              "request_messages": messages}
    try:
        response, content, parsed = client.call_json(messages, max_tokens)
        defaults = {"reasoning_mode": "mixed", "answer_type": "unknown",
                    "target_description": "", "entity_mentions": [], "type_hints": [],
                    "properties": [], "constraints": [], "source_constraints": [],
                    "time_constraints": [], "bridge_terms": [], "subquestions": []}
        defaults.update(parsed)
        output.update({"status": "ok", "response": response,
                       "response_content": content, "plan": defaults})
    except Exception as exc:
        output["error"] = repr(exc)
    write_json(path, output)
    return path


def plan_stage(a: argparse.Namespace, run: Path, qa_file: Path) -> None:
    stage = run / "04_query_plans"
    rows = load_records(qa_file)
    if a.max_queries is not None:
        rows = rows[:a.max_queries]
    write_jsonl(stage / "selected_queries.jsonl", [
        {"row_index": i, "query": row.get("query", "")} for i, row in enumerate(rows)])
    client = LLMClient(a.base_url, a.model, a.api_key, a.timeout, a.retries,
                       a.temperature, a.provider, a.reasoning_effort)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = [pool.submit(plan_one, row, i, client, stage, not a.no_resume,
                               a.max_tokens, a.max_hops)
                   for i, row in enumerate(rows)]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == len(futures):
                print("[plan] completed", done, "/", len(futures))
    plans, errors = [], []
    for path in sorted(stage.glob("q_*.json")):
        item = read_json(path)
        if item.get("status") == "ok":
            plans.append({"query_id": item["query_id"], "row_index": item["row_index"],
                          "query": item["query"], "plan": item["plan"]})
        else:
            errors.append(item)
    write_jsonl(stage / "plans.jsonl", plans)
    write_jsonl(stage / "errors.jsonl", errors)
    write_json(stage / "summary.json", {"queries": len(rows), "successful": len(plans),
                                        "errors": len(errors), "model": client.model})


def plan_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [x for item in value for x in plan_strings(item)]
    if isinstance(value, dict):
        return [x for item in value.values() for x in plan_strings(item)]
    return []


def normalized_subquestions(plan: dict[str, Any], query: str,
                            max_hops: int) -> list[dict[str, Any]]:
    """Normalize planner output into ordered one-hop questions."""
    raw = plan.get("subquestions", [])
    if not isinstance(raw, list):
        raw = []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw, 1):
        if isinstance(item, str):
            question = item.strip()
            value = {}
        elif isinstance(item, dict):
            value = item
            question = str(item.get("question") or item.get("subquestion") or "").strip()
        else:
            continue
        if not question:
            continue
        result.append({
            "hop": len(result) + 1,
            "question": question,
            "target": str(value.get("target") or value.get("answer_target") or ""),
            "depends_on": value.get("depends_on", []) if isinstance(value.get("depends_on", []), list) else [],
            "bridge_terms": value.get("bridge_terms", []) if isinstance(value.get("bridge_terms", []), list) else [],
            "reasoning_type": str(value.get("reasoning_type") or value.get("mode") or ""),
        })
        if len(result) >= max(1, max_hops):
            break
    if not result:
        result = [{"hop": 1, "question": query, "target": "", "depends_on": [],
                   "bridge_terms": [], "reasoning_type": plan.get("reasoning_mode", "mixed")}]
    return result


def reasoning_unit_embedding_text(unit: dict[str, Any]) -> str:
    value = unit.get("search_text")
    if value:
        return str(value)
    fields = [
        unit.get("kind", ""), unit.get("subject", ""), unit.get("subject_kind", ""),
        unit.get("predicate", ""), unit.get("object", ""), unit.get("evidence_quote", ""),
        json.dumps(unit.get("conditions", []), ensure_ascii=False),
        json.dumps(unit.get("conclusion", {}), ensure_ascii=False),
    ]
    return " ".join(str(x) for x in fields if x)


def load_reasoning_embedder(a: argparse.Namespace) -> tuple[Any, str]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "hopwise_candidates requires a working sentence-transformers installation. "
            "Check transformers/tokenizers/huggingface-hub compatibility."
        ) from exc
    model_name = a.reasoning_embedding_model
    if model_name == "all-MiniLM-L6-v2":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device=a.reasoning_embedding_device)
    return model, model_name


def load_or_build_reasoning_embeddings(a: argparse.Namespace, run: Path,
                                      units: dict[str, dict[str, Any]],
                                      model: Any, model_name: str) -> tuple[Any, list[str]]:
    import numpy as np

    stage = run / "05_retrieval"
    cache_path = stage / "reasoning_unit_embeddings.npz"
    metadata_path = stage / "reasoning_unit_embeddings.json"
    unit_ids = sorted(units)
    metadata = {"model": model_name, "unit_ids": unit_ids,
                "dimension": None, "normalized": True}
    if cache_path.exists() and metadata_path.exists() and not a.no_resume:
        cached_meta = read_json(metadata_path)
        if cached_meta.get("model") == model_name and cached_meta.get("unit_ids") == unit_ids:
            return np.load(cache_path)["embeddings"], unit_ids
    texts = [reasoning_unit_embedding_text(units[uid]) for uid in unit_ids]
    embeddings = model.encode(
        texts,
        batch_size=max(1, a.reasoning_embedding_batch_size),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    metadata["dimension"] = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    stage.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    write_json(metadata_path, metadata)
    return embeddings, unit_ids


def hopwise_retrieve_one(plan: dict[str, Any], units: dict[str, dict[str, Any]],
                         embeddings: Any, unit_ids: list[str], model: Any,
                         a: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    subquestions = normalized_subquestions(plan["plan"], plan["query"], a.max_hops)
    allowed = np.ones(len(unit_ids), dtype=bool)
    if a.oracle_gold_docs and plan.get("gold_doc_ids"):
        gold = {norm(x) for x in plan.get("gold_doc_ids", [])}
        allowed = np.array([
            norm((units[uid].get("document", {}).get("parent_document_id") or
                  units[uid].get("document", {}).get("document_id"))) in gold
            for uid in unit_ids
        ], dtype=bool)
    hops = []
    flattened = []
    for subq in subquestions:
        query_text = " ".join([
            subq["question"], subq.get("target", ""),
            " ".join(str(x) for x in subq.get("bridge_terms", [])),
        ]).strip()
        vector = model.encode([query_text], convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.asarray(embeddings @ vector, dtype=float)
        scores[~allowed] = -1.0
        order = np.argsort(-scores)
        candidates = []
        for position in order:
            if scores[position] < -0.999:
                continue
            uid = unit_ids[int(position)]
            candidate = {
                "unit_id": uid,
                "embedding_score": round(float(scores[position]), 6),
                "rank": len(candidates) + 1,
                "unit": units[uid],
            }
            candidates.append(candidate)
            flattened.append({**candidate, "hop": subq["hop"],
                              "subquestion": subq["question"]})
            if len(candidates) >= max(1, a.hop_top_k):
                break
        hops.append({**subq, "query_text": query_text, "candidates": candidates})
    return {
        "query_id": plan["query_id"], "row_index": plan["row_index"],
        "query": plan["query"], "plan": plan["plan"],
        "reasoning_mode": "hopwise_candidates", "hop_top_k": a.hop_top_k,
        "embedding_model": a.reasoning_embedding_model,
        "gold_doc_ids": plan.get("gold_doc_ids", []),
        "oracle_gold_docs": a.oracle_gold_docs, "hops": hops,
        "retrieved_unit_ids": [x["unit_id"] for x in flattened],
        "candidates": flattened,
    }


def hopwise_retrieval_stage(a: argparse.Namespace, run: Path) -> None:
    units, _ = load_index(run)
    plans = read_jsonl(run / "04_query_plans" / "plans.jsonl")
    model, model_name = load_reasoning_embedder(a)
    embeddings, unit_ids = load_or_build_reasoning_embeddings(a, run, units, model, model_name)
    rows = [hopwise_retrieve_one(plan, units, embeddings, unit_ids, model, a)
            for plan in plans]
    stage = run / "05_retrieval"
    write_jsonl(stage / "retrieval.jsonl", rows)
    for row in rows:
        write_json(stage / (row["query_id"] + ".json"), row)
    write_json(stage / "summary.json", {
        "queries": len(rows), "reasoning_mode": "hopwise_candidates",
        "hop_top_k": a.hop_top_k, "max_hops": a.max_hops,
        "embedding_model": model_name, "indexed_units": len(units),
        "candidate_count_per_query": [len(x["candidates"]) for x in rows],
    })


def retrieve_one(plan: dict[str, Any], units: dict[str, dict[str, Any]],
                 indexes: dict[str, Any], top_k: int,
                 oracle_gold_docs: bool = False,
                 resolver: EntityCanonicalizer | None = None) -> dict[str, Any]:
    query = plan["query"]
    p = plan["plan"]
    resolver = resolver or EntityCanonicalizer(list(units.values()))
    qterms = set(tokens(query + " " + " ".join(plan_strings(p))))
    entities = [str(x) for x in p.get("entity_mentions", [])]
    types = [str(x) for x in p.get("type_hints", [])]
    props = [str(x) for x in p.get("properties", [])]
    sources = [str(x) for x in p.get("source_constraints", [])]
    gold_doc_ids = {norm(x) for x in plan.get("gold_doc_ids", [])}
    candidate_ids = set()
    if oracle_gold_docs:
        candidate_ids = {
            uid for uid, unit in units.items()
            if norm(unit.get("document", {}).get("parent_document_id") or
                    unit.get("document", {}).get("document_id")) in gold_doc_ids
        }
    else:
        for word in qterms:
            candidate_ids.update(indexes["token_index"].get(word, []))
        for key, values in (("entity_index", entities), ("type_index", types),
                            ("property_index", props), ("source_index", sources)):
            for value in values:
                candidate_ids.update(indexes[key].get(norm(value), []))
        # Query planning should retrieve both anchors and the units that can
        # continue a chain.  This is intentionally still lexical at the seed
        # stage; the structural expansion below is what makes the index useful
        # for causal long-chain reasoning.
        for value in entities + types:
            candidate_ids.update(indexes.get("subject_index", {}).get(norm(value), []))
            candidate_ids.update(indexes.get("object_index", {}).get(norm(value), []))
    if not candidate_ids and not oracle_gold_docs:
        candidate_ids = set(units)
    scored = []
    for uid in candidate_ids:
        unit = units[uid]
        uterms = set(tokens(unit.get("search_text", "")))
        overlap = len(qterms & uterms) / max(1, math.sqrt(len(qterms) * len(uterms)))
        ehit = any(norm(x) in {norm(unit.get("subject", "")), *map(norm, unit.get("aliases", []))}
                   for x in entities)
        thit = norm(unit.get("subject", "")) in {norm(x) for x in types}
        phit = norm(unit.get("predicate", "")) in {norm(x) for x in props}
        shit = norm(unit.get("document", {}).get("source", "")) in {norm(x) for x in sources}
        family = unit.get("relation_family") or relation_info(unit.get("predicate", ""))[0]
        causal_bonus = 0.35 if p.get("reasoning_mode") in {"causal_chain", "conditional_rule", "mixed"} and family == "causal" else 0.0
        type_bonus = 0.25 if p.get("reasoning_mode") in {"type_inheritance", "mixed"} and family == "type" else 0.0
        score = overlap + 1.5 * ehit + thit + .8 * phit + .8 * shit + causal_bonus + type_bonus
        scored.append({"unit_id": uid, "retrieval_score": round(score, 6),
                       "matched_terms": sorted(qterms & uterms), "unit": unit})
    scored.sort(key=lambda x: (-x["retrieval_score"], x["unit_id"]))

    # Structural expansion: retain lexical anchors, then pull in adjacent
    # states/edges through the subject/object and causal in/out indexes.  This
    # prevents a bridge fact from being discarded merely because it shares few
    # surface words with the question.
    seed_count = min(len(scored), max(5, top_k // 3))
    expanded = {item["unit_id"] for item in scored[:seed_count]}
    expansion_parent: dict[str, str] = {}
    frontier = [item for item in scored[:seed_count]]
    for _ in range(2):
        next_frontier = []
        for item in frontier:
            unit = item["unit"]
            terms = [unit.get("subject", ""), unit.get("object", ""),
                     state_signature(unit.get("source_state") or {}, resolver),
                     state_signature(unit.get("target_state") or {}, resolver),
                     state_join_signature(unit.get("source_state") or {}, resolver),
                     state_join_signature(unit.get("target_state") or {}, resolver)]
            neighbor_ids = set()
            for term in terms:
                key = norm(term)
                if not key:
                    continue
                neighbor_ids.update(indexes.get("subject_index", {}).get(key, []))
                neighbor_ids.update(indexes.get("object_index", {}).get(key, []))
                neighbor_ids.update(indexes.get("causal_out_index", {}).get(key, []))
                neighbor_ids.update(indexes.get("causal_in_index", {}).get(key, []))
                neighbor_ids.update(indexes.get("state_index", {}).get(key, []))
            for neighbor_id in neighbor_ids:
                neighbor = units.get(neighbor_id)
                neighbor_doc = norm((neighbor or {}).get("document", {}).get("parent_document_id") or
                                     (neighbor or {}).get("document", {}).get("document_id"))
                if (neighbor_id not in expanded and neighbor_id in units and
                        (not oracle_gold_docs or neighbor_doc in gold_doc_ids)):
                    expanded.add(neighbor_id)
                    expansion_parent.setdefault(neighbor_id, item["unit_id"])
                    next_frontier.append({"unit_id": neighbor_id, "unit": neighbor})
        frontier = next_frontier
        if not frontier:
            break

    # Re-score expanded candidates.  For an oracle run we still restrict to
    # the gold documents, but we no longer let top-k lexical ranking erase all
    # possible intermediate reasoning states.
    rescored = []
    for item in scored:
        item = dict(item)
        if item["unit_id"] in expansion_parent:
            item["structural_expansion"] = True
            item["expanded_from"] = expansion_parent[item["unit_id"]]
            item["retrieval_score"] = round(item["retrieval_score"] + 0.15, 6)
        rescored.append(item)
    by_id = {item["unit_id"]: item for item in rescored}
    for uid in expanded:
        if uid not in by_id and uid in units:
            unit = units[uid]
            by_id[uid] = {"unit_id": uid, "retrieval_score": 0.1,
                          "matched_terms": [], "structural_expansion": True,
                          "expanded_from": expansion_parent.get(uid), "unit": unit}
    rescored = sorted(by_id.values(), key=lambda x: (-x["retrieval_score"], x["unit_id"]))
    return {"query_id": plan["query_id"], "row_index": plan["row_index"],
            "query": query, "plan": p,
            "gold_doc_ids": sorted(gold_doc_ids),
            "oracle_gold_docs": oracle_gold_docs,
            "retrieval_mode": "anchor_plus_structural_expansion",
            "candidates": rescored[:top_k]}


def load_index(run: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    stage = run / "03_index"
    units = {x["unit_id"]: x for x in read_jsonl(stage / "unit_store.jsonl")}
    names = ("entity_index", "type_index", "property_index", "rule_index",
             "token_index", "source_index", "subject_index", "object_index",
             "alias_index", "state_index", "causal_out_index", "causal_in_index",
             "relation_family_index", "entity_type_index")
    indexes = {name: read_json(stage / (name + ".json")) if (stage / (name + ".json")).exists() else {}
               for name in names}
    return units, indexes


def retrieval_stage(a: argparse.Namespace, run: Path) -> None:
    if a.reasoning_mode == "hopwise_candidates":
        hopwise_retrieval_stage(a, run)
        return
    stage = run / "05_retrieval"
    units, indexes = load_index(run)
    resolver = EntityCanonicalizer(list(units.values()))
    plans = read_jsonl(run / "04_query_plans" / "plans.jsonl")
    rows = [retrieve_one(x, units, indexes, a.top_k, a.oracle_gold_docs, resolver) for x in plans]
    write_jsonl(stage / "retrieval.jsonl", rows)
    for row in rows:
        write_json(stage / (row["query_id"] + ".json"), row)
    write_json(stage / "summary.json", {"queries": len(rows), "top_k": a.top_k,
                                        "indexed_units": len(units),
                                        "oracle_gold_docs": a.oracle_gold_docs})


def derive_type_facts(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memberships, type_props = [], []
    for unit in units:
        predicate = norm(unit.get("predicate", ""))
        family, _, canonical = relation_info(predicate)
        if unit.get("kind") == "entity_type" or predicate in {
                "instance_of", "instanceof", "is_a", "type_of", "type of"}:
            membership = dict(unit)
            membership["predicate"] = canonical
            memberships.append(membership)
        if unit.get("kind") == "type_property" or unit.get("subject_kind") == "type":
            type_props.append(unit)
    derived = []
    for membership in memberships:
        for prop in type_props:
            if norm(prop.get("subject", "")) != norm(membership.get("object", "")):
                continue
            item = dict(prop)
            item.update({"unit_id": "derived_" + h([membership["unit_id"], prop["unit_id"]])[:20],
                         "kind": "derived_type_inheritance", "subject": membership["subject"],
                         "subject_kind": "entity", "derived_from": [membership["unit_id"], prop["unit_id"]],
                         "scope": "derived",
                         "evidence_quote": "Derived from " + membership["unit_id"] + " and " + prop["unit_id"]})
            item["search_text"] = " ".join(str(x) for x in
                                           (item["subject"], item["predicate"], item["object"]))
            derived.append(item)
    return derived


def build_graph(units: list[dict[str, Any]], qterms: set[str],
                max_hops: int, top_chains: int,
                plan: dict[str, Any] | None = None) -> dict[str, Any]:
    derived = derive_type_facts(units)
    all_units = units + derived
    resolver = EntityCanonicalizer(all_units)
    nodes, edges, forward_adjacency = {}, [], defaultdict(list)

    def node(label: Any, kind: str, key: Any | None = None) -> str:
        nid = kind + ":" + norm(key if key is not None else label)
        nodes.setdefault(nid, {"node_id": nid, "label": str(label), "kind": kind})
        return nid

    def edge(source: str, target: str, relation: str, unit: dict[str, Any], kind: str,
             family: str | None = None, surface_relation: str | None = None) -> None:
        family = family or unit.get("relation_family") or relation_info(relation)[0]
        value = {"source": source, "target": target, "relation": relation,
                 "unit_id": unit["unit_id"], "kind": kind,
                 "family": family, "direction": "forward",
                 "surface_relation": surface_relation or unit.get("predicate", relation),
                 "derived_from": unit.get("derived_from", [])}
        edges.append(value)
        # The old graph inserted both directions here, which allowed a causal
        # answer to be obtained by walking from effect back to cause.  Reverse
        # lookup is still available in the retrieval indexes; proof traversal
        # is intentionally directed.
        forward_adjacency[source].append((target, value))

    def state_node(atom_value: dict[str, Any], fallback: Any, kind: str = "state") -> str:
        label = atom_signature(atom_value) or str(fallback or "")
        key = state_join_signature(atom_value, resolver) or state_join_signature({"subject": fallback}, resolver)
        return node(label, kind, key=key)

    def add_flat_unit(unit: dict[str, Any]) -> None:
        subject = unit.get("subject", "")
        obj = unit.get("object", "")
        family, reverse, canonical = relation_info(unit.get("predicate", ""))
        if family == "causal" and (unit.get("source_state", {}).get("subject") or
                                    unit.get("target_state", {}).get("subject")):
            source = state_node(unit.get("source_state") or {}, subject)
            target = state_node(unit.get("target_state") or {}, obj)
        else:
            source_kind = "type" if unit.get("subject_kind") == "type" else "entity"
            target_kind = "value" if norm(obj) in {"true", "false"} else "concept"
            source = node(subject, source_kind, key=resolver.canonicalize(subject))
            target = node(obj, target_kind, key=resolver.canonicalize(obj))
        if reverse:
            source, target = target, source
        edge(source, target, canonical, unit, "derived" if unit["kind"].startswith("derived") else "fact",
             family=family, surface_relation=unit.get("predicate", canonical))

    for unit in all_units:
        if unit["kind"] == "rule":
            conditions = unit.get("conditions", [])
            conclusion = unit.get("conclusion", {})
            if conditions:
                rule = node(unit["unit_id"], "rule")
                for condition in conditions:
                    edge(state_node(condition, condition.get("subject", ""), "condition"), rule,
                         "satisfies", unit, "condition", family="rule")
                edge(rule, state_node(conclusion, conclusion.get("object", "")),
                     "concludes", unit, "conclusion", family="rule")
            else:
                # Legacy extraction often called a one-edge causal statement a
                # rule.  Preserve it as a directed edge rather than building a
                # dangling rule node with no conditions.
                add_flat_unit(unit)
        else:
            add_flat_unit(unit)

    mode = norm((plan or {}).get("reasoning_mode", ""))
    causal_mode = mode in {"causal_chain", "conditional_rule", "mixed"}
    target_terms = set(tokens((plan or {}).get("target_description", "")))
    anchors = [key for key, value in nodes.items() if qterms & set(tokens(value["label"]))]
    if not anchors:
        anchors = list(nodes)[:8]
    paths = []
    for start in anchors[:8]:
        stack = [(start, [start], [], {start})]
        expansions = 0
        while stack and expansions < 3000:
            current, node_path, edge_path, visited = stack.pop()
            expansions += 1
            if edge_path:
                labels = [nodes[x]["label"] for x in node_path]
                path_terms = set(tokens(" ".join(labels + [x["relation"] for x in edge_path])))
                families = {x.get("family", "other") for x in edge_path}
                allowed_families = {"causal", "type", "rule"}
                if (not causal_mode or
                        (families <= allowed_families and families & {"causal", "rule"})):
                    target_hit = len(target_terms & set(tokens(" ".join(labels))))
                    endpoint_target_hit = len(target_terms & set(tokens(labels[-1]))) if target_terms else 0
                    family_score = (1.5 * sum(x.get("family") == "causal" for x in edge_path)
                                    + .35 * sum(x.get("family") == "type" for x in edge_path)
                                    - .5 * sum(x.get("family") == "association" for x in edge_path))
                    paths.append({"node_labels": labels, "edges": edge_path,
                                  "target_hit": target_hit,
                                  "endpoint_target_hit": endpoint_target_hit,
                                  "score": len(qterms & path_terms) + target_hit
                                  + 2.0 * endpoint_target_hit + family_score
                                  + .15 * len(edge_path)})
            if len(edge_path) >= max_hops:
                continue
            for neighbor, value in forward_adjacency.get(current, []):
                if neighbor not in visited:
                    stack.append((neighbor, node_path + [neighbor],
                                  edge_path + [value], visited | {neighbor}))
    unique = {}
    for path in paths:
        key = "|".join(x["unit_id"] for x in path["edges"])
        if key and (key not in unique or path["score"] > unique[key]["score"]):
            unique[key] = path
    chains = sorted(unique.values(), key=lambda x: (-x["score"], len(x["edges"])))[:top_chains]
    return {"nodes": list(nodes.values()), "edges": edges, "units": all_units,
            "derived_units": derived, "chains": chains,
            "graph_diagnostics": {
                "causal_mode": causal_mode,
                "directed_edges": len(edges),
                "causal_edges": sum(x.get("family") == "causal" for x in edges),
                "type_edges": sum(x.get("family") == "type" for x in edges),
                "paths_considered": len(paths),
                "chains_with_causal_edge": sum(any(e.get("family") == "causal" for e in x["edges"])
                                                for x in chains),
                "chains_with_reasoning_edge": sum(any(e.get("family") in {"causal", "rule", "type"}
                                                        for e in x["edges"]) for x in chains),
                "chains_reaching_target": sum(x.get("endpoint_target_hit", 0) > 0 for x in chains),
            }}


def graph_stage(a: argparse.Namespace, run: Path) -> None:
    stage = run / "06_graph"
    outputs = []
    for row in read_jsonl(run / "05_retrieval" / "retrieval.jsonl"):
        units = [x["unit"] | {"retrieval_score": x["retrieval_score"]}
                 for x in row["candidates"]]
        qterms = set(tokens(row["query"] + " " + " ".join(plan_strings(row["plan"]))))
        graph = build_graph(units, qterms, a.max_hops, a.top_chains, row.get("plan", {}))
        output = {"query_id": row["query_id"], "row_index": row["row_index"],
                  "query": row["query"], "plan": row["plan"],
                  "retrieved_unit_ids": [x["unit_id"] for x in units], **graph}
        outputs.append(output)
        write_json(stage / (row["query_id"] + ".json"), output)
    write_jsonl(stage / "graphs.jsonl", outputs)
    write_json(stage / "summary.json", {"queries": len(outputs),
                                        "derived_units": sum(len(x["derived_units"]) for x in outputs)})


AGENT_SYSTEM = """You are the RESIN proof composer.
Answer only from the supplied evidence-backed reasoning units and directed
candidate chains. Treat each state as [entity][property][value]. For causal
questions, every major causal step must correspond to a forward edge in one of
the candidate chains. Do not complete a missing bridge with outside knowledge.
Distinguish causes, associations, type inheritance, diagnosis, and mere
co-occurrence. Bind repeated variables to the same entity. Use type
inheritance only when an explicit entity_type/instance_of path exists. Do not
transfer properties through represents unless the evidence supports it.
If the supplied chains do not support the requested conclusion, return
insufficient instead of guessing.
The `answer` field is the final response shown to the user and evaluated for
Answer Relevancy. For every supported answer, including boolean answers, write
one concise, direct, self-contained sentence that explicitly names the central
subject and relation asked about and includes the answer. A reader who sees
only the answer must be able to infer what question it answers. Do not return a
bare entity, date, number, noun phrase, `yes`, or `no` when a grammatical answer
sentence can be formed. For a boolean question, begin with `Yes.` or `No.` and
then state the supported proposition, including the relevant entities and the
shared or contrasting property when the evidence supplies it. Example:
`Yes. Ada North and Bruno Vale were both architects.` Answer every requested
part, but do not add background facts, citations, reasoning narration, or
unsupported detail.
Return JSON only:
{"answer":"","answer_type":"entity | boolean | date | explanation | unknown",
 "status":"supported | insufficient | conflicted",
 "supporting_unit_ids":[],"reasoning_steps":[],"missing_constraints":[],"confidence":0.0}
"""

VERIFY_SYSTEM = """You are the RESIN proof verifier.
Check every major question constraint against the proposed proof and directed
candidate chains. For causal questions, verify that each stated transition is
supported by a forward causal edge; an association, diagnosis edge, or shared
entity is not a causal bridge. Reject unsupported entity merges, unsupported
type inheritance, reversed causal direction, and claims imported from general
world knowledge. If a required bridge is absent, mark the proposal
insufficient.
Return JSON only:
{"status":"supported | insufficient | conflicted","answer_supported":true,
 "verified_unit_ids":[],"missing_constraints":[],"contradictions":[],
 "correction":"","confidence":0.0}
"""


HOP_SYSTEM = """You are a RESIN hop solver.
Answer only the supplied one-hop subquestion using the candidate reasoning
units for this hop. Do not use candidates from another hop and do not add
outside knowledge. The answer should expose the entity, property, value, or
bridge needed by later hops. If the candidates do not support the subquestion,
return insufficient rather than guessing.
Return JSON only:
{"answer":"","status":"supported | insufficient | conflicted",
 "supporting_unit_ids":[],"reasoning_step":"","confidence":0.0}
"""

FINAL_HOPWISE_SYSTEM = """You are the RESIN final answer composer.
The query was decomposed into ordered one-hop subquestions. Each hop contains
its retrieved candidate reasoning units and a provisional hop answer. Compose
the final answer using the full query, all hop candidates, and the hop answers.
Candidates are alternatives, not facts that are all simultaneously true: use
the best-supported candidate for each step, but keep a candidate available if
it resolves an entity or property ambiguity. Do not invent a bridge that is
not supported by the supplied units. Bind the same entity consistently and
respect causal direction and type inheritance. If the evidence is insufficient,
say so explicitly.
The `answer` field is the final response shown to the user and evaluated for
Answer Relevancy. For every supported answer, including boolean answers, write
one concise, direct, self-contained sentence that explicitly names the central
subject and relation asked about and includes the answer. A reader who sees
only the answer must be able to infer what question it answers. Do not return a
bare entity, date, number, noun phrase, `yes`, or `no` when a grammatical answer
sentence can be formed. For a boolean question, begin with `Yes.` or `No.` and
then state the supported proposition, including the relevant entities and the
shared or contrasting property when the evidence supplies it. Example:
`Yes. Ada North and Bruno Vale were both architects.` Answer every requested
part, but do not add background facts, citations, reasoning narration, or
unsupported detail.
Return JSON only:
{"answer":"","answer_type":"entity | boolean | date | explanation | unknown",
 "status":"supported | insufficient | conflicted",
 "supporting_unit_ids":[],"reasoning_steps":[],"missing_constraints":[],
 "confidence":0.0}
"""


HOLISQA_HOP_SYSTEM = HOP_SYSTEM + """
For a HolisQA subquestion, retain every distinct supported finding needed for
the final synthesis, including mechanisms, conditions, comparisons, outcomes,
and named entities. A provisional hop answer may use two short sentences when
one sentence would omit a supported component. Preserve the source's polarity
and uncertainty; do not pad with general background.
"""


HOLISQA_SOURCE_SELECTION_SYSTEM = """Select the retrieved source article most likely to contain a direct answer to the question.
Use the exact subject, domain, and requested findings, not just broad topic words. Titles may be in another language; use the provided English snippets to understand them. The list contains distractors. Return JSON only: {"article_ids":[0]} with one or two integer article IDs from the list. Choose the most directly relevant first. Do not answer the question.
"""


HOLISQA_FINAL_SYSTEM = """Use only the retrieved source passages to answer the research question.
Return JSON with answer first: {"answer":"...","status":"supported | insufficient","supporting_unit_ids":[]}.
Form a self-contained answer of two to four short sentences, about 55-85 words. Cover each requested factor, mechanism, comparison, outcome, and condition found in the passages. Reuse the source's exact technical terms and short phrases for the central findings; avoid broad synonyms that erase the findings. Preserve polarity and uncertainty.
Begin with the substantive answer. Do not describe the analysis, give background definitions, list unrelated possibilities, cite sources, or add a generic conclusion. If evidence is partial, give the supported facts without a long caveat. Never use a reference answer or outside knowledge.
"""


HOLISQA_ANSWER_REPAIR_SYSTEM = """Answer the question from the supplied evidence only.
The previous response omitted the required answer field. Return JSON with a
nonempty answer field first: {"answer":"...","status":"supported | insufficient",
"supporting_unit_ids":[]}. Give a direct, self-contained answer to the
supported parts of the question. Do not add outside facts or citations.
"""


QASC_CHOICE_SYSTEM = """Answer the multiple-choice question using only the retrieved facts.
Some questions require connecting two facts. Select exactly one of the provided
choices; use the exact choice label. Return JSON only:
{"choice":"A","sentence":"...","status":"supported | insufficient"}.
The sentence must be one short, self-contained answer containing the exact
chosen choice text and the relation asked by the question. Do not include a
choice letter in the sentence. If the evidence does not support any choice,
return an empty choice and sentence with status insufficient. Do not use a
reference answer, gold evidence, or outside knowledge.
"""


QASC_FORMAT_FIRST_SYSTEM = """Turn the selected answer into one natural sentence
that directly answers the question. Keep the selected answer text exactly as
supplied. Use the question's wording for the main fact, with no added facts.
Aim for 5 to 12 words total. Do not mention an answer, a choice, a letter, a
subject, a relation, or the question itself. Do not explain your reasoning.
Return JSON only: {"answer":"..."}.
"""


QASC_FORMAT_SYSTEM = """Write one ordinary declarative sentence that states the
requested fact. The selected text is already chosen and must occur verbatim in
the sentence. Convert the question into a statement with that text as its
answer. Use only the words needed to identify the entity and the asked
relation; aim for 5 to 12 words. The sentence must have no question mark,
colon, choice letter, meta commentary, explanation, or new facts. Return JSON
only: {"answer":"..."}.
"""


MSMARCO_FINAL_SYSTEM = """Answer the question using only the retrieved passages. Return JSON
only: {"answer":"...","status":"supported | insufficient"}.
Give the direct answer first. Use the passage's exact names, numbers, units,
and key terms. For a requested name, date, amount, or other short value,
return only the answer phrase. For a definition, reason, or process, give one
concise self-contained sentence containing the main requested fact. Include
all parts explicitly asked, but no extra background, caveats, citations,
unrelated details, or repeated question. Do not begin with yes or no unless
the question itself is yes/no. If the passages do not support an answer,
return an empty answer and status insufficient. Do not use outside knowledge
or a reference answer.
"""


HOLISQA_QUERY_STOPWORDS = set(
    "the a an and or of to in on for by with from as is are was were be been "
    "being this that these those their its it at which what how does do did "
    "into through within about also can could may should have has had not "
    "but more most other such both all".split()
)


def holisqa_query_terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.lower())
            if len(term) > 2 and term not in HOLISQA_QUERY_STOPWORDS}


def generation_signature(a: argparse.Namespace) -> str:
    if a.generation_style == "holisqa":
        prompt = (HOLISQA_HOP_SYSTEM + HOLISQA_SOURCE_SELECTION_SYSTEM
                  + HOLISQA_FINAL_SYSTEM + HOLISQA_ANSWER_REPAIR_SYSTEM)
    elif a.generation_style == "qasc":
        prompt = QASC_CHOICE_SYSTEM + QASC_FORMAT_FIRST_SYSTEM + QASC_FORMAT_SYSTEM
    elif a.generation_style == "msmarco":
        prompt = MSMARCO_FINAL_SYSTEM
    else:
        prompt = HOP_SYSTEM + FINAL_HOPWISE_SYSTEM
    packing = ("source_article_selector_v3" if a.generation_style == "holisqa"
               else "retrieved_fact_choice_reuse_sentence_v3" if a.generation_style == "qasc"
               else "retrieved_passages_top5_per_hop_v1" if a.generation_style == "msmarco"
               else "source_passages_v1")
    return h([packing, a.generation_style, prompt, a.model, a.base_url,
              a.max_tokens, a.temperature, a.reasoning_effort])


def generation_source_passages(corpus_file: Path) -> dict[str, dict[str, str]]:
    """Read source text only; this does not introduce new retrieval candidates."""
    passages = {}
    for row in load_records(corpus_file):
        document_id = str(row.get("document_id") or row.get("id") or "")
        body = str(row.get("body") or row.get("text") or "").strip()
        if document_id and body:
            passages[document_id] = {
                "document_id": document_id,
                "title": str(row.get("title") or ""),
                "text": body,
            }
    return passages


def retrieved_source_passages(
        retrieval: dict[str, Any], source_by_id: dict[str, dict[str, str]],
        max_chars: int = 16000,
) -> list[dict[str, str]]:
    """Deduplicate already-retrieved passages, balancing candidate ranks across hops."""
    hops = retrieval.get("hops", [])
    depth = max((len(hop.get("candidates", [])) for hop in hops), default=0)
    seen, selected, used = set(), [], 0
    for rank in range(depth):
        for hop in hops:
            candidates = hop.get("candidates", [])
            if rank >= len(candidates):
                continue
            unit = candidates[rank].get("unit", {})
            document = unit.get("document", {})
            document_id = str(document.get("document_id") or unit.get("doc_id") or "")
            if document_id in seen or document_id not in source_by_id:
                continue
            passage = source_by_id[document_id]
            if used + len(passage["text"]) > max_chars:
                continue
            selected.append(passage)
            seen.add(document_id)
            used += len(passage["text"])
    return selected


def focused_holisqa_payload(
        retrieval: dict[str, Any], hop_records: list[dict[str, Any]],
        source_by_id: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], int]:
    """Pack existing retrieval for final generation; never inspect QA answers."""
    passages = retrieved_source_passages(retrieval, source_by_id)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for passage in passages:
        groups[passage["title"]].append(passage)
    query_terms = holisqa_query_terms(retrieval["query"])
    titles = sorted(
        groups,
        key=lambda title: (
            -len(query_terms & holisqa_query_terms(
                title + " " + " ".join(passage["text"] for passage in groups[title])
            )),
            -len(groups[title]),
        ),
    )[:3]
    selected = [passage for title in titles for passage in groups[title]]
    selected_ids = {passage["document_id"] for passage in selected}
    final_hops = []
    for hop, record in zip(retrieval.get("hops", []), hop_records):
        candidates = [
            candidate for candidate in hop.get("candidates", [])
            if str(candidate.get("unit", {}).get("document", {}).get("document_id", ""))
            in selected_ids
        ][:4]
        if not candidates:
            candidates = hop.get("candidates", [])[:2]
        draft = record.get("answer", {})
        final_hops.append({
            "subquestion": hop.get("question", ""),
            "draft_answer": draft.get("answer", "") if isinstance(draft, dict) else "",
            "evidence": [{
                "unit_id": candidate.get("unit_id", ""),
                "quote": candidate.get("unit", {}).get("evidence_quote", ""),
                "subject": candidate.get("unit", {}).get("subject", ""),
                "predicate": candidate.get("unit", {}).get("predicate", ""),
                "object": candidate.get("unit", {}).get("object", ""),
            } for candidate in candidates],
        })
    return {
        "question": retrieval["query"],
        "source_passages": selected,
        "hops": final_hops,
    }, sum(len(passage["text"]) for passage in selected)


def select_holisqa_source_passages(
        retrieval: dict[str, Any], source_by_id: dict[str, dict[str, str]],
        client: LLMClient, max_tokens: int,
        selector_system: str = HOLISQA_SOURCE_SELECTION_SYSTEM,
        include_subquestions: bool = False, snippet_chars: int = 450,
        target_articles: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose among already-retrieved articles using only question and source text."""
    passages = retrieved_source_passages(retrieval, source_by_id)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for passage in passages:
        groups[passage["title"]].append(passage)
    titles = list(groups)
    if not titles:
        return {"question": retrieval["query"], "source_passages": []}, {
            "status": "no_source_passages", "selected_article_ids": [],
        }
    articles = [{
        "id": index, "title": title,
        "snippets": [passage["text"][:snippet_chars] for passage in groups[title][:2]],
    } for index, title in enumerate(titles)]
    selection_payload = {"question": retrieval["query"], "articles": articles}
    if include_subquestions:
        selection_payload["subquestions"] = [
            str(hop.get("question", "")) for hop in retrieval.get("hops", [])
            if str(hop.get("question", "")).strip()
        ]
    messages = [
        {"role": "system", "content": selector_system},
        {"role": "user", "content": json.dumps(selection_payload, ensure_ascii=False)},
    ]
    record: dict[str, Any] = {"request_messages": messages}
    try:
        response, content, choice = client.call_json(messages, min(max_tokens, 500))
        raw_ids = choice.get("article_ids", [])
        ids = []
        if isinstance(raw_ids, list):
            for value in raw_ids:
                if type(value) is int and 0 <= value < len(titles) and value not in ids:
                    ids.append(value)
                if len(ids) == 2:
                    break
        if target_articles is not None and ids:
            for index in range(len(titles)):
                if len(ids) >= min(target_articles, len(titles)):
                    break
                if index not in ids:
                    ids.append(index)
        record.update({"response": response, "response_content": content,
                       "proposal": choice, "selected_article_ids": ids})
        if ids:
            selected = [passage for index in ids
                        for passage in groups[titles[index]]]
            record["status"] = "selected"
            return {"question": retrieval["query"],
                    "source_passages": selected}, record
        record["error"] = "Selector returned no valid article IDs"
    except Exception as exc:
        record["error"] = repr(exc)
    # A selector failure should not turn an otherwise answerable query into an
    # empty generation. This deterministic fallback still uses retrieval only.
    if target_articles is not None:
        fallback_passages = [passage for title in titles[:target_articles]
                             for passage in groups[title]]
    else:
        fallback, _ = focused_holisqa_payload(retrieval, [], source_by_id)
        fallback_passages = fallback["source_passages"]
    record["status"] = "fallback"
    return {"question": retrieval["query"],
            "source_passages": fallback_passages}, record


def selected_generation_rows(
        a: argparse.Namespace, qa_file: Path,
        retrieval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select QA rows by original dataset order without rerunning retrieval."""
    if a.generation_domain is None and a.generation_limit is None:
        return retrieval_rows
    if a.generation_limit is not None and a.generation_limit < 1:
        raise ValueError("--generation_limit must be positive")
    selected_ids = []
    for index, row in enumerate(load_records(qa_file)):
        if (a.generation_domain is not None
                and norm(row.get("category", row.get("domain", "")))
                != norm(a.generation_domain)):
            continue
        query = str(row.get("query", ""))
        selected_ids.append("q_" + str(index).zfill(6) + "_" + h(query)[:8])
        if a.generation_limit is not None and len(selected_ids) >= a.generation_limit:
            break
    if not selected_ids:
        raise ValueError("No QA rows match the requested generation selection")
    by_id = {row["query_id"]: row for row in retrieval_rows}
    missing = [qid for qid in selected_ids if qid not in by_id]
    if missing:
        raise ValueError(
            f"Existing retrieval lacks {len(missing)} selected query IDs: "
            + ", ".join(missing[:5])
        )
    return [by_id[qid] for qid in selected_ids]


def compact(unit: dict[str, Any]) -> dict[str, Any]:
    keys = ("unit_id", "kind", "subject", "subject_kind", "predicate", "object",
            "conditions", "conclusion", "source_state", "target_state",
            "relation_family", "canonical_relation", "modality", "polarity",
            "scope", "derived_from", "evidence_quote")
    # Preserve every populated evidence field while avoiding repeated null/
    # empty values in every hop and final-composer prompt.
    return {
        key: unit.get(key) for key in keys
        if unit.get(key) not in (None, "", [], {})
    }


def bare_boolean_value(result: dict[str, Any],
                       plan: dict[str, Any] | None = None) -> str | None:
    """Return yes/no only when a supported boolean answer is still bare."""
    if not isinstance(result, dict):
        return None
    if norm(result.get("status", "supported")) != "supported":
        return None
    answer_type = norm(result.get("answer_type", ""))
    if not answer_type and isinstance(plan, dict):
        answer_type = norm(plan.get("answer_type", ""))
    if answer_type != "boolean":
        return None
    answer = str(result.get("answer", "") or "").strip()
    key = re.sub(r"[^a-z]+", "", answer.lower())
    return {"true": "yes", "yes": "yes", "false": "no", "no": "no"}.get(key)


BOOLEAN_REWRITE_SYSTEM = """You format an already-decided boolean answer.
Preserve the supplied yes/no value exactly and use only the supplied evidence.
Rewrite it as one concise, direct, self-contained sentence. Begin with `Yes.`
or `No.`; name the entities and relation from the original question; and state
the shared, differing, or requested property when the evidence supports it.
Do not add citations, reasoning narration, or outside knowledge.
Return JSON only: {"answer":""}
"""


def ensure_self_contained_boolean_answer(
        result: dict[str, Any], plan: dict[str, Any] | None, query: str,
        evidence_payload: dict[str, Any], client: LLMClient, max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Expand a bare supported yes/no answer without changing its truth value.

    The normal final-composer call should already produce the requested format.
    This guarded second call runs only when the composer nevertheless emits a
    bare boolean, avoiding an AR failure caused solely by answer formatting.
    """
    value = bare_boolean_value(result, plan)
    if value is None:
        return result, None
    payload = {
        "query": query,
        "boolean_answer": value,
        "query_plan": plan or {},
        "evidence": evidence_payload,
    }
    messages = [
        {"role": "system", "content": BOOLEAN_REWRITE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    record: dict[str, Any] = {"request_messages": messages}
    try:
        response, content, rewritten = client.call_json(
            messages, min(max_tokens, 800)
        )
        answer = str(rewritten.get("answer", "") or "").strip()
        record.update({"response": response, "response_content": content,
                       "rewritten": rewritten})
        rewritten_key = re.sub(r"[^a-z]+", "", answer.lower())
        if answer and rewritten_key not in {"yes", "no", "true", "false"}:
            expanded = dict(result)
            expanded["answer"] = answer
            expanded["answer_type"] = "boolean"
            return expanded, record
        record["error"] = "Formatter returned another bare boolean answer."
    except Exception as exc:
        # Formatting failure must not discard an otherwise valid answer.
        record["error"] = repr(exc)
    return result, record


def qasc_generation_payload(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Use only the question's choices and already retrieved reasoning units."""
    parts = re.split(r"(?<!\w)\(([A-Z])\)\s*", retrieval["query"])
    choices = [
        {"label": parts[index], "text": parts[index + 1].strip()}
        for index in range(1, len(parts), 2)
    ]
    if (len(choices) < 2 or len({item["label"] for item in choices}) != len(choices)
            or any(not item["text"] for item in choices)):
        raise ValueError("Question has no valid labeled answer choices")
    facts, seen = [], set()
    for hop in retrieval.get("hops", []):
        for candidate in hop.get("candidates", [])[:10]:
            unit = candidate.get("unit", {})
            fact = str(unit.get("object") or unit.get("evidence_quote") or "").strip()
            if fact and fact not in seen:
                facts.append({
                    "unit_id": str(candidate.get("unit_id") or ""),
                    "fact": fact,
                })
                seen.add(fact)
    return {
        "question": parts[0].strip(),
        "choices": choices,
        "retrieved_facts": facts,
    }


def qasc_valid_sentence(sentence: str, choice_text: str) -> bool:
    """Accept only a declarative sentence that retains the selected choice."""
    return bool(
        sentence and choice_text and "?" not in sentence and ":" not in sentence
        and sentence.casefold() != choice_text.casefold()
        and choice_text.casefold() in sentence.casefold()
    )


def qasc_generation_task(retrieval: dict[str, Any], client: LLMClient,
                         stage: Path, a: argparse.Namespace) -> Path:
    """Select a supported choice, then phrase it without changing the choice."""
    path = stage / (retrieval["query_id"] + ".json")
    signature = generation_signature(a)
    if not a.no_resume and path.exists():
        cached = read_json(path)
        if (cached.get("status") == "ok"
                and cached.get("generation_signature") == signature):
            return path
    output: dict[str, Any] = {
        "status": "error", "query_id": retrieval["query_id"],
        "query": retrieval["query"], "reasoning_mode": "hopwise_candidates",
        "generation_style": "qasc", "generation_signature": signature,
    }
    try:
        payload = qasc_generation_payload(retrieval)
        output["generation_fact_count"] = len(payload["retrieved_facts"])
        choice_messages = [
            {"role": "system", "content": QASC_CHOICE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        choice_client = LLMClient(
            client.base, client.model, client.key, client.timeout,
            client.retries, client.temperature, client.provider, "medium",
        )
        response, content, proposal = choice_client.call_json(choice_messages, 4000)
        options = {item["label"]: item["text"] for item in payload["choices"]}
        label = str(proposal.get("choice") or "").strip().upper()
        if label not in options:
            output["empty_primary_choice"] = {
                "response": response, "response_content": content,
                "proposal": proposal,
            }
            fallback_client = LLMClient(
                client.base, client.model, client.key, client.timeout,
                client.retries, client.temperature, client.provider, "minimal",
            )
            response, content, proposal = fallback_client.call_json(
                choice_messages, 1800
            )
            label = str(proposal.get("choice") or "").strip().upper()
            output["choice_retry_after_empty"] = True
        choice_text = options.get(label, "")
        answer = choice_text
        format_record: dict[str, Any] | None = None
        if choice_text:
            choice_sentence = str(proposal.get("sentence") or "").strip()
            if qasc_valid_sentence(choice_sentence, choice_text):
                answer = choice_sentence
                format_record = {"source": "choice_selection_sentence"}
        if choice_text and format_record is None:
            format_payload = {
                "question": payload["question"],
                "selected_text": choice_text,
            }
            format_user = json.dumps(format_payload, ensure_ascii=False)
            first_messages = [
                {"role": "system", "content": QASC_FORMAT_FIRST_SYSTEM},
                {"role": "user", "content": format_user},
            ]
            format_record = {}
            candidate = ""
            try:
                format_client = LLMClient(
                    client.base, client.model, client.key, client.timeout,
                    client.retries, client.temperature, client.provider, "minimal",
                )
                fr, fc, formatted = format_client.call_json(first_messages, 800)
                candidate = str(formatted.get("answer") or "").strip()
                format_record["first_attempt"] = {
                    "request_messages": first_messages,
                    "response": fr, "response_content": fc, "proposal": formatted,
                }
            except Exception as exc:
                format_record["first_error"] = repr(exc)
            valid = qasc_valid_sentence(candidate, choice_text)
            if not valid:
                repair_messages = [
                    {"role": "system", "content": QASC_FORMAT_SYSTEM},
                    {"role": "user", "content": format_user},
                ]
                try:
                    rr, rc, repaired = choice_client.call_json(repair_messages, 1800)
                    candidate = str(repaired.get("answer") or "").strip()
                    format_record["repair_attempt"] = {
                        "request_messages": repair_messages,
                        "response": rr, "response_content": rc,
                        "proposal": repaired,
                    }
                    valid = qasc_valid_sentence(candidate, choice_text)
                except Exception as exc:
                    format_record["repair_error"] = repr(exc)
            if valid:
                answer = candidate
            else:
                format_record["fallback"] = "selected_choice_text"
        final = {
            "answer": answer,
            "status": "supported" if choice_text else "insufficient",
            "selected_choice_label": label if choice_text else "",
            "selected_choice_text": choice_text,
            "supporting_unit_ids": [],
        }
        output.update({
            "status": "ok", "hop_answers": [],
            "choice_selection": {
                "request_messages": choice_messages, "response": response,
                "response_content": content, "proposal": proposal,
            },
            "answer_formatting": format_record,
            "final": {"proposal": final}, "final_proposal": final,
            "retrieved_unit_ids": retrieval.get("retrieved_unit_ids", []),
        })
    except Exception as exc:
        output["error"] = repr(exc)
    write_json(path, output)
    return path


def msmarco_generation_payload(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Pack existing retrieved passages, alternating ranks across hops."""
    passages: list[dict[str, str]] = []
    seen: set[str] = set()
    for rank in range(5):
        for hop in retrieval.get("hops", []):
            candidates = hop.get("candidates", [])
            if rank >= len(candidates):
                continue
            item = candidates[rank]
            unit = item.get("unit", {})
            passage = str(unit.get("object") or unit.get("evidence_quote") or "").strip()
            if passage and passage not in seen:
                passages.append({
                    "id": str(item.get("unit_id") or ""), "text": passage,
                })
                seen.add(passage)
    return {"question": retrieval["query"], "retrieved_passages": passages}


def msmarco_generation_task(retrieval: dict[str, Any], client: LLMClient,
                            stage: Path, a: argparse.Namespace) -> Path:
    """Generate from retrieval output alone; references enter only post hoc."""
    path = stage / (retrieval["query_id"] + ".json")
    signature = generation_signature(a)
    if not a.no_resume and path.exists():
        cached = read_json(path)
        if (cached.get("status") == "ok"
                and cached.get("generation_signature") == signature):
            return path
    output: dict[str, Any] = {
        "status": "error", "query_id": retrieval["query_id"],
        "query": retrieval["query"], "reasoning_mode": "hopwise_candidates",
        "generation_style": "msmarco", "generation_signature": signature,
    }
    try:
        payload = msmarco_generation_payload(retrieval)
        output["generation_passage_count"] = len(payload["retrieved_passages"])
        messages = [
            {"role": "system", "content": MSMARCO_FINAL_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response, content, proposal = client.call_json(messages, a.max_tokens)
        answer = str(proposal.get("answer") or "").strip()
        final = {
            "answer": answer,
            "status": "supported" if answer else "insufficient",
            "supporting_unit_ids": [],
        }
        output.update({
            "status": "ok", "hop_answers": [],
            "final": {"request_messages": messages, "response": response,
                      "response_content": content, "proposal": proposal},
            "final_proposal": final,
            "retrieved_unit_ids": retrieval.get("retrieved_unit_ids", []),
        })
    except Exception as exc:
        output["error"] = repr(exc)
    write_json(path, output)
    return path


def hopwise_agent_task(retrieval: dict[str, Any], client: LLMClient,
                       stage: Path, a: argparse.Namespace,
                       source_by_id: dict[str, dict[str, str]] | None = None) -> Path:
    if a.generation_style == "qasc":
        return qasc_generation_task(retrieval, client, stage, a)
    if a.generation_style == "msmarco":
        return msmarco_generation_task(retrieval, client, stage, a)
    path = stage / (retrieval["query_id"] + ".json")
    signature = generation_signature(a)
    if not a.no_resume and path.exists():
        cached = read_json(path)
        if (cached.get("status") == "ok"
                and (cached.get("generation_signature") == signature
                     or (a.generation_style == "concise"
                         and cached.get("generation_signature") is None))):
            return path
    output: dict[str, Any] = {
        "status": "error", "query_id": retrieval["query_id"],
        "query": retrieval["query"], "reasoning_mode": "hopwise_candidates",
        "generation_style": a.generation_style,
        "generation_signature": signature,
    }
    hop_records = []
    try:
        hops = retrieval.get("hops", [])

        def solve_hop(hop: dict[str, Any]) -> dict[str, Any]:
            candidates = [{
                "unit_id": item["unit_id"],
                "embedding_score": item.get("embedding_score"),
                "rank": item.get("rank"),
                "unit": compact(item.get("unit", {})),
            } for item in hop.get("candidates", [])]
            payload = {
                "hop": hop.get("hop"), "subquestion": hop.get("question", ""),
                "target": hop.get("target", ""),
                "depends_on": hop.get("depends_on", []),
                "candidate_reasoning_units": candidates,
            }
            hop_system = (HOLISQA_HOP_SYSTEM if a.generation_style == "holisqa"
                          else HOP_SYSTEM)
            messages = [{"role": "system", "content": hop_system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}]
            response, content, answer = client.call_json(messages, a.max_tokens)
            return {
                "hop": hop.get("hop"), "subquestion": hop.get("question", ""),
                "candidate_unit_ids": [x["unit_id"] for x in candidates],
                "request_messages": messages, "response": response,
                "response_content": content, "answer": answer,
            }

        # Retrieval is already complete and each hop request receives only its
        # own subquestion/candidates; no prior hop answer is consumed here.
        # Running these independent requests concurrently changes wall-clock
        # scheduling, not the reasoning inputs or final composition contract.
        hop_workers = min(max(1, a.hop_workers), max(1, len(hops)))
        if hop_workers > 1 and len(hops) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=hop_workers) as pool:
                hop_records = list(pool.map(solve_hop, hops))
        else:
            hop_records = [solve_hop(hop) for hop in hops]

        if a.generation_style == "holisqa":
            final_payload, source_selection = select_holisqa_source_passages(
                retrieval, source_by_id or {}, client, a.max_tokens,
            )
            output["source_selection"] = source_selection
            output["generation_source_passages"] = len(final_payload["source_passages"])
            output["generation_source_characters"] = sum(
                len(passage["text"]) for passage in final_payload["source_passages"]
            )
        else:
            final_hops = []
            for hop, record in zip(retrieval.get("hops", []), hop_records):
                final_hops.append({
                    "hop": hop.get("hop"), "subquestion": hop.get("question", ""),
                    "candidates": [{
                        "unit_id": item["unit_id"],
                        "embedding_score": item.get("embedding_score"),
                        "rank": item.get("rank"),
                        "unit": compact(item.get("unit", {})),
                    } for item in hop.get("candidates", [])],
                    "hop_answer": record["answer"],
                })
            final_payload = {
                "query": retrieval["query"], "query_plan": retrieval["plan"],
                "hops": final_hops,
            }
        final_system = (HOLISQA_FINAL_SYSTEM if a.generation_style == "holisqa"
                        else FINAL_HOPWISE_SYSTEM)
        final_messages = [{"role": "system", "content": final_system},
                          {"role": "user", "content": json.dumps(
                              final_payload, ensure_ascii=False)}]
        final_max_tokens = (min(a.max_tokens, 1400)
                            if a.generation_style == "holisqa"
                            else a.max_tokens)
        response, content, final = client.call_json(final_messages, final_max_tokens)
        if a.generation_style == "holisqa" and not str(final.get("answer", "") or "").strip():
            repair_messages = [
                {"role": "system", "content": HOLISQA_ANSWER_REPAIR_SYSTEM},
                {"role": "user", "content": json.dumps(final_payload, ensure_ascii=False)},
            ]
            repair_response, repair_content, repaired = client.call_json(
                repair_messages, a.max_tokens
            )
            output["answer_repair"] = {
                "request_messages": repair_messages,
                "response": repair_response,
                "response_content": repair_content,
                "proposal": repaired,
            }
            if not str(repaired.get("answer", "") or "").strip():
                raise ValueError("Final generation omitted a nonempty answer after repair")
            final = repaired
        final, answer_formatting = ensure_self_contained_boolean_answer(
            final, retrieval.get("plan", {}), retrieval["query"],
            final_payload, client, a.max_tokens,
        )
        output.update({
            "status": "ok", "hop_answers": hop_records,
            "final": {"request_messages": final_messages, "response": response,
                       "response_content": content, "proposal": final},
            "answer_formatting": answer_formatting,
            "final_proposal": final,
            "retrieved_unit_ids": retrieval.get("retrieved_unit_ids", []),
        })
    except Exception as exc:
        output["error"] = repr(exc)
    write_json(path, output)
    return path


def hopwise_agent_stage(a: argparse.Namespace, run: Path) -> None:
    stage = a.generation_output_dir or run / "07_agent"
    corpus_file, qa_file = resolve_files(a)
    retrieval_rows = selected_generation_rows(
        a, qa_file, read_jsonl(run / "05_retrieval" / "retrieval.jsonl")
    )
    source_by_id = (generation_source_passages(corpus_file)
                    if a.generation_style == "holisqa" else {})
    client = LLMClient(a.base_url, a.model, a.api_key, a.timeout, a.retries,
                       a.temperature, a.provider, a.reasoning_effort)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = [pool.submit(hopwise_agent_task, row, client, stage, a, source_by_id)
                   for row in retrieval_rows]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == len(futures):
                print("[hop-agent] completed", done, "/", len(futures))
    answers, errors = [], []
    source_counts = []
    for path in [stage / (row["query_id"] + ".json") for row in retrieval_rows]:
        item = read_json(path)
        if a.generation_style == "holisqa":
            source_counts.append(int(item.get("generation_source_passages", 0)))
        if item.get("status") != "ok":
            errors.append(item)
            continue
        final = item.get("final_proposal", {})
        answers.append({
            "query_id": item["query_id"], "query": item["query"],
            "answer": final.get("answer", ""), "status": final.get("status", ""),
            "supporting_unit_ids": final.get("supporting_unit_ids", []),
            "reasoning_steps": final.get("reasoning_steps", []),
            "confidence": final.get("confidence", 0),
            "reasoning_mode": "hopwise_candidates",
            "generation_style": a.generation_style,
            "answer_components": final.get("answer_components", []),
            "hop_answers": item.get("hop_answers", []),
            "retrieved_unit_ids": item.get("retrieved_unit_ids", []),
        })
    write_jsonl(stage / "answers.jsonl", answers)
    write_jsonl(stage / "errors.jsonl", errors)
    summary = {
        "queries": len(retrieval_rows), "successful": len(answers),
        "errors": len(errors), "model": client.model,
        "reasoning_mode": "hopwise_candidates", "hop_top_k": a.hop_top_k,
        "generation_style": a.generation_style,
        "generation_signature": generation_signature(a),
        "generation_domain": a.generation_domain,
        "generation_limit": a.generation_limit,
    }
    if a.generation_style == "holisqa":
        summary["queries_without_source_passages"] = sum(
            count == 0 for count in source_counts
        )
        summary["average_source_passages_in_generation"] = round(
            sum(source_counts) / max(1, len(source_counts)), 2
        )
    if a.generation_style == "msmarco":
        passage_counts = [
            int(read_json(stage / (row["query_id"] + ".json")).get(
                "generation_passage_count", 0))
            for row in retrieval_rows
        ]
        summary["average_retrieved_passages_in_generation"] = round(
            sum(passage_counts) / max(1, len(passage_counts)), 2
        )
    if a.generation_style == "qasc":
        fact_counts = [
            int(read_json(stage / (row["query_id"] + ".json")).get(
                "generation_fact_count", 0))
            for row in retrieval_rows
        ]
        summary["average_retrieved_facts_in_generation"] = round(
            sum(fact_counts) / max(1, len(fact_counts)), 2
        )
    if a.generation_style in {"holisqa", "qasc", "msmarco"}:
        qa_by_id = {}
        for index, row in enumerate(load_records(qa_file)):
            query = str(row.get("query", ""))
            qa_by_id["q_" + str(index).zfill(6) + "_" + h(query)[:8]] = row
        gold_by_id = {
            qid: str(row.get("answer", "") or "")
            for qid, row in qa_by_id.items()
        }
        answers_by_id = {row["query_id"]: row for row in answers}
        subset_dir = stage / "evaluation_subset"
        subset_qa, subset_answers, subset_retrieval = [], [], []
        for index, row in enumerate(retrieval_rows):
            original_id = row["query_id"]
            subset_id = "q_" + str(index).zfill(6) + "_" + h(row["query"])[:8]
            subset_qa.append(qa_by_id[original_id])
            subset_retrieval.append({**row, "query_id": subset_id,
                                     "row_index": index,
                                     "original_query_id": original_id})
            if original_id in answers_by_id:
                subset_answers.append({**answers_by_id[original_id],
                                       "query_id": subset_id,
                                       "original_query_id": original_id})
        write_jsonl(subset_dir / "qa.jsonl", subset_qa)
        write_jsonl(subset_dir / "answers.jsonl", subset_answers)
        write_jsonl(subset_dir / "retrieval.jsonl", subset_retrieval)
        scored = []
        for row in retrieval_rows:
            qid = row["query_id"]
            gold = gold_by_id.get(qid, "")
            prediction = str(answers_by_id.get(qid, {}).get("answer", "") or "")
            scored.append({
                "query_id": qid, "query": row["query"],
                "prediction": prediction, "gold_answer": gold,
                "f1": answer_f1(prediction, gold),
            })
        write_jsonl(stage / "f1_per_query.jsonl", scored)
        summary["F1_diagnostic"] = round(
            100 * sum(row["f1"] for row in scored) / max(1, len(scored)), 2
        )
        baseline_path = run / "07_agent" / "answers.jsonl"
        if baseline_path.exists() and baseline_path != stage / "answers.jsonl":
            baseline_by_id = {
                row["query_id"]: row for row in read_jsonl(baseline_path)
                if row.get("query_id")
            }
            if all(row["query_id"] in baseline_by_id for row in retrieval_rows):
                baseline_f1 = 100 * sum(
                    answer_f1(
                        baseline_by_id[row["query_id"]].get("answer", ""),
                        gold_by_id.get(row["query_id"], ""),
                    ) for row in retrieval_rows
                ) / max(1, len(retrieval_rows))
                summary["baseline_F1_same_queries"] = round(baseline_f1, 2)
                summary["F1_change_vs_baseline"] = round(
                    summary["F1_diagnostic"] - baseline_f1, 2
                )
        summary["F1_diagnostic_protocol"] = (
            "Same whitespace/punctuation-normalized token F1 as 08_evaluation; "
            "includes failed generations as zero. Gold answers were used only "
            "after generation, never in model prompts."
        )
    write_json(stage / "summary.json", summary)


def agent_task(graph: dict[str, Any], client: LLMClient, stage: Path,
               a: argparse.Namespace) -> Path:
    path = stage / (graph["query_id"] + ".json")
    if not a.no_resume and path.exists() and read_json(path).get("status") == "ok":
        return path
    units = [compact(x) for x in graph["units"]]
    chains = [{"node_labels": x["node_labels"], "score": x["score"],
               "edges": [{"relation": e["relation"], "unit_id": e["unit_id"],
                          "kind": e["kind"], "family": e.get("family", "other"),
                          "direction": e.get("direction", "forward"),
                          "surface_relation": e.get("surface_relation", ""),
                          "derived_from": e.get("derived_from", [])}
                         for e in x["edges"]]} for x in graph["chains"]]
    base = {"query": graph["query"], "query_plan": graph["plan"],
            "graph_diagnostics": graph.get("graph_diagnostics", {}),
            "candidate_reasoning_units": units, "candidate_chains": chains}
    reason_messages = [{"role": "system", "content": AGENT_SYSTEM},
                       {"role": "user", "content": json.dumps(base, ensure_ascii=False, indent=2)}]
    output = {"status": "error", "query_id": graph["query_id"], "query": graph["query"]}
    try:
        rr, rc, proposal = client.call_json(reason_messages, a.max_tokens)
        selected = {x["unit_id"]: x for x in units}
        proof = [selected[x] for x in proposal.get("supporting_unit_ids", []) if x in selected]
        verify_payload = {"query": graph["query"], "query_plan": graph["plan"],
                          "proposal": proposal, "proof_units": proof,
                          "candidate_chains": chains,
                          "graph_diagnostics": graph.get("graph_diagnostics", {})}
        verify_messages = [{"role": "system", "content": VERIFY_SYSTEM},
                           {"role": "user", "content": json.dumps(verify_payload, ensure_ascii=False, indent=2)}]
        vr, vc, verification = client.call_json(verify_messages, a.max_tokens)
        repair = None
        repair_messages_used = None
        repair_response = None
        repair_content = None
        final = proposal
        if not a.no_repair and verification.get("status") != "supported":
            repair_messages = [{"role": "system", "content": AGENT_SYSTEM +
                                "\nRepair the proposal or return insufficient."},
                               {"role": "user", "content": json.dumps({
                                   "query": graph["query"], "proposal": proposal,
                                   "verification": verification, "candidate_units": units,
                                   "candidate_chains": chains}, ensure_ascii=False, indent=2)}]
            repair_messages_used = repair_messages
            repair_response, repair_content, repair = client.call_json(repair_messages, a.max_tokens)
            if repair.get("answer") or repair.get("status") == "supported":
                final = repair
        # Do not silently pass through an answer that the verifier rejected.
        # Keep the original proposal and verification in the checkpoint for
        # analysis, but expose an insufficient final answer to evaluation.
        if verification.get("status") != "supported":
            if not repair or repair.get("status") != "supported":
                final = dict(final)
                final["status"] = "insufficient"
                final["confidence"] = min(float(final.get("confidence") or 0.0), 0.3)
        final, answer_formatting = ensure_self_contained_boolean_answer(
            final, graph.get("plan", {}), graph["query"],
            {"candidate_reasoning_units": units,
             "candidate_chains": chains}, client, a.max_tokens,
        )
        output.update({"status": "ok",
                       "reason": {"request_messages": reason_messages, "response": rr,
                                   "response_content": rc, "proposal": proposal},
                       "verification": {"request_messages": verify_messages, "response": vr,
                                         "response_content": vc, "result": verification},
                       "repair": {"request_messages": repair_messages_used,
                                  "response": repair_response,
                                  "response_content": repair_content,
                                  "proposal": repair} if repair is not None else None,
                       "answer_formatting": answer_formatting,
                       "final_proposal": final})
    except Exception as exc:
        output["error"] = repr(exc)
    write_json(path, output)
    return path


def agent_stage(a: argparse.Namespace, run: Path) -> None:
    if a.reasoning_mode == "hopwise_candidates":
        hopwise_agent_stage(a, run)
        return
    stage = run / "07_agent"
    graphs = read_jsonl(run / "06_graph" / "graphs.jsonl")
    client = LLMClient(a.base_url, a.model, a.api_key, a.timeout, a.retries,
                       a.temperature, a.provider, a.reasoning_effort)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = [pool.submit(agent_task, graph, client, stage, a) for graph in graphs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == len(futures):
                print("[agent] completed", done, "/", len(futures))
    answers, errors = [], []
    for path in sorted(stage.glob("q_*.json")):
        item = read_json(path)
        if item.get("status") != "ok":
            errors.append(item)
            continue
        final = item.get("final_proposal", {})
        answers.append({"query_id": item["query_id"], "query": item["query"],
                        "answer": final.get("answer", ""), "status": final.get("status", ""),
                        "supporting_unit_ids": final.get("supporting_unit_ids", []),
                        "reasoning_steps": final.get("reasoning_steps", []),
                        "confidence": final.get("confidence", 0)})
    write_jsonl(stage / "answers.jsonl", answers)
    write_jsonl(stage / "errors.jsonl", errors)
    write_json(stage / "summary.json", {"queries": len(graphs), "successful": len(answers),
                                        "errors": len(errors), "model": client.model})


def answer_norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", norm(value))).strip()


def answer_f1(pred: Any, gold: Any) -> float:
    p, g = answer_norm(pred).split(), answer_norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def ev_key(item: dict[str, Any]) -> tuple[str, str]:
    return norm(item.get("source", "")), norm(item.get("title", ""))


def ev_overlap(fact: str, quote: str) -> float:
    a, b = set(tokens(fact)), set(tokens(quote))
    return len(a & b) / max(1, len(a))


def unit_sentence_ids(unit: dict[str, Any]) -> set[str]:
    """Return source sentence IDs carried by an EnSI-RAG semantic passage."""
    document = unit.get("document", {}) if isinstance(unit.get("document"), dict) else {}
    values = unit.get("source_sentence_ids") or document.get("source_sentence_ids") or []
    if isinstance(values, str):
        values = [values]
    return {str(x) for x in values if str(x).strip()}


def context_recall_score(row: dict[str, Any], units: list[dict[str, Any]]) -> tuple[float, str]:
    """Compute HolisQA context recall from stable gold/retrieved sentence IDs.

    CausalRAG2 releases stable ``context_sentence_ids`` for every QA item.
    The semantic RESIN preparation stores the same IDs on every passage, so
    this is a deterministic implementation of the paper's "gold context
    covered by retrieved evidence" definition.  The lexical fallback is only
    for old fixed-chunk runs that predate sentence-ID propagation and is
    explicitly labeled as such in the artifacts.
    """
    gold_ids = {str(x) for x in (row.get("context_sentence_ids") or []) if str(x).strip()}
    retrieved_ids = set().union(*(unit_sentence_ids(x) for x in units)) if units else set()
    if gold_ids and retrieved_ids:
        return len(gold_ids & retrieved_ids) / len(gold_ids), "sentence_id_overlap"

    gold_ev = row.get("evidence_list", []) if isinstance(row.get("evidence_list"), list) else []
    if not gold_ev:
        return 0.0, "unavailable"
    score = sum(
        any(ev_overlap(x.get("fact", ""), u.get("evidence_quote", "")) >= .5 for u in units)
        for x in gold_ev
    ) / max(1, len(gold_ev))
    return score, "lexical_fact_overlap_fallback"


def answer_relevancy_task(item: dict[str, Any], client: JudgeClient, stage: Path,
                          resume: bool, strictness: int, max_tokens: int) -> Path:
    """Run the Ragas reverse-question generation for one answer."""
    path = stage / (item["query_id"] + ".json")
    if resume and path.exists() and read_json(path).get("status") in {"judged", "ok"}:
        return path
    prompt = causalrag2_ar_prompt(item.get("prediction", ""))
    output: dict[str, Any] = {
        "status": "error",
        "query_id": item["query_id"],
        "query": item["query"],
        "prediction": item.get("prediction", ""),
        "prompt": prompt,
        "strictness": strictness,
        "judge_provider": client.provider,
        "judge_model": client.model,
        "responses": [],
    }
    try:
        for generation_index in range(max(1, strictness)):
            response, content, parsed = client.call_json(prompt, max_tokens)
            question = str(parsed.get("question") or "").strip()
            noncommittal = parsed.get("noncommittal", 0)
            try:
                noncommittal = int(bool(int(noncommittal)))
            except (TypeError, ValueError):
                noncommittal = int(str(noncommittal).strip().lower() in {"true", "yes", "1"})
            output["responses"].append({
                "generation_index": generation_index,
                "response": response,
                "response_content": content,
                "parsed": parsed,
                "generated_question": question,
                "noncommittal": noncommittal,
            })
        output["status"] = "judged"
    except Exception as exc:
        output["error"] = repr(exc)
    write_json(path, output)
    return path


def compute_answer_relevancy(a: argparse.Namespace, stage: Path,
                             base_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute CausalRAG2/Ragas Answer Relevancy and persist every artifact."""
    ar_stage = stage / "answer_relevancy"
    client = JudgeClient(a.judge_provider, a.judge_model, a.judge_base_url,
                         a.judge_api_key, a.judge_timeout, a.judge_retries, 0.0)
    print("[ar] judge:", client.model, "provider:", client.provider,
          "strictness:", a.judge_strictness)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.judge_workers)) as pool:
        futures = [pool.submit(answer_relevancy_task, item, client, ar_stage,
                               not a.no_resume, a.judge_strictness, a.judge_max_tokens)
                   for item in base_rows]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % 100 == 0 or done == len(futures):
                print("[ar] judged", done, "/", len(futures))

    artifacts = {}
    for path in sorted(ar_stage.glob("q_*.json")):
        item = read_json(path)
        if item.get("status") == "judged":
            artifacts[item["query_id"]] = item
    if not artifacts:
        return {}

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Answer Relevancy requires sentence-transformers for the paper's "
            "all-MiniLM-L6-v2 embeddings. Install requirements.txt first."
        ) from exc

    model = SentenceTransformer(a.embedding_model)
    ordered = [artifacts[key] for key in sorted(artifacts)]
    flat_texts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for artifact in ordered:
        questions = [str(x.get("generated_question", ""))
                     for x in artifact.get("responses", [])]
        questions = [x for x in questions if x]
        start = len(flat_texts)
        flat_texts.append(str(artifact.get("query", "")))
        flat_texts.extend(questions)
        spans[artifact["query_id"]] = (start, len(questions))

    embeddings = model.encode(flat_texts, normalize_embeddings=True,
                              batch_size=64, show_progress_bar=True)
    for artifact in ordered:
        start, question_count = spans[artifact["query_id"]]
        cosine = [float(embeddings[start] @ embeddings[start + index + 1])
                  for index in range(question_count)]
        flags = [int(x.get("noncommittal", 0))
                 for x in artifact.get("responses", []) if x.get("generated_question", "")]
        all_noncommittal = bool(flags) and all(flags)
        score = (sum(cosine) / len(cosine)) if cosine else 0.0
        if all_noncommittal:
            score = 0.0
        artifact.update({
            "status": "ok",
            "generated_questions": [x.get("generated_question", "")
                                     for x in artifact.get("responses", [])],
            "noncommittal": flags,
            "all_noncommittal": all_noncommittal,
            "cosine_similarities": cosine,
            "answer_relevancy": score,
            "embedding_model": a.embedding_model,
            "metric": "Ragas AnswerRelevancy / ResponseRelevancy",
        })
        write_json(ar_stage / (artifact["query_id"] + ".json"), artifact)
        artifacts[artifact["query_id"]] = artifact
    write_json(ar_stage / "config.json", {
        "metric": "Ragas AnswerRelevancy / ResponseRelevancy",
        "prompt": "CausalRAG2 Appendix F.4 Answer Relevancy prompt",
        "judge_provider": client.provider,
        "judge_model": client.model,
        "strictness": a.judge_strictness,
        "embedding_model": a.embedding_model,
        "note": "Three identical prompt generations are averaged, and the score is zero only when all are noncommittal, matching Ragas.",
    })
    return artifacts


def evaluate_stage(a: argparse.Namespace, run: Path, qa_file: Path) -> None:
    stage = run / "08_evaluation"
    gold = load_records(qa_file)
    answers = {x["query_id"]: x for x in read_jsonl(run / "07_agent" / "answers.jsonl")}
    retrieval = {x["query_id"]: x for x in read_jsonl(run / "05_retrieval" / "retrieval.jsonl")}
    rows = []
    for i, row in enumerate(gold):
        query = str(row.get("query", ""))
        query_id = "q_" + str(i).zfill(6) + "_" + h(query)[:8]
        if query_id not in answers:
            continue
        answer = answers[query_id]
        units = [x["unit"] for x in retrieval.get(query_id, {}).get("candidates", [])]
        gold_ev = row.get("evidence_list", []) if isinstance(row.get("evidence_list"), list) else []
        gold_keys = {ev_key(x) for x in gold_ev}
        ret_keys = {ev_key(x.get("document", {})) for x in units}
        source_recall = len(gold_keys & ret_keys) / max(1, len(gold_keys))
        fact_recall = sum(any(ev_overlap(x.get("fact", ""), u.get("evidence_quote", "")) >= .5
                             for u in units) for x in gold_ev) / max(1, len(gold_ev))
        pred, ref = answer.get("answer", ""), row.get("answer", "")
        rows.append({"query_id": query_id, "query": query,
                     "domain": row.get("category", row.get("domain", "")),
                     "question_type": row.get("question_type", ""),
                     "prediction": pred, "gold_answer": ref,
                     "exact_match": answer_norm(pred) == answer_norm(ref),
                     "f1": answer_f1(pred, ref),
                     "retrieved_source_recall": source_recall,
                     "retrieved_fact_recall": fact_recall,
                     "agent_status": answer.get("status", "")})

    write_jsonl(stage / "per_query.jsonl", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["domain"] or "unknown")].append(row)

    def avg(items: list[dict[str, Any]], key: str) -> float:
        return sum(float(x.get(key, 0)) for x in items) / max(1, len(items))

    def paper_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        # This stage reports answer F1 only. RAGAS AR/CR are produced by the
        # standalone RAGAS evaluator so that 08_evaluation cannot mix the
        # paper metric with the legacy approximation implemented here.
        return {
            "count": len(items),
            "F1": round(avg(items, "f1") * 100, 2),
        }

    summary = {
        "evaluated_queries": len(rows),
        "F1": round(avg(rows, "f1") * 100, 2),
        "f1_0_to_1": avg(rows, "f1"),
        "exact_match": avg(rows, "exact_match"),
        "retrieved_source_recall_diagnostic": avg(rows, "retrieved_source_recall"),
        "retrieved_fact_recall_diagnostic": avg(rows, "retrieved_fact_recall"),
        "paper_table_by_domain": {
            key: paper_metrics(items) for key, items in sorted(groups.items())
        },
        "evaluation_protocol": {
            "f1": "Whitespace/punctuation-normalized token F1; reported as percent.",
            "ragas_ar_cr": "Computed separately by the standalone RAGAS evaluator; not mixed into 08_evaluation.",
        },
        "warning": "Use the standalone RAGAS evaluator for the comparable AR/CR columns.",
    }
    write_json(stage / "summary.json", summary)


def main() -> int:
    a = args()
    generation_options = (a.generation_style != "concise"
                          or a.generation_domain is not None
                          or a.generation_limit is not None
                          or a.generation_output_dir is not None)
    if generation_options and a.stage != "agent":
        raise ValueError("Generation options require --stage agent")
    if generation_options and a.reasoning_mode != "hopwise_candidates":
        raise ValueError("Generation options require --reasoning_mode hopwise_candidates")
    if (a.generation_style in {"holisqa", "qasc", "msmarco"} or a.generation_domain is not None
            or a.generation_limit is not None) and a.generation_output_dir is None:
        raise ValueError("Use --generation_output_dir to preserve the original agent run")
    corpus, qa = resolve_files(a)
    run = a.run_dir or DEFAULT_RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    if (a.generation_output_dir is not None
            and a.generation_output_dir.resolve() == (run / "07_agent").resolve()):
        raise ValueError("--generation_output_dir must differ from run_dir/07_agent")
    run.mkdir(parents=True, exist_ok=True)
    preprocess_reused = False
    if a.reuse_preprocess_run is not None:
        source_run = a.reuse_preprocess_run.resolve()
        target_run = run.resolve()
        if source_run != target_run:
            for folder in ("01_extraction", "02_normalization", "03_index"):
                source_folder = source_run / folder
                if not source_folder.exists():
                    raise FileNotFoundError(
                        f"Cannot reuse preprocessing: missing {source_folder}"
                    )
                shutil.copytree(source_folder, run / folder, dirs_exist_ok=True)
        preprocess_reused = True
    config = {
        "stage": a.stage, "corpus": str(corpus), "qa": str(qa),
        "provider": a.provider, "model": a.model, "base_url": a.base_url,
        "workers": a.workers, "hop_workers": a.hop_workers,
        "reasoning_effort": a.reasoning_effort,
        "max_docs": a.max_docs, "max_queries": a.max_queries,
        "top_k": a.top_k, "top_chains": a.top_chains, "max_hops": a.max_hops,
        "reasoning_mode": a.reasoning_mode, "hop_top_k": a.hop_top_k,
        "reasoning_embedding_model": a.reasoning_embedding_model,
        "reasoning_embedding_device": a.reasoning_embedding_device,
        "reasoning_index_version": "state_transition_v1",
        "oracle_gold_docs": a.oracle_gold_docs,
        "resume": not a.no_resume, "skip_extract": a.skip_extract,
        "reuse_preprocess_run": str(a.reuse_preprocess_run) if a.reuse_preprocess_run else None,
        "preprocess_reused": preprocess_reused,
        "evaluation_judge_provider": a.judge_provider,
        "evaluation_judge_model": a.judge_model,
        "evaluation_embedding_model": a.embedding_model,
        "evaluation_strictness": a.judge_strictness}
    if a.generation_output_dir is not None:
        write_json(a.generation_output_dir / "generation_config.json", {
            **config,
            "source_run_dir": str(run),
            "generation_style": a.generation_style,
            "generation_domain": a.generation_domain,
            "generation_limit": a.generation_limit,
            "generation_signature": generation_signature(a),
        })
    else:
        write_json(run / "config.json", config)
    if a.stage in {"all", "extract"} and not a.skip_extract and not preprocess_reused:
        run_timed_stage(run, "01_extraction", extraction_stage, a, run, corpus)
    if a.stage in {"all", "normalize"} and not preprocess_reused:
        run_timed_stage(run, "02_normalization", normalize_stage, run)
    if a.stage in {"all", "index"} and not preprocess_reused:
        run_timed_stage(run, "03_index", index_stage, run)
    if a.stage in {"all", "plan"}:
        run_timed_stage(run, "04_query_plans", plan_stage, a, run, qa)
    if a.stage in {"all", "retrieve"}:
        run_timed_stage(run, "05_retrieval", retrieval_stage, a, run)
    if a.stage in {"all", "graph"} and a.reasoning_mode == "graph":
        run_timed_stage(run, "06_graph", graph_stage, a, run)
    if a.stage in {"all", "agent"}:
        if a.generation_output_dir is not None:
            started = time.perf_counter()
            agent_stage(a, run)
            write_json(a.generation_output_dir / "timing.json", {
                "wall_time_seconds": round(time.perf_counter() - started, 6)
            })
        else:
            run_timed_stage(run, "07_agent", agent_stage, a, run)
    if a.stage in {"all", "evaluate"}:
        run_timed_stage(run, "08_evaluation", evaluate_stage, a, run, qa)
    print("[done] artifacts:", run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
