#!/usr/bin/env python3
"""Recover legacy HolisQA passage checkpoints after cross-domain ID collisions.

Official HolisQA restarts article IDs at A000001 in every domain.  Older
versions of the semantic preparer used that raw ID as the checkpoint filename,
so a full five-domain run retained only one of five completed calls for each
ID.  This migration identifies the domain represented by every surviving
checkpoint from its recorded article title/source text, writes a correctly
scoped checkpoint, and moves the ambiguous legacy file to a recoverable
archive.  Missing domain/article pairs are then generated normally by resume.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DOMAINS = "biology,business,computer_science,medicine,psychology"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def scoped_id(domain: str, raw_id: str) -> str:
    return f"{domain}__{raw_id}"


def unique_archive_path(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 2
    while True:
        candidate = directory / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def load_catalog(source_dir: Path, domains: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_raw_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for domain in domains:
        domain_dir = source_dir / domain
        articles = {
            str(row["article_id"]): row
            for row in read_jsonl(domain_dir / "articles.jsonl")
        }
        sentences: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(domain_dir / "sentences.jsonl"):
            sentences[str(row["article_id"])].append(row)
        for article_id, article in articles.items():
            rows = sorted(sentences.get(article_id, []), key=lambda x: int(x.get("sent_idx", 0)))
            by_raw_id[article_id].append({
                "domain": domain,
                "article_id": article_id,
                "title": str(article.get("title", "")),
                "sentences": [str(row.get("text", "")) for row in rows if str(row.get("text", "")).strip()],
            })
    return dict(by_raw_id)


def recorded_prompt(checkpoint: dict[str, Any]) -> str:
    calls = checkpoint.get("calls", [])
    if not isinstance(calls, list):
        return ""
    return "\n".join(
        str(call.get("prompt", ""))
        for call in calls
        if isinstance(call, dict)
    )


def candidate_score(prompt: str, checkpoint: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, list[str]]:
    prompt_norm = norm(prompt)
    score = 0
    evidence: list[str] = []

    title = norm(candidate.get("title"))
    if title and title in prompt_norm:
        score += 1000 + min(len(title), 250)
        evidence.append("title")

    sentences = candidate.get("sentences", [])
    if isinstance(sentences, list) and sentences:
        sample_indices = sorted({0, len(sentences) // 2, len(sentences) - 1})
        for index in sample_indices:
            sentence = norm(sentences[index])
            # A long prefix is discriminative while tolerating minor trailing
            # punctuation/whitespace differences in the recorded prompt.
            sample = sentence[:120]
            if len(sample) >= 24 and sample in prompt_norm:
                score += 150
                evidence.append(f"sentence_{index}")

        expected_units = len(sentences)
        observed_units = checkpoint.get("diagnostics", {}).get("units")
        try:
            if int(observed_units) == expected_units:
                score += 5
                evidence.append("unit_count")
        except (TypeError, ValueError):
            pass

    return score, evidence


def infer_domain(raw_id: str, checkpoint: dict[str, Any],
                 catalog: dict[str, list[dict[str, Any]]]) -> tuple[str | None, dict[str, Any]]:
    candidates = catalog.get(raw_id, [])
    if not candidates:
        return None, {"reason": "article_id_not_found", "scores": []}
    if len(candidates) == 1:
        return str(candidates[0]["domain"]), {
            "reason": "single_candidate", "scores": [{"domain": candidates[0]["domain"], "score": 1}]
        }

    prompt = recorded_prompt(checkpoint)
    scores = []
    for candidate in candidates:
        score, evidence = candidate_score(prompt, checkpoint, candidate)
        scores.append({"domain": candidate["domain"], "score": score, "evidence": evidence})
    scores.sort(key=lambda row: (-int(row["score"]), str(row["domain"])))
    best = scores[0]
    second_score = int(scores[1]["score"]) if len(scores) > 1 else -1
    if int(best["score"]) >= 100 and int(best["score"]) > second_score:
        return str(best["domain"]), {"reason": "content_match", "scores": scores}
    return None, {"reason": "ambiguous_or_no_content_match", "scores": scores}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--bundle_dir", type=Path, required=True)
    parser.add_argument("--domains", default=DEFAULT_DOMAINS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    raw_dir = args.bundle_dir / "01_passage_construction" / "raw"
    if not raw_dir.exists():
        print(f"[checkpoint-migration] no raw checkpoint directory: {raw_dir}")
        return 0

    legacy_paths = sorted(
        path for path in raw_dir.glob("*.json")
        if re.fullmatch(r"A\d+\.json", path.name)
    )
    if not legacy_paths:
        print("[checkpoint-migration] no colliding legacy Axxxxxx.json checkpoints found")
        return 0

    catalog = load_catalog(args.source_dir, domains)
    archive_dir = args.bundle_dir / "01_passage_construction" / "legacy_colliding_raw"
    archive_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "legacy_files": len(legacy_paths),
        "migrated": 0,
        "already_present": 0,
        "unmatched": 0,
        "records": [],
    }

    for path in legacy_paths:
        raw_id = path.stem
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            domain, inference = infer_domain(raw_id, checkpoint, catalog)
        except Exception as exc:
            checkpoint = None
            domain = None
            inference = {"reason": "invalid_checkpoint", "error": repr(exc), "scores": []}

        record: dict[str, Any] = {
            "legacy_file": path.name,
            "raw_article_id": raw_id,
            "inference": inference,
        }
        if domain is not None and isinstance(checkpoint, dict):
            canonical_id = scoped_id(domain, raw_id)
            target = raw_dir / f"{canonical_id}.json"
            record.update({"domain": domain, "canonical_document_id": canonical_id,
                           "target_file": target.name})
            if target.exists():
                report["already_present"] += 1
                record["status"] = "already_present"
            else:
                migrated = dict(checkpoint)
                migrated["document_id"] = canonical_id
                migrated["article_id"] = canonical_id
                migrated["original_document_id"] = raw_id
                migrated["original_article_id"] = raw_id
                migrated["category"] = domain
                migrated["checkpoint_migration"] = {
                    "from": path.name,
                    "reason": inference.get("reason"),
                    "note": "Recovered from legacy cross-domain HolisQA checkpoint collision.",
                }
                write_json(target, migrated)
                report["migrated"] += 1
                record["status"] = "migrated"
        else:
            report["unmatched"] += 1
            record["status"] = "unmatched_will_rerun"

        archive_path = unique_archive_path(archive_dir, path.name)
        shutil.move(str(path), str(archive_path))
        record["archived_as"] = archive_path.name
        report["records"].append(record)

    report_path = args.bundle_dir / "01_passage_construction" / "checkpoint_migration.json"
    write_json(report_path, report)
    print(
        "[checkpoint-migration] "
        f"legacy={report['legacy_files']} migrated={report['migrated']} "
        f"already_present={report['already_present']} unmatched={report['unmatched']}"
    )
    print(f"[checkpoint-migration] report: {report_path}")
    print(f"[checkpoint-migration] recoverable legacy archive: {archive_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
