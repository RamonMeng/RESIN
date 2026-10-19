#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh, full-corpus HolisQA experiment.
#
# This script deliberately does not import any earlier passage, extraction,
# planning, retrieval, generation, or evaluation checkpoints.  On its first
# invocation use START_MODE=fresh with new DATA_DIR/RUN_DIR paths.  If the job
# is interrupted, rerun with the same paths and START_MODE=resume.
#
# Configuration: all five official domains and all official queries,
# GPT-5-nano/minimal, hop-wise candidate reasoning, top-10 per hop, four hops,
# and Ragas AR + source-passage CR (the protocol that produced CR=68.70 in the
# 500-query context-representation comparison).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOURCE_DIR="${SOURCE_DIR:-$SCRIPT_DIR/data/holisqa}"
DATA_DIR="${DATA_DIR:-data/holisqa_resin_gpt5nano_all_queries_v1}"
RUN_DIR="${RUN_DIR:-data/resin_runs/holisqa_gpt5nano_fast_hopwise_k10_all_queries_v1}"
START_MODE="${START_MODE:-fresh}"
DOMAINS="${DOMAINS:-biology,business,computer_science,medicine,psychology}"
PREP_WORKERS="${PREP_WORKERS:-8}"
RUN_WORKERS="${RUN_WORKERS:-8}"
HOP_WORKERS="${HOP_WORKERS:-2}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
HOP_TOP_K=10
MAX_HOPS=4
MODEL="gpt-5-nano"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.5-flash-lite}"
LOG_DIR="${LOG_DIR:-data/holisqa_full_logs}"
LOG_FILE="$LOG_DIR/holisqa_all_queries_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

fail() {
  echo "[FATAL] $*" >&2
  echo "[FATAL] log: $LOG_FILE" >&2
  exit 1
}
trap 'echo "[FATAL] command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }

count_jsonl() {
  python3 -c 'import pathlib,sys; print(sum(bool(x.strip()) for x in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()))' "$1"
}

summary_field() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get(sys.argv[2],0))' "$1" "$2"
}

official_count() {
  local filename="$1"
  python3 - "$SOURCE_DIR" "$DOMAINS" "$filename" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
domains = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
name = sys.argv[3]
total = 0
for domain in domains:
    path = root / domain / name
    if not path.exists():
        raise SystemExit(f"missing official file: {path}")
    total += sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
print(total)
PY
}

validate_bundle() {
  require_file "$DATA_DIR/corpus/corpus.json"
  require_file "$DATA_DIR/qa/holisqa.jsonl"
  python3 - "$DATA_DIR/corpus/corpus.json" "$DATA_DIR/qa/holisqa.jsonl" "$OFFICIAL_QA_COUNT" <<'PY'
import json, pathlib, sys
corpus_path, qa_path = map(pathlib.Path, sys.argv[1:3])
expected = int(sys.argv[3])
corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
if isinstance(corpus, dict):
    corpus = corpus.get("data", corpus.get("corpus", []))
queries = sum(bool(x.strip()) for x in qa_path.read_text(encoding="utf-8").splitlines())
if not isinstance(corpus, list) or not corpus:
    raise SystemExit("prepared semantic corpus is empty")
if queries != expected:
    raise SystemExit(f"prepared queries {queries} != official queries {expected}")
print(f"[check] semantic passages={len(corpus)}, queries={queries}/{expected}")
PY
}

validate_passage_construction() {
  python3 - "$DATA_DIR/01_passage_construction/raw" "$SOURCE_DIR" "$DOMAINS" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
domains = [x.strip() for x in sys.argv[3].split(",") if x.strip()]
expected_ids = set()
for domain in domains:
    path = source / domain / "articles.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected_ids.add(f"{domain}__{json.loads(line)['article_id']}")
files = sorted(root.glob("*.json"))
bad = []
actual_ids = set()
for path in files:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        document_id = str(row.get("document_id", ""))
        actual_ids.add(document_id)
        if (path.stem != document_id
                or row.get("status") != "ok"
                or int(row.get("diagnostics", {}).get("passages", 0)) <= 0):
            bad.append(path.name)
    except Exception:
        bad.append(path.name)
missing = sorted(expected_ids - actual_ids)
unexpected = sorted(actual_ids - expected_ids)
print(f"[check] passage documents={len(actual_ids)}/{len(expected_ids)}, "
      f"failed={len(bad)}, missing={len(missing)}, unexpected={len(unexpected)}")
if bad:
    print("[check] failed examples:", ", ".join(bad[:10]))
if missing:
    print("[check] missing examples:", ", ".join(missing[:10]))
if unexpected:
    print("[check] unexpected examples:", ", ".join(unexpected[:10]))
if len(files) != len(expected_ids) or actual_ids != expected_ids or bad:
    raise SystemExit(1)
PY
}

run_stage() {
  local stage="$1"
  local max_tokens="$2"
  python3 -u run_resin_fast.py \
    --stage "$stage" \
    --provider openai \
    --model "$MODEL" \
    --base_url https://api.openai.com/v1 \
    --data_dir "$DATA_DIR" \
    --qa_file "$DATA_DIR/qa/holisqa.jsonl" \
    --run_dir "$RUN_DIR" \
    --workers "$RUN_WORKERS" \
    --hop_workers "$HOP_WORKERS" \
    --reasoning_effort minimal \
    --reasoning_mode hopwise_candidates \
    --reasoning_embedding_model all-MiniLM-L6-v2 \
    --reasoning_embedding_device cpu \
    --hop_top_k "$HOP_TOP_K" \
    --max_hops "$MAX_HOPS" \
    --max_tokens "$max_tokens"
}

for file in prepare_holisqa_resin_semantic.py migrate_holisqa_passage_checkpoints.py \
            run_resin_fast.py \
            evaluate_ar_cr.py summarize_ragas_by_category.py summarize_resin_costs.py; do
  require_file "$file"
done
[[ -n "${OPENAI_API_KEY:-}" ]] || fail "OPENAI_API_KEY is not set"
[[ -n "${GEMINI_API_KEY:-}" ]] || fail "GEMINI_API_KEY is not set"
[[ -d "$SOURCE_DIR" ]] || fail "HolisQA source directory not found: $SOURCE_DIR"
[[ "$START_MODE" == "fresh" || "$START_MODE" == "resume" ]] || fail "START_MODE must be fresh or resume"

if [[ "$START_MODE" == "fresh" ]]; then
  [[ ! -e "$DATA_DIR" ]] || fail "fresh DATA_DIR already exists: $DATA_DIR (choose a new path or use START_MODE=resume)"
  [[ ! -e "$RUN_DIR" ]] || fail "fresh RUN_DIR already exists: $RUN_DIR (choose a new path or use START_MODE=resume)"
fi

OFFICIAL_QA_COUNT="$(official_count qa.jsonl)"
OFFICIAL_ARTICLE_COUNT="$(official_count articles.jsonl)"

echo "[start] complete HolisQA run from the full official corpus"
echo "[config] mode=$START_MODE"
echo "[config] domains=$DOMAINS"
echo "[config] official queries=$OFFICIAL_QA_COUNT"
echo "[config] official articles=$OFFICIAL_ARTICLE_COUNT"
echo "[config] data=$DATA_DIR"
echo "[config] run=$RUN_DIR"
echo "[config] model=$MODEL; reasoning_effort=minimal"
echo "[config] reasoning=hopwise_candidates; hop_top_k=$HOP_TOP_K; max_hops=$MAX_HOPS"
echo "[config] Ragas judge=$JUDGE_MODEL; CR context=source_passage"
echo "[config] log=$LOG_FILE"

echo "[0/9] repairing any legacy cross-domain passage checkpoints"
python3 -u migrate_holisqa_passage_checkpoints.py \
  --source_dir "$SOURCE_DIR" \
  --bundle_dir "$DATA_DIR" \
  --domains "$DOMAINS"

echo "[1/9] constructing semantic passages for every official article"
python3 -u prepare_holisqa_resin_semantic.py \
  --source_dir "$SOURCE_DIR" \
  --out_dir "$DATA_DIR" \
  --domains "$DOMAINS" \
  --model "$MODEL" \
  --provider openai \
  --workers "$PREP_WORKERS" \
  --max_tokens 10000 \
  --reasoning_effort minimal

validate_bundle
for retry in 1 2 3; do
  validate_passage_construction && break
  echo "[retry] passage construction retry $retry/3"
  python3 -u prepare_holisqa_resin_semantic.py \
    --source_dir "$SOURCE_DIR" \
    --out_dir "$DATA_DIR" \
    --domains "$DOMAINS" \
    --model "$MODEL" \
    --provider openai \
    --workers "$PREP_WORKERS" \
    --max_tokens 10000 \
    --reasoning_effort minimal
done
validate_passage_construction || fail "passage construction remains incomplete"

QA_COUNT="$(count_jsonl "$DATA_DIR/qa/holisqa.jsonl")"
[[ "$QA_COUNT" -eq "$OFFICIAL_QA_COUNT" ]] || fail "query count mismatch: $QA_COUNT/$OFFICIAL_QA_COUNT"

echo "[2/9] extracting typed reasoning units"
run_stage extract 4000
for retry in 1 2 3; do
  errors="$(summary_field "$RUN_DIR/01_extraction/summary.json" errors)"
  [[ "$errors" -eq 0 ]] && break
  echo "[retry] extraction has $errors errors; retry $retry/3"
  run_stage extract 4000
done
[[ "$(summary_field "$RUN_DIR/01_extraction/summary.json" errors)" -eq 0 ]] || fail "extraction remains incomplete"

echo "[3/9] normalization and index"
run_stage normalize 1800
run_stage index 1800

echo "[4/9] fast query decomposition"
run_stage plan 1200
for retry in 1 2 3; do
  successful="$(summary_field "$RUN_DIR/04_query_plans/summary.json" successful)"
  [[ "$successful" -eq "$QA_COUNT" ]] && break
  echo "[retry] planning completed $successful/$QA_COUNT; retry $retry/3"
  run_stage plan 1200
done
[[ "$(summary_field "$RUN_DIR/04_query_plans/summary.json" successful)" -eq "$QA_COUNT" ]] || fail "planning remains incomplete"

echo "[5/9] hop-wise top-$HOP_TOP_K retrieval"
run_stage retrieve 1800
[[ "$(count_jsonl "$RUN_DIR/05_retrieval/retrieval.jsonl")" -eq "$QA_COUNT" ]] || fail "retrieval remains incomplete"

echo "[6/9] fast hop and final-answer generation"
run_stage agent 1800
for retry in 1 2 3; do
  successful="$(summary_field "$RUN_DIR/07_agent/summary.json" successful)"
  [[ "$successful" -eq "$QA_COUNT" ]] && break
  echo "[retry] agent completed $successful/$QA_COUNT; retry $retry/3"
  run_stage agent 1800
done
[[ "$(summary_field "$RUN_DIR/07_agent/summary.json" successful)" -eq "$QA_COUNT" ]] || fail "agent remains incomplete"

echo "[7/9] Ragas AR and source-passage CR"
EVAL_STARTED="$(python3 -c 'import time; print(time.time())')"
for attempt in 1 2 3 4; do
  python3 -u evaluate_ar_cr.py \
    --run_dir "$RUN_DIR" \
    --qa_file "$DATA_DIR/qa/holisqa.jsonl" \
    --corpus_file "$DATA_DIR/corpus/corpus.json" \
    --output_dir "$RUN_DIR/09_ar_cr" \
    --judge_provider gemini \
    --judge_model "$JUDGE_MODEL" \
    --base_url https://generativelanguage.googleapis.com/v1beta/openai \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --strictness 3 \
    --max_contexts $((MAX_HOPS * HOP_TOP_K)) \
    --cr_context_mode source_passage \
    --workers "$EVAL_WORKERS"
  successful="$(summary_field "$RUN_DIR/09_ar_cr/summary.json" successful_ragas_queries)"
  [[ "$successful" -eq "$QA_COUNT" ]] && break
  echo "[retry] Ragas completed $successful/$QA_COUNT; retry $attempt/4"
done
[[ "$(summary_field "$RUN_DIR/09_ar_cr/summary.json" successful_ragas_queries)" -eq "$QA_COUNT" ]] || fail "Ragas remains incomplete"
EVAL_FINISHED="$(python3 -c 'import time; print(time.time())')"
EVAL_SECONDS="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$EVAL_STARTED" "$EVAL_FINISHED")"

echo "[8/9] aggregate HolisQA AR/CR by domain"
python3 -u summarize_ragas_by_category.py \
  --qa_file "$DATA_DIR/qa/holisqa.jsonl" \
  --evaluation_dir "$RUN_DIR/09_ar_cr"

echo "[9/9] preprocessing/online token and latency report"
python3 -u summarize_resin_costs.py \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --qa_file "$DATA_DIR/qa/holisqa.jsonl" \
  --evaluation_wall_seconds "$EVAL_SECONDS"

echo "[done] all official HolisQA queries: $QA_COUNT"
echo "[done] Ragas AR/source-passage CR: $RUN_DIR/09_ar_cr/summary.json"
echo "[done] domain summary: $RUN_DIR/09_ar_cr/summary_by_category.json"
echo "[done] token/latency report: $RUN_DIR/10_cost/summary.json"
echo "[done] log: $LOG_FILE"
