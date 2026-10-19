#!/usr/bin/env bash
# Run configurable-size experiments for QASC and MS MARCO.
# Set TWOQA_LIMIT to the desired number of examples per benchmark.
# Query plans may be seeded from compatible prior runs; retrieval and
# generation are always run against the collection built for the chosen size.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWOQA_PROJECT_ROOT=${TWOQA_PROJECT_ROOT:-$SCRIPT_DIR}
TWOQA_PYTHON=${TWOQA_PYTHON:-python3}
TWOQA_LIMIT=${TWOQA_LIMIT:-500}
TWOQA_REUSED_PLAN_COUNT=${TWOQA_REUSED_PLAN_COUNT:-100}
TWOQA_WORKERS=${TWOQA_WORKERS:-8}
TWOQA_HOP_WORKERS=${TWOQA_HOP_WORKERS:-2}
TWOQA_RAGAS_WORKERS=${TWOQA_RAGAS_WORKERS:-4}
TWOQA_STAGE_RETRIES=${TWOQA_STAGE_RETRIES:-4}
TWOQA_RAGAS_PASSES=${TWOQA_RAGAS_PASSES:-2}
TWOQA_HOP_TOP_K=${TWOQA_HOP_TOP_K:-10}
TWOQA_MAX_HOPS=${TWOQA_MAX_HOPS:-4}
TWOQA_REASONING_DEVICE=${TWOQA_REASONING_DEVICE:-cpu}

ROOT=$TWOQA_PROJECT_ROOT
PY=$TWOQA_PYTHON
RUNS_DIR=$ROOT/data/resin_runs
LOG_DIR=$ROOT/data/qasc_msmarco${TWOQA_LIMIT}_logs
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG=$LOG_DIR/qasc_msmarco${TWOQA_LIMIT}_$STAMP.log

CORE=$ROOT/run_resin_fast.py
RAGAS=$ROOT/evaluate_ar_cr.py
SEEDER=$ROOT/seed_query_plan_checkpoints.py
COST=$ROOT/summarize_resin_costs.py
COLLECTOR=$ROOT/summarize_two_qa.py

QASC_DATA=$ROOT/data/qasc_resin_fact${TWOQA_LIMIT}
QASC_RUN=$RUNS_DIR/qasc_fact${TWOQA_LIMIT}_gpt5nano_k${TWOQA_HOP_TOP_K}
QASC_QA=$QASC_DATA/qa/qasc.jsonl
QASC_CORPUS=$QASC_DATA/corpus/corpus.json

MSMARCO_DATA=$ROOT/data/msmarco_v11_resin_passage${TWOQA_LIMIT}
MSMARCO_RUN=$RUNS_DIR/msmarco_v11_passage${TWOQA_LIMIT}_gpt5nano_k${TWOQA_HOP_TOP_K}
MSMARCO_QA=$MSMARCO_DATA/qa/msmarco.jsonl
MSMARCO_CORPUS=$MSMARCO_DATA/corpus/corpus.json

TWOQA_QASC100_RUN=${TWOQA_QASC100_RUN:-$RUNS_DIR/qasc_fact100_gpt5nano_k10}
TWOQA_MSMARCO100_RUN=${TWOQA_MSMARCO100_RUN:-$RUNS_DIR/msmarco_v11_passage100_gpt5nano_k10}

mkdir -p "$LOG_DIR" "$RUNS_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[config] project root: $ROOT"
echo "[config] examples per benchmark: $TWOQA_LIMIT"
echo "[config] benchmarks: QASC, MS MARCO"
echo "[config] hop top-k: $TWOQA_HOP_TOP_K"
echo "[config] log: $MASTER_LOG"

: "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running.}"
: "${GEMINI_API_KEY:?Export GEMINI_API_KEY before running.}"

for required in "$CORE" "$RAGAS" "$SEEDER" "$COST" "$COLLECTOR" \
  "$ROOT/prepare_qasc_fact_units.py" \
  "$ROOT/prepare_msmarco_passage_units.py"; do
  if [[ ! -f "$required" ]]; then
    echo "[FATAL] missing required file: $required" >&2
    exit 1
  fi
done

cd "$ROOT"

json_value() {
  local path=$1
  local key=$2
  "$PY" - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    print("")
    raise SystemExit(0)
value = json.loads(path.read_text(encoding="utf-8"))
for part in key.split("."):
    value = value.get(part) if isinstance(value, dict) else None
print("" if value is None else value)
PY
}

qa_count() {
  local path=$1
  awk 'NF {count += 1} END {print count + 0}' "$path"
}

dataset_ready() {
  local qa_file=$1
  local corpus_file=$2
  local run_dir=$3
  [[ -s "$qa_file" && -s "$corpus_file" && \
     -s "$run_dir/01_extraction/units.jsonl" ]] || return 1
  [[ "$(qa_count "$qa_file")" == "$TWOQA_LIMIT" ]]
}

same_line_count() {
  local first=$1
  local second=$2
  [[ -s "$first" && -s "$second" ]] || return 1
  [[ "$(qa_count "$first")" == "$(qa_count "$second")" ]]
}

summary_complete() {
  local summary=$1
  local expected=$2
  [[ "$(json_value "$summary" successful)" == "$expected" ]]
}

run_command_with_retries() {
  local label=$1
  local attempts=$2
  shift 2
  local attempt
  for ((attempt=1; attempt<=attempts; attempt++)); do
    echo "[$label] attempt $attempt/$attempts"
    if "$@"; then
      return 0
    fi
    echo "[warning] $label attempt $attempt failed" >&2
  done
  return 1
}

run_resumable_stage() {
  local label=$1
  local summary=$2
  local expected=$3
  shift 3
  local attempt
  if summary_complete "$summary" "$expected"; then
    echo "[$label] already complete ($expected/$expected); skipping"
    return 0
  fi
  for ((attempt=1; attempt<=TWOQA_STAGE_RETRIES; attempt++)); do
    echo "[$label] pass $attempt/$TWOQA_STAGE_RETRIES"
    "$@" || true
    local successful
    successful=$(json_value "$summary" successful)
    echo "[$label] successful: ${successful:-0}/$expected"
    if [[ "$successful" == "$expected" ]]; then
      return 0
    fi
  done
  echo "[ERROR] $label remains incomplete after retries" >&2
  return 1
}

prepare_qasc() {
  if dataset_ready "$QASC_QA" "$QASC_CORPUS" "$QASC_RUN"; then
    echo "[QASC] prepared $TWOQA_LIMIT-query data already exists; skipping download"
    return 0
  fi
  run_command_with_retries "QASC preparation" 3 \
    "$PY" "$ROOT/prepare_qasc_fact_units.py" \
      --from_huggingface \
      --hf_dataset allenai/qasc \
      --hf_split validation \
      --max_queries "$TWOQA_LIMIT" \
      --data_dir "$QASC_DATA" \
      --run_dir "$QASC_RUN"
}

prepare_msmarco() {
  if dataset_ready "$MSMARCO_QA" "$MSMARCO_CORPUS" "$MSMARCO_RUN"; then
    echo "[MS MARCO] prepared $TWOQA_LIMIT-query data already exists; skipping download"
    return 0
  fi
  run_command_with_retries "MS MARCO preparation" 3 \
    "$PY" "$ROOT/prepare_msmarco_passage_units.py" \
      --from_huggingface \
      --hf_dataset microsoft/ms_marco \
      --hf_config v1.1 \
      --hf_split validation \
      --max_queries "$TWOQA_LIMIT" \
      --data_dir "$MSMARCO_DATA" \
      --run_dir "$MSMARCO_RUN"
}

seed_plans() {
  local label=$1
  local source_run=$2
  local target_run=$3
  local qa_file=$4
  if summary_complete "$target_run/04_query_plans/summary.json" "$TWOQA_LIMIT"; then
    return 0
  fi
  if [[ ! -d "$source_run/04_query_plans" ]]; then
    echo "[$label] prior 100-query planner run not found: $source_run"
    echo "[$label] all $TWOQA_LIMIT plans will be generated"
    return 0
  fi
  "$PY" "$SEEDER" \
    --source_run "$source_run" \
    --target_run "$target_run" \
    --target_qa_file "$qa_file" \
    --count "$TWOQA_REUSED_PLAN_COUNT"
}

run_dataset() {
  local label=$1
  local data_dir=$2
  local run_dir=$3
  local qa_file=$4
  local corpus_file=$5
  local old_run=$6

  echo
  echo "============================================================"
  echo "[dataset] $label"
  echo "[dataset] data: $data_dir"
  echo "[dataset] run:  $run_dir"
  echo "============================================================"

  local count
  count=$(qa_count "$qa_file")
  if [[ "$count" != "$TWOQA_LIMIT" ]]; then
    echo "[ERROR] $label expected $TWOQA_LIMIT QA rows, found $count" >&2
    return 1
  fi

  if ! same_line_count "$run_dir/01_extraction/units.jsonl" \
      "$run_dir/02_normalization/units.jsonl"; then
    "$PY" "$CORE" \
      --stage normalize \
      --data_dir "$data_dir" \
      --qa_file "$qa_file" \
      --corpus_file "$corpus_file" \
      --run_dir "$run_dir" \
      --reasoning_mode hopwise_candidates || return 1
  else
    echo "[$label normalize] checkpoint is complete; skipping"
  fi

  if ! same_line_count "$run_dir/02_normalization/units.jsonl" \
      "$run_dir/03_index/unit_store.jsonl"; then
    "$PY" "$CORE" \
      --stage index \
      --data_dir "$data_dir" \
      --qa_file "$qa_file" \
      --corpus_file "$corpus_file" \
      --run_dir "$run_dir" \
      --reasoning_mode hopwise_candidates || return 1
  else
    echo "[$label index] checkpoint is complete; skipping"
  fi

  seed_plans "$label" "$old_run" "$run_dir" "$qa_file" || return 1

  run_resumable_stage "$label planning" \
    "$run_dir/04_query_plans/summary.json" "$TWOQA_LIMIT" \
    "$PY" "$CORE" \
      --stage plan \
      --provider openai \
      --model gpt-5-nano \
      --base_url https://api.openai.com/v1 \
      --data_dir "$data_dir" \
      --qa_file "$qa_file" \
      --corpus_file "$corpus_file" \
      --run_dir "$run_dir" \
      --reasoning_mode hopwise_candidates \
      --max_hops "$TWOQA_MAX_HOPS" \
      --hop_top_k "$TWOQA_HOP_TOP_K" \
      --workers "$TWOQA_WORKERS" \
      --temperature 0.0 \
      --reasoning_effort minimal \
      --max_tokens 1200 || return 1

  "$PY" "$CORE" \
    --stage retrieve \
    --data_dir "$data_dir" \
    --qa_file "$qa_file" \
    --corpus_file "$corpus_file" \
    --run_dir "$run_dir" \
    --reasoning_mode hopwise_candidates \
    --reasoning_embedding_model all-MiniLM-L6-v2 \
    --reasoning_embedding_device "$TWOQA_REASONING_DEVICE" \
    --hop_top_k "$TWOQA_HOP_TOP_K" \
    --max_hops "$TWOQA_MAX_HOPS" || return 1

  if [[ "$(json_value "$run_dir/05_retrieval/summary.json" queries)" != \
        "$TWOQA_LIMIT" ]]; then
    echo "[ERROR] $label retrieval did not produce $TWOQA_LIMIT queries" >&2
    return 1
  fi

  run_resumable_stage "$label agent" \
    "$run_dir/07_agent/summary.json" "$TWOQA_LIMIT" \
    "$PY" "$CORE" \
      --stage agent \
      --provider openai \
      --model gpt-5-nano \
      --base_url https://api.openai.com/v1 \
      --data_dir "$data_dir" \
      --qa_file "$qa_file" \
      --corpus_file "$corpus_file" \
      --run_dir "$run_dir" \
      --reasoning_mode hopwise_candidates \
      --max_hops "$TWOQA_MAX_HOPS" \
      --hop_top_k "$TWOQA_HOP_TOP_K" \
      --workers "$TWOQA_WORKERS" \
      --hop_workers "$TWOQA_HOP_WORKERS" \
      --temperature 0.0 \
      --reasoning_effort minimal \
      --max_tokens 1800 || return 1

  local ragas_pass
  for ((ragas_pass=1; ragas_pass<=TWOQA_RAGAS_PASSES; ragas_pass++)); do
    echo "[$label Ragas] pass $ragas_pass/$TWOQA_RAGAS_PASSES"
    "$PY" "$RAGAS" \
      --run_dir "$run_dir" \
      --qa_file "$qa_file" \
      --corpus_file "$corpus_file" \
      --judge_provider gemini \
      --judge_model gemini-3.5-flash-lite \
      --base_url https://generativelanguage.googleapis.com/v1beta/openai \
      --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
      --strictness 3 \
      --max_contexts $((TWOQA_MAX_HOPS * TWOQA_HOP_TOP_K)) \
      --cr_context_mode source_passage \
      --workers "$TWOQA_RAGAS_WORKERS" || return 1
  done

  "$PY" "$COST" \
    --run_dir "$run_dir" \
    --data_dir "$data_dir" \
    --qa_file "$qa_file" || return 1

  echo "[$label] Ragas AR/CR"
  cat "$run_dir/09_ar_cr/summary.json"
}

FAILED=()

if ! prepare_qasc; then
  FAILED+=("QASC preparation")
elif ! run_dataset "QASC" "$QASC_DATA" "$QASC_RUN" \
    "$QASC_QA" "$QASC_CORPUS" "$TWOQA_QASC100_RUN"; then
  FAILED+=("QASC run")
fi

if ! prepare_msmarco; then
  FAILED+=("MS MARCO preparation")
elif ! run_dataset "MS MARCO" "$MSMARCO_DATA" "$MSMARCO_RUN" \
    "$MSMARCO_QA" "$MSMARCO_CORPUS" "$TWOQA_MSMARCO100_RUN"; then
  FAILED+=("MS MARCO run")
fi

SUMMARY_DIR=$RUNS_DIR/qasc_msmarco${TWOQA_LIMIT}_summary
collector_args=(
  --qasc_run "$QASC_RUN"
  --msmarco_run "$MSMARCO_RUN"
  --output_dir "$SUMMARY_DIR"
)
"$PY" "$COLLECTOR" "${collector_args[@]}" || true

if ((${#FAILED[@]})); then
  echo "[FATAL] incomplete components: ${FAILED[*]}" >&2
  echo "[FATAL] rerun the same command; completed checkpoints will be reused" >&2
  echo "[FATAL] log: $MASTER_LOG" >&2
  exit 1
fi

echo
echo "[done] QASC and MS MARCO completed ($TWOQA_LIMIT queries per benchmark)"
echo "[done] combined summary: $SUMMARY_DIR/summary.json"
echo "[done] log: $MASTER_LOG"
