# RESIN reproduction guide

Run commands from the `RESIN_submit` directory. The generation model is
`gpt-5-nano`. The standalone RAGAS evaluator reports Answer Relevancy (AR) and
source-passage Context Recall (CR); the generation stage reports answer F1.

## 1. Environment and credentials

```bash
conda create -n resin python=3.10 -y
conda activate resin
pip install -r requirements.txt
cp .env.example .env
```

Set your own `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env`, then load them in
the shell used for the run:

```bash
set -a
source .env
set +a
```

Keep `.env`, generated `data/`, and `__pycache__/` out of a source-code archive.

## 2. Build the data, query plans, and retrieval checkpoints

The existing batch scripts create a complete **baseline** run, including data
preparation, indexing, query planning, top-10-per-hop retrieval, default
`concise` generation, RAGAS evaluation, and cost summaries. They do not select
the benchmark-specific generation styles described in section 3. Running
section 3 afterward reuses their retrieval results without recomputing them.

### HolisQA

Use the released `data/holisqa/` directory from the
[CausalRAG2 repository](https://github.com/Pwnb/CausalRAG2). Preserve all five
domain directories and each domain's `articles.jsonl`, `sentences.jsonl`, and
`qa.jsonl`. HolisQA preparation reads these local files; the QASC/MS MARCO
Hugging Face download path below does not download HolisQA. Then run:

```bash
START_MODE=fresh bash run_holisqa_all_queries.sh
```

If CausalRAG2 is already checked out elsewhere, point to its dataset directory:

```bash
SOURCE_DIR=/absolute/path/to/CausalRAG2/data/holisqa \
  START_MODE=fresh bash run_holisqa_all_queries.sh
```

A fresh run needs unused `DATA_DIR` and `RUN_DIR` paths; override those
variables when needed. Resume an interrupted run with the same paths:

```bash
START_MODE=resume bash run_holisqa_all_queries.sh
```

The default prepared data and run directories are:

```text
data/holisqa_resin_gpt5nano_all_queries_v1/
data/resin_runs/holisqa_gpt5nano_fast_hopwise_k10_all_queries_v1/
```

### QASC and MS MARCO

The existing script downloads the selected validation examples through Hugging
Face and runs both benchmarks. Its default limit is 500 eligible examples
per benchmark; rerun the same command to resume:

| Benchmark | Dataset | Configuration | Split |
|---|---|---|---|
| QASC | `allenai/qasc` | default | `validation` |
| MS MARCO | `microsoft/ms_marco` | `v1.1` | `validation` |

```bash
TWOQA_LIMIT=500 bash run_two_qa.sh
```

The script runs QASC and MS MARCO. Its combined baseline summary is written under
`data/resin_runs/qasc_msmarco500_summary/`.

For this setting, its default data and run directories are:

| Benchmark | Prepared data | Run directory |
|---|---|---|
| QASC | `data/qasc_resin_fact500` | `data/resin_runs/qasc_fact500_gpt5nano_k10` |
| MS MARCO v1.1 | `data/msmarco_v11_resin_passage500` | `data/resin_runs/msmarco_v11_passage500_gpt5nano_k10` |

These paths come from `run_two_qa.sh`. Runs built separately, such as a
`next500` batch, can use the same generation commands below by setting `DATA`
and `RUN` to their actual directories. A usable run must contain
`RUN/05_retrieval/retrieval.jsonl`, and its matching `DATA` directory must
contain the QA and corpus files. Do not mix retrieval from one run with QA or
corpus from another.

## 3. Run the final generation style on existing retrieval

Select **one** benchmark by setting `BENCHMARK` below. The case block sets the
paths created by the batch scripts. If using a separately prepared run,
replace `DATA` and `RUN` after the case block with its actual paths.

```bash
BENCHMARK=holisqa  # holisqa | qasc | msmarco

case "$BENCHMARK" in
  holisqa)
    DATA="$PWD/data/holisqa_resin_gpt5nano_all_queries_v1"
    RUN="$PWD/data/resin_runs/holisqa_gpt5nano_fast_hopwise_k10_all_queries_v1"
    QA_NAME=holisqa; STYLE=holisqa; EFFORT=minimal; TOKENS=1800 ;;
  qasc)
    DATA="$PWD/data/qasc_resin_fact500"
    RUN="$PWD/data/resin_runs/qasc_fact500_gpt5nano_k10"
    QA_NAME=qasc; STYLE=qasc; EFFORT=minimal; TOKENS=1800 ;;
  msmarco)
    DATA="$PWD/data/msmarco_v11_resin_passage500"
    RUN="$PWD/data/resin_runs/msmarco_v11_passage500_gpt5nano_k10"
    QA_NAME=msmarco; STYLE=msmarco; EFFORT=medium; TOKENS=3000 ;;
  *) echo "Unknown benchmark: $BENCHMARK" >&2; exit 1 ;;
esac

# For an existing separately prepared batch, override both paths here.
# DATA=/absolute/path/to/prepared_data
# RUN=/absolute/path/to/its_run

QA="$DATA/qa/$QA_NAME.jsonl"
CORPUS="$DATA/corpus/corpus.json"
RETRIEVAL="$RUN/05_retrieval/retrieval.jsonl"
OUT="$RUN/07_agent_${STYLE}_final"

test -s "$QA" && test -s "$CORPUS" && test -s "$RETRIEVAL" || {
  echo "Missing matching QA, corpus, or retrieval file" >&2
  exit 1
}

python3 -u run_resin_fast.py \
  --stage agent \
  --data_dir "$DATA" \
  --qa_file "$QA" \
  --corpus_file "$CORPUS" \
  --run_dir "$RUN" \
  --reasoning_mode hopwise_candidates \
  --generation_style "$STYLE" \
  --generation_output_dir "$OUT" \
  --provider openai --model gpt-5-nano \
  --base_url https://api.openai.com/v1 \
  --reasoning_effort "$EFFORT" \
  --temperature 0.1 --max_tokens "$TOKENS" \
  --hop_top_k 10 --max_hops 4 \
  --workers 4 --hop_workers 2

cat "$OUT/summary.json"
```

The generation stage writes checkpoints and `answers.jsonl` under `OUT` and
resumes successful checkpoints with the same generation signature. Set `OUT`
to a **new directory** when changing the style or running a new comparison.
The original `RUN/07_agent` baseline remains separate. No extraction,
planning, or retrieval stage is invoked by this command. It requires the same
QA/corpus bundle used for the existing retrieval run.

## 4. Read answer F1

The generation command in section 3 computes F1 automatically after writing
the answers. No separate evaluator command is needed for the final generation
style. `F1_diagnostic` in `OUT/summary.json` is the percentage over the queries
actually generated, including failed generations as zero; `OUT/f1_per_query.jsonl`
contains each query's prediction, reference answer, and F1. For example:

```bash
python3 - "$OUT/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"queries={summary['queries']} successful={summary['successful']} "
      f"errors={summary['errors']} F1={summary['F1_diagnostic']:.2f}")
PY
```

This uses the same whitespace/punctuation-normalized token F1 as
`RUN/08_evaluation`. The latter belongs to the batch script's default
`concise` baseline, so use `OUT/summary.json` for the final generation style.

## 5. Evaluate those exact answers with RAGAS AR/CR

Use the variables from section 3 in the same shell. Generation writes a
matching QA/answer/retrieval subset under `OUT/evaluation_subset/`; use these
three files even for a full run. This also restricts evaluation to the generated
queries when `--generation_domain` or `--generation_limit` is used:

```bash
EVAL_QA="$OUT/evaluation_subset/qa.jsonl"
EVAL_ANSWERS="$OUT/evaluation_subset/answers.jsonl"
EVAL_RETRIEVAL="$OUT/evaluation_subset/retrieval.jsonl"

python3 -u evaluate_ar_cr.py \
  --run_dir "$RUN" \
  --qa_file "$EVAL_QA" \
  --corpus_file "$CORPUS" \
  --answers_file "$EVAL_ANSWERS" \
  --retrieval_file "$EVAL_RETRIEVAL" \
  --output_dir "$OUT/09_ar_cr" \
  --judge_provider gemini \
  --judge_model gemini-3.5-flash-lite \
  --base_url https://generativelanguage.googleapis.com/v1beta/openai \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --strictness 3 --max_contexts 40 \
  --cr_context_mode source_passage --workers 4

cat "$OUT/09_ar_cr/summary.json"
```

For HolisQA, aggregate AR/CR by domain from this final generation run:

```bash
python3 -u summarize_ragas_by_category.py \
  --qa_file "$EVAL_QA" \
  --evaluation_dir "$OUT/09_ar_cr"
```

To tabulate HolisQA F1 by domain for the same generated answers:

```bash
python3 - "$QA" "$OUT/f1_per_query.jsonl" "$OUT/f1_by_category.json" <<'PY'
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]

scores = {row["query_id"]: row["f1"] for row in rows(sys.argv[2])}
grouped = defaultdict(list)
for index, row in enumerate(rows(sys.argv[1])):
    query_id = f'q_{index:06d}_{hashlib.sha1(row["query"].encode()).hexdigest()[:8]}'
    if query_id in scores:
        grouped[str(row.get("category") or row.get("domain") or "unknown")].append(scores[query_id])
result = {name: {"count": len(values), "F1": round(100 * sum(values) / len(values), 2)}
          for name, values in sorted(grouped.items())}
Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
```

## 6. Read and report results

| Artifact | Meaning |
|---|---|
| `OUT/summary.json` | Query count, failures, `F1_diagnostic`, and optional F1 comparison with `RUN/07_agent` on the same queries. |
| `OUT/f1_per_query.jsonl` | Per-query answer F1 after generation. |
| `OUT/answers.jsonl` | Generated answers used for AR. |
| `OUT/evaluation_subset/{qa,answers,retrieval}.jsonl` | Matched rows for standalone AR/CR evaluation, including selected query subsets. |
| `OUT/09_ar_cr/summary.json` | RAGAS AR, CR, and diagnostic CR_ID for the specified answer and retrieval files. |
| `OUT/09_ar_cr/per_query.jsonl` | Per-query RAGAS results. |
| `OUT/09_ar_cr/summary_by_category.json` | HolisQA AR/CR by domain after running the category summarizer. |
| `OUT/f1_by_category.json` | HolisQA F1 by domain after running the optional tabulation above. |

F1 is the whitespace/punctuation-normalized token F1 reported as a percentage.
Reference answers are used only **after** generation for this diagnostic and
for RAGAS evaluation; the generation prompts use the question and existing
retrieved evidence. CR is evaluated from the unchanged retrieval output, while
AR depends on the generated answer. Always compare scores from the same query
set and evaluation protocol. `RUN/08_evaluation` and `RUN/09_ar_cr` generated
by the batch scripts belong to the default baseline; report `OUT` artifacts
for the benchmark-specific generation styles.

The batch scripts' `10_cost` summaries also describe their default baseline
generation and should not be attributed to a later `OUT` generation run.
