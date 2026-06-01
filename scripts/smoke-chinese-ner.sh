#!/usr/bin/env bash
# Top-down smoke workflow: build gepa, run Chinese NER optimize, inspect the run.
# Requires project-root .env with API_KEY and BASE_URL (see .env.example).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

load_env() {
  local env_file="$ROOT/.env"
  if [[ ! -f "$env_file" ]]; then
    echo "error: missing $env_file (copy from .env.example and set API_KEY, BASE_URL)" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim leading/trailing whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    export "$line"
  done < "$env_file"
}

load_env

if [[ -z "${API_KEY:-}" ]]; then
  echo "error: API_KEY not set in .env" >&2
  exit 1
fi
if [[ -z "${BASE_URL:-}" ]]; then
  echo "error: BASE_URL not set in .env" >&2
  exit 1
fi

EXAMPLE="$ROOT/examples/chinese_ner"
for f in program.json config.json train.jsonl val.jsonl; do
  if [[ ! -f "$EXAMPLE/$f" ]]; then
    echo "error: missing fixture $EXAMPLE/$f" >&2
    exit 1
  fi
done

echo "building gepa..."
go build -o "$ROOT/gepa" ./cmd/gepa

RUN_ID="chinese-ner-$(date +%Y%m%d-%H%M%S)"
echo "run id: $RUN_ID"

"$ROOT/gepa" optimize \
  --program "$EXAMPLE/program.json" \
  --config "$EXAMPLE/config.json" \
  --train "$EXAMPLE/train.jsonl" \
  --val "$EXAMPLE/val.jsonl" \
  --run-id "$RUN_ID"

RUN_DIR="$ROOT/runs/$RUN_ID"
echo ""
echo "run directory: $RUN_DIR"
echo "artifacts:"
for name in state.json events.jsonl result.json candidates/0000.json; do
  if [[ -e "$RUN_DIR/$name" ]]; then
    echo "  $RUN_DIR/$name"
  fi
done

echo ""
echo "inspect:"
"$ROOT/gepa" inspect "$RUN_DIR"
