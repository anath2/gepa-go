#!/usr/bin/env bash
# Minimal live smoke workflow: 1 module + 1 example to validate pipeline integrity.
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

EXAMPLE="$ROOT/examples/smoke_minimal"
for f in program.json config.json train.jsonl val.jsonl; do
  if [[ ! -f "$EXAMPLE/$f" ]]; then
    echo "error: missing fixture $EXAMPLE/$f" >&2
    exit 1
  fi
done

echo "building gepa..."
go build -o "$ROOT/gepa" ./cmd/gepa

RUN_ID="smoke-minimal-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"

echo "run id: $RUN_ID"
start_ts=$(date +%s)

"$ROOT/gepa" optimize \
  --program "$EXAMPLE/program.json" \
  --config "$EXAMPLE/config.json" \
  --train "$EXAMPLE/train.jsonl" \
  --val "$EXAMPLE/val.jsonl" \
  --run-id "$RUN_ID"

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))

for name in state.json events.jsonl result.json candidates/0000.json; do
  if [[ ! -f "$RUN_DIR/$name" ]]; then
    echo "error: missing required artifact $RUN_DIR/$name" >&2
    exit 1
  fi
done

python3 - "$RUN_DIR/events.jsonl" <<'PY'
import json, sys
path = sys.argv[1]
types = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        types.append(json.loads(line).get('type'))

must_have = {'seed_evaluated', 'proposal_requested'}
if not must_have.issubset(set(types)):
    missing = sorted(must_have - set(types))
    raise SystemExit(f"error: events.jsonl missing required event types: {missing}")

terminal = {'candidate_accepted', 'candidate_rejected', 'proposal_failed'}
if not (set(types) & terminal):
    raise SystemExit("error: events.jsonl missing terminal proposal outcome (accepted/rejected/proposal_failed)")

print("event types:", sorted(set(types)))
PY

"$ROOT/gepa" inspect "$RUN_DIR" >/dev/null

python3 - "$RUN_DIR/result.json" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    r = json.load(f)
print(f"metric_calls={r.get('metric_calls')} best_candidate={r.get('best_candidate')} train_mean={r.get('train_mean')}")
PY

echo "elapsed_seconds=$elapsed"
echo "run directory: $RUN_DIR"
echo "inspect:"
"$ROOT/gepa" inspect "$RUN_DIR"
