#!/usr/bin/env bash
# Generalization subset of the paper: source chains, target chains from scratch, transfer chains, videos.
#   bash scripts/run_generalization.sh [system]      (default gpt-6-astra; see configs/systems/)
# Long-running (up to ~17 rounds x ~70 min per chain): start it inside tmux.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(python3 -c "import json;print(json.load(open('$REPO/configs/local.json'))['robodojo_python'])")"
RUNS="$(python3 -c "import json;print(json.load(open('$REPO/configs/local.json')).get('runs_dir','runs'))")"
S="${1:-gpt-6-astra}"
cd "$REPO"
run() { echo ">>> $*"; "$PY" -m physrsi_eval "$@"; }

for T in stack_blocks_by_language insert_key cover_blocks plug_in_charger; do
  run k0    --system "$S" --task "$T"
  run chain --system "$S" --task "$T"
done
run transfer --system "$S" --target cover_blocks    --source "stack_blocks_by_language=$RUNS/$S/stack_blocks_by_language/c1"
run transfer --system "$S" --target plug_in_charger --source "insert_key=$RUNS/$S/insert_key/c1"
for T in cover_blocks plug_in_charger; do
  run video --chain "$RUNS/$S/$T/c1"          --out "videos/$S/$T"
  run video --chain "$RUNS/$S/$T/transfer-c1" --out "videos/$S/$T"
done
