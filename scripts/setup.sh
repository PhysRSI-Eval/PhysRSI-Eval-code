#!/usr/bin/env bash
# One-step setup of the PhysRSI-Eval reference infrastructure.
#
#   bash scripts/setup.sh            # everything: RoboDojo + assets + this package + Codex CLI + verification
#   bash scripts/setup.sh --verify   # verification only
#
# Pins (tasks/lock.json): RoboDojo commit and submodules, RoboDojo asset dataset revision, Codex CLI version.
# Nothing large lives in this repository; RoboDojo, its assets and Isaac Sim are fetched from their upstreams.
#
# Environment overrides:
#   PHYSRSI_DEPS      where RoboDojo is cloned            (default: <repo>/third_party)
#   ROBODOJO_PYTHON   an existing RoboDojo Python to reuse (default: the conda env "RoboDojo" made by RoboDojo's installer)
#   SKIP_CODEX=1      do not install the Codex CLI
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$REPO/tasks/lock.json"
DEPS="${PHYSRSI_DEPS:-$REPO/third_party}"
ROBODOJO="$DEPS/RoboDojo"
lock() { python3 -c "import json,sys;d=json.load(open('$LOCK'));print(eval('d'+sys.argv[1]))" "$1"; }
info() { echo -e "\e[1;32m>>> $*\e[0m"; }
fail() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }

if [[ "${1:-}" != "--verify" ]]; then
  command -v git >/dev/null || fail "git is required"
  git lfs version >/dev/null 2>&1 || fail "git-lfs is required (RoboDojo assets)"

  # 1. RoboDojo at the pinned commit, submodules at the commits recorded by that commit.
  RD_REPO="$(lock "['robodojo']['repo']")"; RD_COMMIT="$(lock "['robodojo']['commit']")"
  if [[ ! -d "$ROBODOJO/.git" ]]; then
    info "Cloning RoboDojo into $ROBODOJO"
    mkdir -p "$DEPS"; git clone "$RD_REPO" "$ROBODOJO"
  fi
  git -C "$ROBODOJO" fetch --quiet origin "$RD_COMMIT" || true
  git -C "$ROBODOJO" checkout --quiet "$RD_COMMIT"
  git -C "$ROBODOJO" submodule sync --quiet
  git -C "$ROBODOJO" submodule update --init --recursive

  # 2. RoboDojo's own installer (conda env "RoboDojo", Isaac Sim 5.1, IsaacLab, cuRobo). Its submodule step
  #    tracks the remote branch head; the copy below keeps the pinned submodule commits instead.
  if [[ -z "${ROBODOJO_PYTHON:-}" ]]; then
    sed 's/submodule update --init --remote/submodule update --init/' "$ROBODOJO/scripts/install.sh" \
      > "$ROBODOJO/scripts/install_pinned.sh"
    info "Running RoboDojo installer (pinned submodules)"
    (cd "$ROBODOJO" && bash scripts/install_pinned.sh -i)
  fi

  # 3. RoboDojo assets at the pinned dataset revision (RoboDojo's own sparse git-lfs download).
  info "Downloading RoboDojo assets (revision $(lock "['assets']['revision']"))"
  (cd "$ROBODOJO" && HF_REVISION="$(lock "['assets']['revision']")" bash scripts/init_assets.sh)
fi

# 4. Resolve the RoboDojo Python and install this package into it.
if [[ -z "${ROBODOJO_PYTHON:-}" ]]; then
  for candidate in "$HOME/miniconda3/envs/RoboDojo/bin/python" "$(conda info --base 2>/dev/null)/envs/RoboDojo/bin/python"; do
    [[ -x "$candidate" ]] && ROBODOJO_PYTHON="$candidate" && break
  done
fi
[[ -x "${ROBODOJO_PYTHON:-}" ]] || fail "RoboDojo Python not found; set ROBODOJO_PYTHON"
if [[ "${1:-}" != "--verify" ]]; then
  "$ROBODOJO_PYTHON" -m pip install --quiet -e "$REPO"
  if [[ "${SKIP_CODEX:-0}" != "1" ]]; then
    command -v npm >/dev/null || fail "npm is required for the Codex CLI (or set SKIP_CODEX=1)"
    npm install -g "$(lock "['codex']['package']")@$(lock "['codex']['version_in_paper']")"
  fi
fi

# 5. Local configuration (not committed) and verification.
CONFIG="$REPO/configs/local.json"
if [[ ! -f "$CONFIG" ]]; then
  "$ROBODOJO_PYTHON" - "$CONFIG" "$ROBODOJO" "$ROBODOJO_PYTHON" "$REPO" <<'PY'
import json, sys
config, robodojo, python, repo = sys.argv[1:]
example = json.load(open(repo + '/configs/local.example.json'))
example.update(robodojo_root=robodojo, robodojo_python=python, runs_dir=repo + '/runs')
json.dump(example, open(config, 'w'), indent=1)
print('wrote', config)
PY
fi
"$ROBODOJO_PYTHON" -m physrsi_eval check --config "$CONFIG"
