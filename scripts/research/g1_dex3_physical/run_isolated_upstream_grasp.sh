#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
branch=$(git branch --show-current)
if [[ -z "$branch" || "$branch" == main || "$branch" == master ]]; then
  printf '%s\n' "Run this from a dedicated, non-default worktree." >&2
  exit 2
fi
git_dir=$(git rev-parse --git-dir)
common_dir=$(git rev-parse --git-common-dir)
if [[ "$git_dir" == "$common_dir" ]]; then
  printf '%s\n' "Run this from an isolated Git worktree, not the primary checkout." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  printf '%s\n' "The benchmark worktree must be clean." >&2
  exit 2
fi

python_bin=${G1_MUJOCO_PYTHON:-/home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
  printf '%s\n' "The pinned G1 MuJoCo Python environment is unavailable." >&2
  exit 2
fi
"$python_bin" -c 'import mujoco; assert mujoco.__version__ == "3.3.6"'

cd "$repo_root"
export MUJOCO_GL=${MUJOCO_GL:-egl}
"$python_bin" scripts/research/g1_dex3_physical/baseline.py

plan_output=$("$python_bin" scripts/research/g1_dex3_physical/plan.py)
printf '%s\n' "$plan_output"
plan_dir=$(printf '%s\n' "$plan_output" | "$python_bin" -c '
import json
import sys

text = sys.stdin.read()
decoder = json.JSONDecoder()
matches = []
for index, char in enumerate(text):
    if char != "{":
        continue
    try:
        value, _ = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    plan = value.get("plan") if isinstance(value, dict) else None
    evidence_dir = plan.get("evidence_directory") if isinstance(plan, dict) else None
    if isinstance(evidence_dir, str):
        matches.append(evidence_dir)
if not matches:
    raise SystemExit("plan output did not include an evidence_directory")
print(matches[-1])
')

run_id="local02-upstream-$(date -u +%Y%m%dT%H%M%SZ)"
"$python_bin" scripts/research/g1_dex3_physical/episode.py \
  --run-id "$run_id" \
  --attempt 1 \
  --plan-dir "$plan_dir"
