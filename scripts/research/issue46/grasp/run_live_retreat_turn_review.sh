#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo"
exec /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46/grasp/live_retreat_turn_review.py "$@"
