#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOCK="$ROOT/third_party/LOCK.md"

if (( $# > 1 )); then
  printf 'Usage: %s [destination-directory]\n' "$0" >&2
  exit 2
fi

if [[ ! -f "$LOCK" ]]; then
  printf 'Missing third-party lock file: %s\n' "$LOCK" >&2
  exit 1
fi

TP_DIR="${1:-$ROOT/third_party}"
mkdir -p "$TP_DIR"
TP_DIR="$(cd "$TP_DIR" && pwd -P)"

die() {
  printf 'fetch_third_party: %s\n' "$*" >&2
  exit 1
}

pin_record() {
  local project="$1"
  awk -F '|' -v project="$project" '
    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }
    trim($2) == project {
      print trim($3) " " trim($4)
      count++
    }
    END { if (count != 1) exit 1 }
  ' "$LOCK"
}

fetch_repo() {
  local project="$1"
  local expected_url="$2"
  local commit="$3"
  local destination="$TP_DIR/$project"
  local top_level origin_url current_commit state
  local -a origin_urls=()

  if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
    die "invalid full commit SHA for $project: $commit"
  fi

  if [[ -L "$destination" ]]; then
    die "refusing symlink destination: $destination"
  elif [[ ! -e "$destination" ]]; then
    mkdir "$destination"
    git -C "$destination" init --quiet
    git -C "$destination" remote add origin "$expected_url"
  elif [[ ! -d "$destination/.git" ]]; then
    die "destination exists but is not a normal Git checkout: $destination"
  fi

  top_level="$(git -C "$destination" rev-parse --show-toplevel 2>/dev/null)" \
    || die "destination is not a Git worktree: $destination"
  [[ "$top_level" == "$destination" ]] \
    || die "destination resolves to a different worktree: $destination"

  mapfile -t origin_urls < <(git -C "$destination" remote get-url --all origin 2>/dev/null) \
    || die "checkout has no origin remote: $destination"
  [[ ${#origin_urls[@]} -eq 1 && "${origin_urls[0]}" == "$expected_url" ]] \
    || die "origin for $project must be exactly $expected_url"

  state="$(git -C "$destination" status --porcelain --untracked-files=all)" \
    || die "could not inspect checkout status: $destination"
  [[ -z "$state" ]] \
    || die "refusing to change a checkout with local changes: $destination"

  current_commit="$(git -C "$destination" rev-parse --verify HEAD 2>/dev/null || true)"
  if [[ "$current_commit" != "$commit" ]]; then
    git -C "$destination" fetch --quiet --no-tags --depth=1 origin "$commit" \
      || die "could not fetch pinned commit $commit for $project"
    git -C "$destination" checkout --quiet --detach "$commit" \
      || die "could not check out pinned commit $commit for $project"
  else
    git -C "$destination" checkout --quiet --detach "$commit" \
      || die "could not detach $project at pinned commit $commit"
  fi

  current_commit="$(git -C "$destination" rev-parse --verify HEAD)"
  [[ "$current_commit" == "$commit" ]] \
    || die "$project ended at $current_commit instead of $commit"
  state="$(git -C "$destination" status --porcelain --untracked-files=all)"
  [[ -z "$state" ]] \
    || die "$project checkout is not clean after pinning: $destination"

  printf 'Verified %-16s %s\n' "$project" "$commit"
}

while IFS= read -r project; do
  record="$(pin_record "$project")" \
    || die "expected exactly one $project pin in $LOCK"
  read -r repository commit <<< "$record"

  case "$project" in
    unitree_sdk2|unitree_mujoco|unitree_ros2)
      expected_url="https://github.com/unitreerobotics/$project.git"
      [[ "$repository" == "$expected_url" ]] \
        || die "unexpected repository URL for $project: $repository"
      fetch_repo "$project" "$expected_url" "$commit"
      ;;
    *)
      die "unsupported third-party project in fetch list: $project"
      ;;
  esac
done <<'EOF'
unitree_sdk2
unitree_mujoco
unitree_ros2
EOF

printf 'Pinned source checkouts are ready in %s\n' "$TP_DIR"
printf 'No vendor binaries are downloaded by this script. See %s for provenance.\n' "$LOCK"
