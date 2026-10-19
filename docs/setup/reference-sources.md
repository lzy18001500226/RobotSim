# External Reference Sources

## Canonical location

All new external open-source repositories, vendor SDK source checkouts, and
third-party source dependencies belong under:

```text
reference/<GitHubOwner>/<RepositoryName>/
```

For example, `reference/unitreerobotics/unitree_mujoco/`. The root `.gitignore`
ignores only the repository-root `/reference/` tree. It is a local source
checkout area, not a place for RobotSim experiments, generated evidence,
build/install output, tracked metadata, or copied production assets.

Keep tracked provenance outside this ignored tree. Record the canonical
repository URL, immutable 40-character commit SHA, license file/path and review
status, plus any asset-relative paths and verified hashes in an appropriate
tracked source or asset registry. A moving branch or tag alone is not a pin.
Do not infer redistribution rights for model/mesh files from a repository-level
license without reviewing their scope.

## Fetch procedure

For a new source, first verify the owner/repository spelling, the exact commit,
and that the destination does not exist. Then fetch only that commit, without
tags, LFS payload smudging, or recursive submodules:

```bash
owner='GitHubOwner'
repo='RepositoryName'
url="https://github.com/${owner}/${repo}.git"
commit='0123456789abcdef0123456789abcdef01234567'
destination="reference/${owner}/${repo}"

test ! -e "$destination" && test ! -L "$destination"
mkdir -p "reference/${owner}"
git init "$destination"
git -C "$destination" remote add origin "$url"
GIT_LFS_SKIP_SMUDGE=1 git -C "$destination" fetch --no-tags --depth=1 origin "$commit"
git -C "$destination" checkout --detach FETCH_HEAD
test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$destination" status --porcelain)"
git check-ignore -v "$destination"
```

Do not initialize submodules automatically. Inspect `.gitmodules` and gitlink
entries and record their exact pins and missing resources first. When sparse
checkout is necessary, list the included and excluded paths in the evidence
record; do not imply the excluded files are present. Use
`GIT_LFS_SKIP_SMUDGE=1`, do not run `git lfs pull`, and identify LFS pointer
files explicitly. Do not execute upstream scripts or install dependencies as
part of source collection. Keep build and install directories outside the
checkout.

## Existing `third_party/` compatibility

`third_party/LOCK.md`, `scripts/fetch_third_party.sh`,
`scripts/bootstrap_g1_sim.sh`, and `scripts/run_g1_mujoco_smoke.sh` currently
form a legacy Unitree path contract. The fetch script's default destination
and the G1 consumers are deliberately unchanged by this policy. Do not use
that script to add new repositories.

Moving any existing checkout or changing its active path requires a separate
reviewed migration that updates the lock/source registry, fetch behavior,
bootstrap and smoke consumers, documentation, and tests together. Preserve
compatibility for existing research branches until their owners migrate. Do
not copy or delete old checkouts as part of the path change; verify pins and
clean status before and after an explicitly approved migration.

PR #73's asset registry is tracked metadata, not an upstream source checkout.
It can remain tracked outside `reference/`, but before adoption its repository
records should state the canonical `reference/<owner>/<repo>` checkout path
and must not describe `third_party/` as the new source root. The registry must
continue to distinguish verified asset provenance from unknown provenance.
