# RobotSim PR #45 launcher / dependency / provenance review

Reviewed head: `ddcab12057fcd84d64b979d49d91771302668280`

Review scope: launcher, runtime dependency declaration, result identity, output/re-run behavior, README and M0 status instructions, and third-party provenance. This is not a physics or acceptance review. Read-only review; no production files, simulation runs, or tests were performed.

## BLOCKING

### 1. Mesh override bypasses the checked Unitree pin

`scripts/run_m0_pick_place.sh:57-58` allows `ROBOTSIM_M0_MESH_DIR` to select any directory and checks only that it exists. The Unitree checkout is separately checked for exact commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` and clean status, but the selected mesh directory is not required to be within that checkout. The result JSON nevertheless unconditionally reports that pinned Unitree commit at `simulation/mujoco/m0_pick_place.py:1208-1210`.

Because README.md advertises this override, a normal invocation can use different meshes while claiming the pinned source. Remove the override or require the resolved mesh path to be inside the verified checkout and record/verify its provenance.

### 2. Preflight failure can leave a previous PASS result visible

The launcher creates the output directory, then checks Git/UV, upstream checkouts, mesh path, and installs dependencies before it enters the Python process (`scripts/run_m0_pick_place.sh:42-82`). Python marks the JSON `RUNNING` only at `simulation/mujoco/m0_pick_place.py:1272-1276`.

If a rerun fails before Python starts—for example missing `uv`, wrong/dirty upstream checkout, or package installation failure—the previous `m0_result.json` is left intact. A prior `passed: true` can therefore be mistaken for the result of the failed current invocation. Initialize a run-specific result state before preflight or use a unique output directory/run ID.

### 3. Result JSON does not identify the RobotSim source or complete runtime/config

The success result includes Humanoid VLA and Unitree commit constants, MuJoCo 3.2.6, seed, some scene geometry, and measured results (`simulation/mujoco/m0_pick_place.py:1200-1247`). It does not record RobotSim HEAD, actual Python version, installed dependency versions, or all effective acceptance/config thresholds. Error JSON is even less complete: the initial RUNNING object contains only issue/state, and the exception handler adds only the error (`:1272-1292`).

This is insufficient to associate a saved PASS/FAIL artifact with the exact RobotSim source and settings that produced it. Record RobotSim SHA, runtime Python and relevant package versions, upstream identities, seed, and the full effective thresholds/config in both success and failure records.

### 4. A pre-existing venv can silently bypass the Python 3.10 assumption

`scripts/run_m0_pick_place.sh:60-65` creates a Python 3.10 environment only if `venv/bin/python` is absent. If an existing executable is Python 3.11 or another version, the launcher reuses it, installs requirements, and runs without checking the interpreter version. README.md documents Python 3.10, but the launcher does not enforce it. Verify the existing interpreter and refuse/recreate an incompatible environment.

## NON-BLOCKING

- `requirements-m0.txt` pins the four direct packages exactly: NumPy 1.26.4, MuJoCo 3.2.6, h5py 3.14.0, and opencv-python-headless 4.11.0.86. This is a useful direct pin set, but there is no lockfile or hashes for the resolved transitive dependency graph. A platform-specific lock would strengthen repeatability.
- The default run, checkout, venv, and output paths are under `/tmp`; no absolute user-specific path or credential is needed. Environment path overrides are not checked to ensure outputs stay outside the repository. Fixed shared venv/output names can also collide across simultaneous invocations; sequential runs overwrite the prior artifacts.
- Exact full upstream SHAs are checked before normal launcher execution, and dirty/mismatched existing checkouts fail rather than being silently switched. The Python adapter independently checks the Humanoid VLA SHA and checks both the Python binding and native MuJoCo version against 3.2.6.
- Python 3.10 is explicit in the launcher and README. The exact patch release is not pinned.
- README.md and M0_STATUS.md reproduction commands match the launcher's default paths and supported checkout/output overrides. The shell uses `set -euo pipefail`; the Python process status is preserved from the `tee` pipeline, and Python task failure returns nonzero.

## NIT / third-party boundary

- At the checked Humanoid VLA commit `3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12`, the root `LICENSE` is MIT. At Unitree MuJoCo commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`, the root `LICENSE` is BSD 3-Clause.
- The PR tree does not include either external checkout, Unitree mesh files, or generated scene/model outputs. The adapter reads upstream model/scene files and writes its adapted XML under the output directory; defaults place this under `/tmp`. Vendor source and assets are fetched into external checkouts and the launcher does not intentionally edit those trees.
- Preserve both upstream license notices with their external checkouts. The existing `third_party/LOCK.md` cautions that repository-level notices do not settle rights for every specific G1 model/mesh asset. If generated XML, meshes, or derived media are copied into RobotSim or redistributed, resolve that asset-specific boundary and retain applicable notices first.

## Documentation / command consistency

The README default command and M0_STATUS reproduction command are accepted by the script. The documented Python 3.10 and MuJoCo 3.2.6 assumptions correspond to the intended environment, with the pre-existing-venv bypass above. Default outputs are external, but output-directory overrides are not constrained.

## Final review

**REQUEST CHANGES** due to the mesh provenance override, stale PASS-on-preflight-failure behavior, incomplete result identity/config, and unverified existing Python environment.

GitHub PR comment attempt failed with `Post https://api.github.com/graphql: Forbidden`; this report is published on the requested report-only branch instead.
