# PR #45 clean-environment reproducibility audit

- Reviewed head: `ddcab12057fcd84d64b979d49d91771302668280`
- Checkout: fresh clone of the canonical RobotSim repository, detached at the
  exact PR head.
- Scope: launcher/environment reproducibility only; no PR production code was
  changed.
- Result: the full M0 passes from a clean run directory when invoked with
  `bash`, but the documented direct command fails on the fresh checkout because
  the launcher is not executable.

## BLOCKING

The README instructs users to run `./scripts/run_m0_pick_place.sh`, but Git
records that file as mode `100644`. On the fresh checkout, the documented
command returned `Permission denied` (exit 126) before the launcher ran. Running
`bash scripts/run_m0_pick_place.sh` succeeds. For reproducibility as documented,
the executable bit must be committed, or the README must instruct users to run
the script through Bash. This audit did not change that file.

## Environment and execution

The fresh checkout was run from `/` with `PYTHONPATH` unset, an empty isolated
`HOME`, an empty `uv` cache, and a new run directory at
`/tmp/robotsim-pr45-clean-run`. No prior `/tmp` run directory, vendor checkout,
Python environment, Local02 patch, ROS 2, Unity, display, or GPU was used. The
launcher created its checkouts, Python environment, and output directories.

The direct README invocation failed as described above. The equivalent
`bash` invocation then completed successfully: all five task stages
(`GRASP`, `LIFT`, `TRANSFER`, `RELEASE`, `PLACE`) passed, final result JSON said
`passed: true`, and the process exited 0. This demonstrates headless execution
in the available CPU-only cloud environment. It had no `DISPLAY` and no
`nvidia-smi`, but did provide Mesa `libEGL.so.1`; MuJoCo's configured EGL path
rendered successfully without a visible window.

## Hidden prerequisites and pin checks

- No upstream checkout is required beforehand. From an empty run directory the
  launcher cloned both public upstream repositories itself.
- Humanoid VLA resolved to `3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12` from
  `https://github.com/ozkannceylan/humanoid_vla.git`.
- Unitree MuJoCo resolved to `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` from
  `https://github.com/unitreerobotics/unitree_mujoco.git`.
- The runtime created CPython `3.10.21` through `uv`; the M0 code checked both
  the Python and native MuJoCo versions and ran with MuJoCo `3.2.6`.
- The launcher installs from `simulation/mujoco/requirements-m0.txt` without
  an existing venv. Direct requirements are version-pinned (`numpy==1.26.4`,
  `mujoco==3.2.6`, `h5py==3.14.0`, `opencv-python-headless==4.11.0.86`). The
  clean install resolved and installed 12 packages. Transitive versions and the
  Python patch release are not locked by a checked-in lockfile, so this is
  repeatable at the direct-dependency level, not a byte-for-byte environment
  lock.
- No credentials were used. Fresh clones and package downloads require public
  network access to GitHub and Python package hosting; the cloud proxy allowed
  those requests.
- The script requires `bash`, `git`, and `uv` on `PATH`. It finds the repository
  root relative to its own location and completed from `/` when invoked via
  Bash; it does not depend on a particular current directory.
- The renderer requires an EGL runtime. This cloud image already supplied
  Mesa's `libEGL.so.1`; the Python requirements do not install the system EGL
  library, and the README does not name it as an OS prerequisite.
- `ROBOTSIM_M0_RUN_DIR`, `ROBOTSIM_M0_CANDIDATE_DIR`,
  `ROBOTSIM_M0_UNITREE_DIR`, `ROBOTSIM_M0_OUTPUT_DIR`, and
  `ROBOTSIM_M0_MESH_DIR` are optional overrides. No RobotSim-specific override
  was necessary except the run-directory override used to isolate this audit.
  No external data file or hidden local patch was needed.

## Rejection and vendor-integrity checks

Synthetic temporary checkout fixtures exercised the rejection paths without
altering the canonical fetched checkouts:

- A clean Humanoid VLA checkout at the wrong commit was rejected with the
  expected and actual SHA in the message.
- A dirty Humanoid VLA checkout at the expected commit was rejected with a
  checkout-specific message.
- The same wrong-commit and dirty-checkout checks passed for Unitree MuJoCo.

Both actual vendor checkouts used by the demo remained clean after execution.
The launcher generated the adapted scene only under the output directory; it
did not patch either vendor tree.

## Artifacts

The successful run created non-empty files under
`/tmp/robotsim-pr45-clean-run/output`:

- `m0_result.json` (45,984 bytes; `passed: true`; all task and safety checks
  passed)
- `run.log` (2,962 bytes)
- `m0_pick_place.mp4` (5,606,063 bytes; probe reports MP4, 27.2 seconds)
- `m0_final.png` (210,316 bytes)
- `model/model/g1_m0_bottle.xml` (generated scene)

The launcher tees the simulation output to `run.log`, captures the Python
process status from the pipeline, and writes a failure JSON plus a `FAIL:`
diagnostic when the demo raises an exception. Wrong-commit and dirty-checkout
errors are clear. The documented direct-launch permission failure itself is
clear, but prevents reaching those safeguards until the mode/documentation issue
is corrected.

## Classification

### BLOCKING

- The advertised `./scripts/run_m0_pick_place.sh` command cannot execute from a
  fresh clone because the file mode is `100644`. This directly fails the
  documented clean-machine workflow.

### NON-BLOCKING

- Python top-level packages are pinned but no complete resolver lock or hashes
  pin transitive dependencies and Python patch version.
- EGL availability is not declared in the README. The clean Cloud run had Mesa
  EGL and proved that a visible display and GPU are unnecessary, but a host
  without `libEGL.so.1` will need an OS EGL/Mesa runtime package.
- The README could mention that the launcher clones public repositories and
  installs wheels, which requires network access; no credentials are required.

### ENVIRONMENT-ONLY

- None blocked this run. The available cloud image supplied public network
  access, `uv`, and Mesa EGL; no ROS, Unity, display server, or GPU was present
  or needed for the M0 acceptance run.

## Review conclusion

Another developer can reproduce the successful M0 using the Bash workaround
and public network access. They cannot reproduce it using the exact documented
command on a fresh checkout until the launcher executable mode (or documented
invocation) is corrected. No PR code or vendor source was modified by this
audit.

## Final launcher/evidence-integrity re-review

- Reviewed exact PR head: `5f08abaa748165936676f8b4c36dbe5cbd0adbfb`.
- This was a read-only review of the final launcher/evidence patch. The PR
  checkout remained clean; only this report-only branch was updated.
- The prior executable-bit blocker is resolved: Git records
  `scripts/run_m0_pick_place.sh` as `100755`, and direct invocation from `/`
  completed the M0 demo.

### Reproduced behavior

- A fresh invocation writes `PREFLIGHT` / `passed: false` before dependency
  checks. With a stale prior `PASS` result present and `uv` hidden from `PATH`,
  the invocation replaced it with a new `PREFLIGHT_FAILED` record and a new run
  ID. The ordinary failure record parsed as JSON.
- The exact pinned Humanoid VLA and Unitree MuJoCo commits are checked. Wrong
  commits and dirty checkouts for both repositories fail closed with actionable
  diagnostics. An external `ROBOTSIM_M0_MESH_DIR` is rejected unless it
  resolves to the G1 mesh directory inside the verified Unitree checkout.
- A reused Python 3.12 environment is rejected; Python 3.10.x is required.
- The direct exact-head run passed all task and safety checks in CPU/EGL
  headless mode. Its JSON recorded run ID, RobotSim SHA and clean state, Python
  and MuJoCo/NumPy/h5py/OpenCV versions, both upstream SHAs, mesh provenance,
  seed, and output directory. JSON, log, MP4, screenshot, trace, and generated
  model outputs were present. No manipulation/controller or acceptance logic
  changed in this patch.
- Upstream test checkouts used for rejection cases were fixtures under `/tmp`;
  the pinned vendor checkouts used by the successful run remained clean.
- No credential handling or hardcoded personal home path was introduced. The
  selected output directory and checkout paths can appear in result/error
  fields by design; the output directory is part of the requested run identity.

### BLOCKING: JSON control-character escaping

`scripts/run_m0_pick_place.sh:17-22` escapes backslash, quote, and newline,
but does not escape other JSON control characters. A valid configured output
path containing a tab (`ROBOTSIM_M0_OUTPUT_DIR=$'/tmp/...\t...'`) followed by
an early preflight failure wrote a literal tab into `output_directory` at
`scripts/run_m0_pick_place.sh:47`; Python `json.loads` rejected the failure
record with `Invalid control character`. Thus failure records are not
guaranteed to remain valid JSON for all accepted path values. Escape all JSON
U+0000–U+001F characters (or use a JSON encoder) before considering the final
evidence-integrity gate complete.

### CI and review status

The exact-head GitHub checks page showed successful `Agent infrastructure` and
`G1 MuJoCo smoke` workflows for both `push` and `pull_request` events (four
successful runs total). `gh` could not authenticate in this environment, so a
PR review comment could not be posted; this finding is persisted in the existing
report-only branch instead.

Final focused review status: **REQUEST CHANGES** for the JSON control-character
edge case. The direct M0 reproduction and the other launcher/evidence checks
passed.
