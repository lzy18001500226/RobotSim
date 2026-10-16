# Local evidence path governance

RobotSim experiment outputs belong under the canonical WSL checkout's existing
`reviews/` tree:

```text
/home/lzy18001500226/projects/RobotSim/reviews/
```

Use the existing experiment-specific directory. The helper validates the path
before creating a run directory and emits these mappings for each output:

- `WSL_PATH`: canonical absolute Linux path.
- `WINDOWS_UNC`: `\\wsl.localhost\Ubuntu-22.04\...` path.
- `OPEN_COMMAND`: quoted PowerShell `Start-Process` command for Explorer.

## Runtime and path rules

`scripts/agent/path_governance.py` detects native Windows, WSL, and native Linux.
It accepts canonical POSIX paths in WSL and the matching Ubuntu-22.04 UNC path
in Windows. Native Windows rejects Linux `/home` and `/mnt` spellings and drive
paths such as `C:\home\...`; WSL rejects drive paths and `/mnt/c` outputs. Native
Linux can normalize POSIX paths, but output creation fails closed because this
host cannot verify the required Windows WSL UNC mapping.

Outputs outside the canonical `reviews/` tree are rejected unless a caller
explicitly passes an authorized root. Path resolution checks symlink and
junction containment. Run directories are created exclusively and receive a
small reservation marker. Existing outputs are never replaced by the normal
write functions. `write_reserved_text(..., overwrite_existing=True)` is only
for progress files in a newly reserved run directory, and uses an atomic
replacement.

```python
from pathlib import Path
from agent import path_governance

parent = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/native-omnihand-grasp-20261016")
run = Path(path_governance.create_unique_output_directory(parent, prefix="geometry-first").wsl_path)
path_governance.write_new_text(run / "result.json", "{}\n")
manifest = path_governance.write_artifact_manifest(run)
```

`artifact_paths.json` lists the verified path triple for every regular file in
the reserved run, including itself. It is generated only after the run is
finished; the manifest excludes its internal reservation marker. Do not use
this helper to migrate, rename, or clean existing evidence.

## Experiment entry points

The current X2 entry point uses `ISSUE46_GEOMETRY_EVIDENCE_DIR` as the parent
directory, not as a reusable run directory. Its default is the existing
`reviews/issue-46/native-omnihand-grasp-20261016/`. Each invocation allocates a
new `geometry-first-*` or `analyze-*` child and writes an `artifact_paths.json`.
An override outside the canonical reviews tree is rejected.

```bash
python3 scripts/research/issue46_native_omnihand_geometry_first.py --geometry-only
```

The G1 Dex3 `baseline.py` and `plan.py` similarly allocate new `baseline-*` and
`plan-*` directories under the existing
`reviews/issue-43/dex3-physical-20261016/` root. `episode.py` now requires the
exact plan directory, then allocates a fresh run directory and writes a
verified manifest when the episode ends.

```bash
python3 scripts/research/g1_dex3_physical/baseline.py
python3 scripts/research/g1_dex3_physical/plan.py
python3 scripts/research/g1_dex3_physical/episode.py \
  --run-id dex3-attempt01 --attempt 1 \
  --plan-dir /home/lzy18001500226/projects/RobotSim/reviews/issue-43/dex3-physical-20261016/plan-<timestamp>
```

The commands above document the integration; compiling or unit-testing these
entry points does not execute the experiments. Their existing physics and
rendering acceptance gates still apply.

## Task closeout

The `robotsim.task-closeout.v1` envelope may include an optional
`local_artifacts` list. Prefer one descriptor for the run's `artifact_paths.json`
rather than listing many files. Before recording a terminal outcome, the
closeout handler verifies the path triple, file type, existence, approved-root
containment, and every manifest member. A missing, forged, stale, or escaped
artifact blocks closeout. Local paths are validation inputs only and are
removed from the canonical task event; GitHub evidence continues to use durable
URLs or commit/PR references.

```json
{
  "local_artifacts": [
    {
      "WSL_PATH": "/home/lzy18001500226/projects/RobotSim/reviews/issue-46/run-20261016/artifact_paths.json",
      "WINDOWS_UNC": "\\\\wsl.localhost\\Ubuntu-22.04\\home\\lzy18001500226\\projects\\RobotSim\\reviews\\issue-46\\run-20261016\\artifact_paths.json",
      "OPEN_COMMAND": "Start-Process -FilePath 'explorer.exe' -ArgumentList @('\\\\wsl.localhost\\Ubuntu-22.04\\home\\lzy18001500226\\projects\\RobotSim\\reviews\\issue-46\\run-20261016\\artifact_paths.json')"
    }
  ]
}
```

## Applying to other Local experiments

Local01 should use the X2 entry-point pattern above. Local02 should pass the
specific generated G1 plan directory to `episode.py`; it should not reuse a
fixed `hand_geometry_plan.json` path. Local03 should import the helper from the
same checkout, allocate a unique child of its existing experiment directory,
write outputs through `write_new_text`/`write_new_bytes`, and finish with
`write_artifact_manifest`. If an experiment runs outside WSL, it must first
prove a valid Windows mapping; native Linux output remains blocked until such a
mapping is configured and tested.
