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

Do not create a native Windows `Path` from a Linux `/home/...` spelling. With a
drive-letter working directory such as `C:\Users\<user>\Desktop\Robot`, Python
resolves `Path("/home/...").absolute()` as `C:\home\...`. That path is not the
WSL checkout. The helper rejects it, but scripts that bypass the helper remain
unprotected. From PowerShell, run WSL experiments with an explicit distro and
working directory instead of passing Linux paths to native Python:

```powershell
wsl.exe --distribution Ubuntu-22.04 --cd /home/lzy18001500226/projects/RobotSim --exec /usr/bin/python3 -c "from pathlib import Path; print(Path.cwd())"
```

Pass raw canonical WSL strings to the helper in WSL. Native Windows callers
must pass the corresponding `\\wsl.localhost\Ubuntu-22.04\...` UNC path.
`describe_output_path` keeps `WSL_PATH`, `WINDOWS_UNC`, and `OPEN_COMMAND` as
its serialized schema; lowercase lookups remain available for older research
callers without adding fields to manifests or closeout records.

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

## Integration boundary

The helper protects only code that calls it. It does not intercept `Path.mkdir`,
`Path.write_text`, `open`, or output paths hard-coded by research scripts. PR
#68 changes the shared helper and closeout validation; it does not patch the
active X2/G1 research worktrees. Do not run a research entry point until its
owner has migrated its output path to a unique helper-reserved run and writes
through `write_new_text`/`write_new_bytes` or `write_reserved_text`, then
generates and verifies `artifact_paths.json`.

The `Path("/home/...")` example below is WSL-only; do not run it with native
Windows Python.

The path API remains compatible with older code that reads
`describe_output_path(...).get("windows_unc")`; persisted descriptor keys stay
uppercase. This compatibility does not make direct filesystem writes safe.

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
