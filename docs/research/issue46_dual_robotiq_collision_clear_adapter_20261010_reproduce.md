# Reproduction: Issue #46 Right Robotiq Clearance Study

Run inside WSL Ubuntu-22.04 with the pinned runtime used for the evidence:

```bash
task_python=/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python
task_repo=/tmp/robotsim-issue46-dual-robotiq-recovery-20261010-closeout
task_evidence=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-dual-robotiq-recovery-20261010/collision-clear-adapter-20261010
task_input="$task_evidence/inputs"
task_script="$task_repo/scripts/research/issue46_x2_robotiq_standoff_adapter.py"

"$task_python" -m py_compile "$task_script"

run_geometry_case() {
  task_case="$1"
  shift
  "$task_python" "$task_script" \
    --baseline-xml "$task_input/dual_simulation_only_margin_0p001.xml" \
    --reference-left-trace "$task_input/left_open_close_trace.jsonl" \
    --helper "$task_input/issue46_x2_robotiq_m0_5954a772.py" \
    --platform "$task_input/issue46_x2_dual_robotiq_platform.py" \
    --runner "$task_input/issue46_x2_robotiq_coupler_recovery.py" \
    --output-dir "$task_evidence/$task_case" \
    "$@"
}

run_geometry_case insertion-axis --direction-source insertion
run_geometry_case contact-normal --direction-source saved-contact-normal
run_geometry_case worst-envelope-normal --direction-source worst-envelope-normal
run_geometry_case translation-solver --direction-source collision-clear-solve
run_geometry_case translation-solver-75mm \
  --direction-source collision-clear-solve --max-translation-mm 75
```

The script sets `MUJOCO_GL=egl` by default. It replays no physics: the retained left finger qpos trace is copied onto the right gripper while the right X2 arm remains fixed, and compiled collision geometry is measured with `mj_geomDistance`. The 3D translation solver evaluates only the listed critical frames. A result is not considered clear unless it satisfies those constraints and the complete retained 1,700-step envelope. No candidate passed, so no new actuator-driven cycle command exists for this adapter.

The starting repository HEAD was `1eb9f2f1671a209c5873094293f161b926f1a31b`; the final research commit is recorded in the closeout metadata. The generated inputs and the exact imported helper snapshots are preserved in `inputs/`; raw geometry outputs are in the named case directories.
