# Issue #46: Ideal Parallel-Gripper Baseline

## Outcome

**FAIL — `SIMULATION_ONLY_IDEAL_PARALLEL_GRIPPER` did not sustain contact-driven load transfer.** The free 0.570 kg cylinder rose at most 16.78 mm, then slid down between the two pads. It was not held 50 mm above the table for one second. This is a failure of the bounded baseline configuration, not evidence that all ideal grippers or the source OmniPicker are infeasible.

Phase A is the required first gate. Native OmniPicker geometry candidates (Phase B) and native physical testing (Phase C) were therefore **NOT RUN**.

## Model and Conditions

- Independent opposed box-pad gripper with two prismatic coordinates; only the pads contact the bottle.
- Powered vertical slide is the only lift mechanism. Its actuator applies force to the carriage, not directly to the bottle.
- Bottle: free-jointed 70 mm diameter x 244.5 mm tall cylinder, 0.570 kg, friction `[1.4, 0.02, 0.001]`, on the accepted G1 M0 table at its canonical placement. Table dimensions, position, material and friction match `canonical_manipulation_assets.py` (SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`).
- MuJoCo Python/native 3.3.6, Python 3.10.12, WSL2, 2 ms timestep, implicitfast integrator, Newton solver, 80 iterations, gravity 9.81 m/s^2.
- Tested RobotSim snapshot: branch `research/issue46-omnipicker-p0-automation-20261011`, HEAD `079dc2f8955448da13329fb3d3ac37c9fd359711`; the runner SHA below identifies the uncommitted research script used with that snapshot.
- Pad force cap: 8 N per side. Carriage force cap: 25 N. The source gripper, source geometry, and OmniPicker actuators are not part of this model.

## Measured Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Open clearance and initial pad/bottle clearance | PASS | No pad/bottle contact at reset; 30 mm total lateral opening clearance around the 70 mm cylinder. |
| Bilateral physical contact | PASS | First contact at 0.960 s; contact persisted through the on-table hold and the later lift attempt. |
| Gripper effort and state | PASS | Peak pad effort 5.05/5.15 N, below 8 N; no saturation, no NaNs, no slide range violations in the corrected run. |
| 50 mm contact-driven lift | FAIL | Maximum bottle rise 16.78 mm. Carriage rose 45.18 mm, while bottle-to-carriage relative displacement reached 49.02 mm, predominantly vertical. |
| One-second suspended hold | FAIL | During the 1 s commanded hold, 295/500 samples had bottle/table contact. Mean vertical hand support was 3.456 N and mean table support was 2.299 N. Final bottle bottom was 0.82 mm below the table top. |

In the one-second commanded suspension phase, both pad contacts remained present for 100% of samples, with mean pad normal forces 4.716 N and 4.756 N. Those contacts did not retain the bottle vertically. Peak measured upward hand support was 5.924 N, only transiently above the 5.592 N static bottle weight; sustained support was insufficient. Maximum measured pad/bottle penetration was 2.564 mm. Maximum horizontal bottle displacement during lift and hold was 0.163 mm in x and 0.017 mm in y, so this run did not reproduce the earlier lateral-escape mode.

The measured failure is vertical slip under the commanded lift/hold. Actuators did not saturate, all states remained finite, and bilateral contact alone did not imply a load-bearing grasp. The trace does not isolate whether a slower carriage profile would retain the existing preload; no further correction or trajectory change was attempted after the single bounded closure correction.

## Bounded Correction and Integrity

Run 04 revealed that the diagnostic carriage started exactly at its custom lower stop and drifted slightly below it under gravity. The final run gives the carriage an internal 10 mm reference offset, keeping its initial hold inside its unchanged `[0, 80]` mm model range. This is a fixture initialization correction, not a vendor or source joint-limit change.

One evidence-based physical correction was tested: the commanded pad close target increased from 20 mm to 22 mm travel. Gains, force limits, pad/bottle/table friction, cylinder mass, geometry, and 0.6 s lift trajectory remained unchanged. This raised mean pad normal force but did not produce a stable pickup.

- Bottle qpos writes: 0.
- Welds/equalities: 0.
- Mocap bodies or carry: 0.
- Hidden bottle support or direct bottle force: none.
- Unexpected bottle contact pairs: none; observed contacts were the table and intended left/right pads.
- Joint-limit violations, NaNs, actuator saturation: none in the final run.

## Reproduction and Evidence

The runner is [`scripts/research/issue46_ideal_parallel_gripper_baseline.py`](../../scripts/research/issue46_ideal_parallel_gripper_baseline.py). It refuses to overwrite an existing output directory.

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011
PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 \
MUJOCO_GL=egl /usr/bin/python3 scripts/research/issue46_ideal_parallel_gripper_baseline.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-ideal-parallel-gripper-20261011-run05
```

The executed command also piped output through `tee`; because the shell pipeline did not enable `pipefail`, its shell exit code reflected `tee`. The runner's own exit code was 2 and `result.json` records `FAIL`.

Final raw packet: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-ideal-parallel-gripper-20261011-run05\`

- `result.json` SHA-256: `1ce89d482679dc30e24efedf1c4ba28bdb209d764b7a20764122b8ebb87d1a0d`
- `physics_trace.jsonl` SHA-256: `e1b89e4a43957d5a4f03024eba8cc5613404fb10f3e42d529e6846dde8d2d150`
- `ideal_gripper_lift.mp4` SHA-256: `e9363d85bf6b7638880b4ae380e4cfcb7bb9d139b121a11f84f09af692129422`
- `ideal_parallel_gripper.xml` SHA-256: `b44a29746c1a7871236f47b96ca6d086ea5f6533ebcbe8014dea6490a2ebf24e`
- `ideal_gripper_lift.mp4` was decoded after collection: H.264, 640 x 480, 25 fps, 237 frames, 9.48 s.
- Runner SHA-256 used: `2122679aaf76e93dc537ae2a2cf8852bf8553160701a4a827ee53473b724d837`
- Native MuJoCo library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`

The first three attempt directories preserve model-build/render/trace-recorder setup failures. Run 04 preserves the initial full physical run with the carriage lower-stop issue; run 05 is the corrected, complete physical trace. Earlier OmniPicker and collision-proxy evidence remain untouched.

## Disposition

Keep the runner as a reproducible diagnostic tool; keep all run directories as immutable evidence. Stop this conditional investigation at Phase A. A next attempt would require a separately authorized change to the lift trajectory or grasp preload controller, followed by the same table-separation and 50 mm suspended-hold gates. Do not infer native OmniPicker contact compatibility from this idealized-gripper result.
