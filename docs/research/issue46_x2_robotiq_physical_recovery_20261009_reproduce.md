# Issue #46 X2 + Robotiq Recovery Reproduction

Run from the isolated research branch at the publication commit. The raw runs recorded Git HEAD `366c8f716b2cf25042c2ab653a4a39939919ec39` while the research scripts were working-tree changes; their exact SHA-256 values are recorded in each raw result JSON and the committed branch contains those same script contents. Runtime:

```bash
export MUJOCO_GL=egl
PY=/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python
X2=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf
MENAGERIE=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie
CANONICAL=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py
OUT=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robotiq-physical-recovery-20261009
```

## Coupler source-limit probes

```bash
"$PY" scripts/research/issue46_x2_robotiq_coupler_diagnosis.py \
  --x2-root "$X2" \
  --menagerie-root "$MENAGERIE" \
  --canonical-helper "$CANONICAL" \
  --output-dir "$OUT/coupler/attempt_03" \
  --station-base-pos -0.08 0 0.68
```

The command records endpoint probes at 1 ms and 0.5 ms, gravity-off response, reset-to-interior-target response, and a constraint-consistent interior OPEN/CLOSE/REOPEN diagnostic. The latter is retained as a strict source-limit FAIL; it is not a corrected PASS.

## Isolated fixture

Unshifted reference attempt:

```bash
"$PY" scripts/research/issue46_x2_robotiq_isolated_bottle.py \
  --output-dir "$OUT/isolated-bottle/attempt_02"
```

Single trace-derived fixture placement correction and complete diagnostic dynamics:

```bash
"$PY" scripts/research/issue46_x2_robotiq_isolated_bottle.py \
  --pad-midpoint-offset-world-m -0.012 0 0 \
  --output-dir "$OUT/isolated-bottle/attempt_03"
```

The offset moves the fixed gripper pad midpoint 12 mm along world `-X`, away from the measured bottle-to-base contact. It changes neither the gripper controller nor any source/model physical parameter. The run stays classified `NON-ACCEPTANCE` because the source coupler limit fails at the first physics step. It does not count as bottle lift acceptance.

## Signed-distance X2 IK

```bash
"$PY" scripts/research/issue46_x2_robotiq_reachability.py \
  --output-dir "$OUT/reachability/attempt_04"
```

The runner evaluates three fixed, documented station/approach/grasp-height configurations with bounded, source-valid arm seeds. It stores per-evaluation poses and signed-distance data in each configuration's `raw/candidate_trace.json`. No X2 physics approach is run because none passes the static endpoint/corridor gates.

## Expected outputs

- Coupler summary: `coupler/attempt_03/result.json`; raw step traces and `full_cycle/`.
- Isolated corrected fixture: `isolated-bottle/attempt_03/result.json`, `raw/isolated_physics_trace.jsonl`, `raw/isolated_contact_trace.jsonl`, three PNGs and `isolated_grasp_lift_release.mp4`.
- IK: `reachability/attempt_04/result.json`; one raw candidate trace and pregrasp PNG per configuration.
- Consolidated summary: `result.json` in the same external evidence root.

Reproduction requires the pinned X2 and Menagerie checkouts above, the canonical helper with SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`, and MuJoCo Python/native 3.3.6 using the native library SHA recorded in the report.
