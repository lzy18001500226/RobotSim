# Issue #46 Dual-Robotiq Mount Recovery: Reproduction

This reproduces the two candidate traces and the full contact audit without touching the pinned source checkouts. Use a new output folder so that the preserved run is not overwritten.

```bash
export MUJOCO_GL=egl
E=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-dual-robotiq-recovery-20261010/mount-corridor-recovery-20261010
PY=/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python
X2=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf

git -C "$X2" rev-parse HEAD
git -C /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie rev-parse HEAD
sha256sum "$E/prior-control/dual_simulation_only_margin_0p001.xml"

"$PY" "$E/source/issue46_x2_dual_robotiq_mount_recovery.py" \
  --baseline-xml "$E/prior-control/dual_simulation_only_margin_0p001.xml" \
  --platform-module "$E/source/issue46_x2_dual_robotiq_platform.py" \
  --cycle-runner "$E/source/issue46_x2_robotiq_coupler_recovery.py" \
  --controller-helper "$E/source/issue46_x2_robotiq_m0_5954a772.py" \
  --tool-urdf "$X2/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf" \
  --output-dir "$E/reproduced/candidate-run" \
  --run-dynamics

"$PY" "$E/source/issue46_x2_dual_robotiq_collision_audit.py" \
  --baseline-result "$E/prior-control/result.json" \
  --baseline-trace "$E/prior-control/right_open_close_trace.jsonl" \
  --candidate-result "$E/reproduced/candidate-run/mount_recovery_result.json" \
  --baseline-xml "$E/prior-control/dual_simulation_only_margin_0p001.xml" \
  --candidate-xml "$E/reproduced/candidate-run/candidate_b_right_local_y_roll_180_simulation_only.xml" \
  --platform-module "$E/source/issue46_x2_dual_robotiq_platform.py" \
  --cycle-runner "$E/source/issue46_x2_robotiq_coupler_recovery.py" \
  --controller-helper "$E/source/issue46_x2_robotiq_m0_5954a772.py" \
  --output-dir "$E/reproduced/contact-audit"

# Recreate the bounded contact-normal replay from the saved A/B traces.
# This stops at the listed first-contact samples; it does not rerun full cycles.
"$PY" "$E/source/reproduction/issue46_x2_dual_robotiq_contact_normal_probe.py" \
  --candidate-a-xml "$E/prior-control/dual_simulation_only_margin_0p001.xml" \
  --candidate-a-trace "$E/prior-control/right_open_close_trace.jsonl" \
  --candidate-a-steps 583 584 \
  --candidate-b-xml "$E/candidate-run/candidate_b_right_local_y_halfturn_180_simulation_only.xml" \
  --candidate-b-trace "$E/candidate-run/candidate_b/right_open_close_trace.jsonl" \
  --candidate-b-steps 277 292 316 318 320 \
  --platform-module "$E/source/issue46_x2_dual_robotiq_platform.py" \
  --cycle-runner "$E/source/issue46_x2_robotiq_coupler_recovery.py" \
  --controller-helper "$E/source/issue46_x2_robotiq_m0_5954a772.py" \
  --output "$E/reproduced/contact-audit/contact_normal_replay.json"
```

The contact-audit command exits with code `2` when it observes the expected candidate-B collision FAIL. The dynamic cycle runner's legacy `status=PASS` only means its narrow wrist-roll guard and limit checks completed; use `full_robot_gripper_contact_audit.json` as the mounted-contact verdict.

Runtime used: Python `3.10.12`, MuJoCo Python/native `3.3.6`, `MUJOCO_GL=egl`; native MuJoCo library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`. X2 commit is `575cc6b988f976c23550e0db85aa1e5475d3652d` and Menagerie commit is `0059d4335f8156206f63a35662313385f7ad6d74`. Source asset hashes are in `REPORT.md`; the run JSON records model and runtime module hashes. The run uses the existing `0.001 rad` simulation-only coupler activation margin and the exact previously authorized head/torso exclusion. SOURCE_FAITHFUL remains FAIL.

### Source Hashes

Hashes distinguish the exact files used by the preserved run from later reproduction helpers. `source/` retains the original run snapshots; `source/reproduction/` contains the current bounded reproduction and audit scripts.

| Role | File | SHA-256 |
|---|---|---|
| Candidate generation, exact executed snapshot | `source/issue46_x2_dual_robotiq_mount_recovery.py` | `bb5a646449bbffc7d10fe004cda2357c1c5c5a37746e69293a32b8e25e246c8f` |
| Platform, exact executed snapshot | `source/issue46_x2_dual_robotiq_platform.py` | `6183979f925123f1182b05f81c929bb251ec16db647094948ed2255ba7724e0b` |
| Dynamic cycle runner, exact executed snapshot | `source/issue46_x2_robotiq_coupler_recovery.py` | `2d9effdb3aa69a00dc1e4c1888764fb8c424ec78cbe7f1d23e4843638f88d752` |
| Controller helper, exact executed snapshot | `source/issue46_x2_robotiq_m0_5954a772.py` | `5954a772be1bede09b7c8f25425b61ddce809516cf316d77ecd5fd11cbbeb063` |
| Full contact audit | `source/issue46_x2_dual_robotiq_collision_audit.py` | `b311ef75d6107127f95a58ab96ffbbdef15ab69b7d85121b70fac8cbe8975085` |
| Candidate generation, reproduction snapshot | `source/reproduction/issue46_x2_dual_robotiq_mount_recovery.py` | `08645178ba97076020efb2178ab405d09062ac66bedb34c8468ead03891d4098` |
| Contact-normal probe | `source/reproduction/issue46_x2_dual_robotiq_contact_normal_probe.py` | `8f88fb279e1831d09168b37c13885c7055eeaa45e794ac7c2bcb571adc81c762` |
| Candidate B compiled XML | `candidate-run/candidate_b_right_local_y_halfturn_180_simulation_only.xml` | `95e687f0c55e0610117c3c2931355c64af64a3fc7cc20db7e08cae9e950dd737` |
| Candidate A compiled XML | `prior-control/dual_simulation_only_margin_0p001.xml` | `f99328cd16680aced60196a8ccd87ea1b875a021890b059e6d988f237e1b25ad` |

The original candidate-B output file retains `roll` in its historical filename; it is byte-identical to the corrected `halfturn` filename listed above. See `TRANSFORM_LABEL_CORRECTION.md`.

The contact-normal replay is expected to exit `0` only when all sampled contact distances and positions match the reference traces. The preserved replay returned `0`; its comparison summary is `contact-audit/contact_normal_replay.json`.
