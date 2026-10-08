# Issue #46 X2 OmniPicker long-goal geometry gate

Status: **BLOCKED at static geometry feasibility**

The authorized three distinct right-arm geometry candidates were exhausted before contact or physics. None established a collision-free, source-limited pregrasp corridor around the canonical bottle at the fixed accepted station. This does not prove global infeasibility over all joint configurations.

## Scope and provenance

- Task branch: `codex/issue46-omnipicker-1dof-m0-20261007`
- Starting RobotSim HEAD: `46d3dfb5c4845ac6904b62a5548aa4820d4d5df5`
- X2 source: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`
- Runtime: Python 3.10.12, MuJoCo Python/native 3.3.6, NumPy 1.26.4, `MUJOCO_GL=egl`
- Native library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`
- Canonical scene: G1 table and existing 70 mm x 244.5 mm, 0.570 kg X2 bottle; scene assets and placements were not changed.
- Physics steps: `0`; object state writes: `0`; controller/source limits and collision geometry were not modified.

## Candidate results

| Candidate | Geometry choice | Result |
| --- | --- | --- |
| 1 | Fixed wrist orientation, compiled jaw-gap target at bottle body center | Best recorded target residual was 42.248 mm; jaw-axis angular error was about 0.403 rad. The next proposed pose intersected the bottle shoulder with the right wrist-yaw collision geometry by 9.095 mm. Rejected. |
| 2 | Horizontal jaw-axis constraint, elbow seed `-0.10 rad` | Best recorded target residual was about 271.531 mm. Improving steps were blocked by the canonical front-left table leg against the right wrist-roll and narrow-jaw collision geometry. Rejected. |
| 3 | Horizontal jaw-axis constraint, elbow seed `-1.00 rad` | The source-limited seed itself was invalid: table-top/right-wrist-roll penetration 41.774 mm, front-left-leg/right-wrist-roll penetration 52.625 mm, and front-left-leg/narrow-jaw penetration 2.156 mm. Solver target residual was 297.544 mm. Rejected. Full zero-step trace is preserved externally. |

Candidates 1 and 2 were recorded in the preceding experiment session, but their raw per-iteration traces were not present in the recovered scratch outputs. Their figures above are retained as rounded screening results, not represented as independently reproducible raw evidence. Candidate 3 has its complete machine-readable solver trace and scene screenshots.

## Gate decision

The fixed-station geometric gate failed within the three-candidate budget. No fixed-object contact, bottle approach, hold, or lift was attempted. There is no physical grasp result. The next experiment requires maintainer authorization to change the X2 base-to-table station placement or use a different right-arm approach side. Do not infer that the gripper/controller failed: the work stopped before a valid collision-free target was reached.

## Evidence and reproduction

External packet: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-long-goal\`

- `geometry-candidate-3/candidate-3-zero-step.json`: full zero-step solver/contact trace.
- `geometry-candidate-3/candidate-3-flexed-open-seed.png` and `canonical-scene-reset.png`: static views only.
- `geometry-candidate-3/runtime_identity.json` and `experiment_commands.txt`: pinned runtime and command.
- Candidate 1 and 2 raw traces: unavailable in recovered scratch outputs.

Candidate 3 reproduction command:

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_omnipicker_geometric_feasibility.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-long-goal/geometry-candidate-3
```
