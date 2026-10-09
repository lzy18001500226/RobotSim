# Issue #46 X2 + Robotiq Reachable Grasp Checkpoint

**Outcome:** `BLOCKED — no collision-free, source-limited pregrasp was found in the bounded candidate set.` This is not evidence of global X2 + Robotiq infeasibility. Bottle contact and lift were not run.

## Provenance

- Research branch: `research/issue46-x2-robotiq-m0-20261009`
- Source HEAD at experiment launch: `aaa2570acb7381207532f97f9ba3bbcc7d8bd831`; the research runner/solver files listed in the evidence packet were modified in that worktree and their exact hashes are recorded there.
- X2: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, Mulan PSL v2.
- Menagerie: `google-deepmind/mujoco_menagerie`, commit `0059d4335f8156206f63a35662313385f7ad6d74`, Robotiq `robotiq_2f85/2f85.xml`, BSD-2-Clause.
- Runtime: Python `3.10.12`; MuJoCo Python/native `3.3.6`; `MUJOCO_GL=egl`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- X2 MJCF SHA-256: `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292`.
- Menagerie XML SHA-256: `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`.
- Canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Canonical bottle/table definitions and source physics were unchanged. Bottle: 70 mm body diameter, 244.5 mm overall height, 0.570 kg. The tabletop top is at `z=0.800 m`; its positive-Y edge is `y=0.100 m`.
- No vendor checkout, production implementation, bottle, table, gripper geometry, source joint range, actuator cap, or tendon/equality topology was changed.

Raw machine-readable evidence, renders, trace files and reproduction commands are in:
`C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robotiq-reachable-contact-20261009\`.

## A. Head/Torso Exception

At the same neutral head pose, the pinned X2 source and mounted model had the same sole self-contact: `head_pitch_link` / `torso_link`, signed contact distance `-0.007112461424559683 m`. Both body transforms and contact position matched exactly (maximum delta `0`). The one authorized experimental exclusion removes that pair only; the excluded model had no remaining robot self-contact at initialization. The pinned source and its geoms were not edited. This remains a `SIMULATION_ONLY` exception, not a source-faithful collision pass.

## B. Coupler Endpoint Check

The unchanged source ranges are `[-1.57, 0] rad`; both coupler `qpos0` values are exactly the upper hard stop `0`, with initial velocity zero. In the retained original 1,900-step cycle, the first-step equality position residual was zero to recorded precision, actuator force was zero, while constraint generalized force was approximately `+0.0129 Nm` and coupler acceleration approximately `+12.6 rad/s^2`. The first post-step qpos was positive (`+1.26456e-5` right, `+1.25720e-5` left). The maximum upper-limit excursion was `+0.000150054301 rad` and persisted through the cycle. This is constraint-driven at the endpoint, not a command outside the range.

One authorized initialization correction was tested: initialize each coupler at `-0.001 rad`, keep OPEN actuator control at `0`, and make no rollout qpos writes. Before stepping it was within source ranges and collision-free, but it introduced equality position residual `4.7605570003272923e-5` and outward acceleration `41.7224` / `42.5423 rad/s^2`. Both couplers crossed the upper limit at step 9; maximum excursions worsened to `+0.000439165170` / `+0.000438739917 rad`. The correction was rejected and not adopted.

The same cycle still demonstrated functional motion: open / closed / reopened pad-center gaps were `93.162683 / 14.690997 / 86.245835 mm`; the functional open-close-reopen motion gate passed. Peak gripper actuator effort was `0.676788183` against the unchanged `5 N` cap. There were no bottle contacts, no robot self-contacts after the exact exclusion, and no post-start qpos writes. Strict source-limit compliance **failed**; the corrected cycle is not a clean source-limit pass.

## C. Grasp Frame and Solver Audit

The TCP is the compiled `rq_m0_tcp` site at the midpoint of the actual compiled pad centers, not the wrist origin. The measured OPEN pad centers are at `-0.0467 / +0.0467 m` on the TCP opening axis: center separation `0.0934 m`. The target is the center of the compiled canonical `bottle_body` cylinder (radius `0.035 m`). Thus the open pad-center radial clearance at a centered target is `0.0117 m` per side before closure. The selected target heights (`0.8425`, `0.8775`, `0.9125 m`) are inside the cylindrical-body vertical span (`0.800–0.955 m`). The 30-degree pitch is applied around the compiled jaw-opening axis, preserving a horizontal opposing-pad axis while raising the 120 mm pregrasp by `60 mm`.

The source-limit check in `arm_metadata` verified the compiled seven right-arm joint ranges against the pinned URDF. Optimizer variables were bounded by those ranges. Position-only FK reproduced the requested TCP point to `1.3e-15–3.4e-14 m`; this confirms the TCP/FK mapping, but those exact-position seeds collided and are not valid solutions. The orientation-only wrist seed was initially computed but not used in the full-pose solve; the corrected bounded run used it as a third full-pose seed alongside neutral and position-only seeds. No full-pose seed met the position (`3 mm`) and orientation (`0.02 rad`) gates.

The existing solver's contact penalty uses active MuJoCo contacts. Post-processing with `mj_geomDistance` over compiled active robot geoms against every table and bottle geom caught signed negative clearances that were absent from its active-contact gate. Therefore the post-processed exact distances below, rather than the solver's contact-only boolean, govern collision interpretation.

## D. Bounded Static Candidates

All candidates started with zero robot/table, robot/bottle and robot self-contacts and no arm limit violations. Exact joint poses and all optimizer iterations are in each `raw/candidate_trace.json`; exact signed distances are in `raw/candidate_clearances.json`.

| Candidate | Target / pregrasp TCP (m) | Best full-pose error (position / orientation) | Exact selected-pregrasp minimum distance | Outcome |
|---|---|---|---|---|
| `outboard_station_plus_x_upper_body` | `[0.300, 0, 0.9125]` / `[0.196077, 0, 0.9725]` | `0.311133 m / 0.603015 rad` | table `-0.032925 m`; bottle `+0.119078 m` | Not a valid pose; exact table distance shows 32.9 mm penetration. |
| `outboard_station_minus_y_body_center` | `[0.300, 0, 0.8775]` / `[0.300, 0.103923, 0.9375]` | `0.052319 m / 0.179674 rad` | table `+0.057470 m`; bottle `+0.030356 m` | Closest clear full-pose seed, but misses both pose gates. |
| `outboard_station_plus_x_lower_body` | `[0.300, 0, 0.8425]` / `[0.196077, 0, 0.9025]` | `0.183291 m / 0.704491 rad` | table `0 m`; bottle `+0.081585 m` | No positive table clearance and misses both pose gates. |

The position-only exact-target seeds were not feasible: their minimum table clearances were `-0.010765`, `-0.030981`, and `-0.077132 m`; the body-center opposite-side seed also penetrated the bottle by `-0.071630 m`, involving the wrist/gripper-base structures rather than intended pad contact. These contacts falsify those position-only seeds. The upper-body selected full-pose seed likewise had a `-0.032925 m` table overlap. The middle candidate is the best recorded state, but its 52.3 mm / 0.180 rad TCP mismatch means no pregrasp or approach path was accepted.

## Gates A–H

| Gate | Result | Evidence |
|---|---|---|
| A. Source-versus-mounted head/torso comparison | **PASS for identity of the pre-existing pair** | Exact pair, depth and transforms match; see prior comparison JSON referenced by the evidence packet. |
| B. Exact one-pair experimental exclusion | **PASS, SIMULATION_ONLY** | Only `head_pitch_link` / `torso_link` excluded; no additional exclusions. |
| C. Mounted gripper functional OPEN/CLOSE motion | **PASS** | Pad-gap and actuator-force measurements above. |
| D. Strict Robotiq source position limits | **FAIL** | Both coupler joints exceed upper stop; one startup correction worsened the peak. |
| E. Collision-free source-limited PREGRASP | **FAIL / not demonstrated** | No candidate met both TCP pose and exact-clearance requirements. |
| F. Dynamic APPROACH | **NOT RUN** | Blocked at E. |
| G. Bilateral bottle contact, HOLD, and 1/5/30 mm lift | **NOT RUN** | No accepted approach. No bottle physics was started. |
| H. Transfer/release | **NOT RUN** | Outside reached gates. |

No bottle qpos writes, welds, equality attachments, mocap, follow-hand carry, hidden supports, or left-hand assistance occurred. Static IK qpos assignments were confined to zero-step solver evaluation and are not rollout state forcing.

## Stop Reason

The stop is at the first unresolved physical gate: no source-limited, collision-free TCP pregrasp was found in the bounded set. The 3-candidate search is not a proof of global infeasibility. The contact-only collision cost also needs a more complete exact-distance formulation before its boolean can be relied on for future candidate selection. No additional station/pose search or controller tuning was performed.
