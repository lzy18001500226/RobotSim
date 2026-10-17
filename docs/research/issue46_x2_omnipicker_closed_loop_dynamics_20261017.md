# Issue #46: X2 OmniPicker Closed-Loop Dynamics

## Result

**BLOCKED — the source-faithful X2 internal linkage is not specified sufficiently to validate a unique closed-loop mechanism.** The official URDF is a serial/tree approximation for the OmniPicker internals: it defines the two root jaw revolutes and a `-1` mimic relation, while the narrow/wide secondary links and loop links are `fixed`.

Source STL geometry supports plausible hinge axes and several pivot centers. A SIMULATION_ONLY passive-loop candidate was constructed and run with real `mj_step` dynamics. The candidate is not valid for full-X2 integration: static branch selection is discontinuous; the dynamic run misses the close endpoint, violates the root/follower source limits, and has source-mesh intersections at mid travel. No bottle/grasp experiment or full-X2 integration was run.

The incomplete tree representation is a credible contributor to the previous physics failure, but this evidence does not prove it was the sole cause. Do not treat the SIMULATION_ONLY candidate as a hardware mechanism reconstruction.

## Sources and Mechanism Comparison

| Item | Pinned X2 OmniPicker | MuJoCo Menagerie Robotiq 2F-85 |
|---|---|---|
| Source | [`X2-Ultra_omnipicker.urdf`](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf) | [`2f85.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/robotiq_2f85/2f85.xml) |
| Pin / hash | `575cc6b988f976c23550e0db85aa1e5475d3652d`; SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`; Mulan PSL v2 | `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`; SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`; BSD-2-Clause |
| Driven/coupled coordinates | `right_claw_joint [-1,0] rad`; `R_hand_wide1_joint [0,1] rad` mimics `-1 × right_claw_joint`; URDF effort/velocity fields are zero | Driver, coupler, spring-link, and follower revolutes are explicit; one fixed tendon distributes actuator effort to both drivers; a joint equality couples left/right driver roots |
| Link closure | No loop equality in URDF; secondary narrow/wide and `*_loop` joints are fixed | One `connect` per side (two total) joins follower/coupler; passive spring-link has explicit stiffness, reference angle, and damping |
| Collision policy | No MuJoCo contact-exclusion policy in the URDF | Explicit exclusions for designed adjacent linkage pairs and dedicated pad collision geometry |

The Menagerie model demonstrates the structure needed for its particular gripper, including passive coordinates, closure constraints, root coupling, spring behavior, actuator mapping, and exclusions. Its dimensions and parameters were not transferred to X2. MuJoCo's `connect` equality constrains two points to coincide; it cannot supply missing linkage topology or spring/transmission laws by itself ([MuJoCo equality reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#equality)).

## X2 Pivot Evidence

The source fixed-joint frames align with near-coaxial cylindrical features in the STL meshes. Representative retained fits:

| Feature | Source/mesh position | Fitted axis / surface residual |
|---|---|---|
| `narrow1 -> narrow2` | parent frame `(30.852, 18.551, 0) mm` | local Z; fit RMS `0.092 mm` |
| `narrow2 -> narrow3` | parent frame `(18.118, -15.740, 0) mm` | local Z; fit RMS `0.044 mm` |
| `narrow3` terminal bore | source dummy terminal `(0, -10.400, 0) mm`; fit near `(-0.055, -10.453, 0) mm` | local Z; fit RMS `0.142 mm` |
| `narrow_loop` distal bore | raw-STL refit center `(31.525, -9.755) mm` | local Z; radius `2.46 mm`; fit RMS `0.123 mm` |

These measurements support plausible pivot locations and axis lines, but are not CAD datums or pin-clearance tolerances. The URDF fixed-joint `axis` fields are not active-joint semantics. The mesh does not identify whether the features are rotating pins, fixed fasteners, slots, or bearings, or which assembly branch the hardware uses.

## Static Topology Checks

All three variants used the same source meshes and inertias:

| Variant | `nq / neq / nu` | Static result over 101 root samples, `q=0…-1 rad` |
|---|---:|---|
| Source tree control | `2 / 1 / 1` | Official fixed internal links and source root mimic |
| Loop link fixed to ground | `6 / 3 / 1` | Maximum distal closure residual `30.315 mm`; full-open target distance exceeds the two-link reach by `30.283 mm` |
| Loop link promoted to passive hinge | `8 / 3 / 1` | Best-fit closure residual at most `0.1201 mm`, but optimizer branch jumps up to `0.9624 rad` for a `0.01 rad` root increment on narrow3 and `0.8219 rad` on wide3 |

The fixed-loop branch cannot cover the source root range under the tested, mesh-derived pin pairing. Letting the loop link move restores static coincidence but adds an internal mode that the X2 source does not constrain.

## Dynamics and Collision Results

The source tree control and passive-loop candidate each ran `3000` MuJoCo steps (`6000` total), with `dt=0.001 s`, gravity `-9.81 m/s²`, and `1 s` smoothstep legs for open, close, and reopen. The same SIMULATION_ONLY root servo was used for both: `kp=0.5 Nm/rad`, `kv=0.05 Nm·s/rad`, effort cap `0.1 Nm`. No passive qpos writes occurred after rollout started.

| Measurement | Source tree control | Passive-loop candidate |
|---|---:|---:|
| Non-finite state | none | none |
| Maximum root target error | `0.8999 rad` | `0.5800 rad` |
| Close endpoint, target `0 rad` | actual `-0.8999 rad` | actual `-0.5800 rad` |
| Root/follower source-range violations | `0` steps | `324` steps; root minimum `-1.0081 rad`, follower maximum `1.0090 rad` |
| Dynamic pin-gap maximum | n/a | narrow `1.8711 mm`, wide `0.7615 mm` |
| Maximum passive step change | n/a | `0.0225 rad/step` |
| Convex-hull penetration proxy | maximum `7.9800 mm` | maximum `7.9774 mm` |
| Maximum solver iterations observed | `3` | `3` |

The run completed finite but did not achieve a validated full close/reopen cycle. Actuator effort hit the diagnostic cap repeatedly. The source provides no nonzero root effort/velocity limit, so this servo is not vendor data.

### Raw Surface Cross-Check

At closed `q=0`, nonadjacent `narrow1/loop` and `narrow2/loop` have compiled convex-hull distances near `-7.96 mm` and `-1.5 mm`, while the raw STL triangle AABBs are disjoint. These are convex-hull false-positive candidates. The adjacent `narrow3/loop` pin region has actual surface intersections and is filtered as a pin-joint contact.

At static mid travel `q=-0.5 rad`, the passive-loop candidate has raw STL triangle-surface intersections for `narrow1/loop` and `narrow2/loop`; the source tree at the same root pose has no such intersections. The tested passive branch therefore passes through original source surfaces. Raw triangle surface tests do not detect full containment; exact pair counts/witnesses are in `contact_surface_audit.json`.

## Required Human Decision

The tree model omits internal passive motion, but the tested loop candidate is not valid. A source-grounded replacement requires authoritative X2 assembly/CAD or a vendor kinematic description that specifies:

1. Which fixed-looking sublinks are genuinely revolute versus rigidly assembled.
2. The exact loop graph and which bores share each pin.
3. The missing relation that removes the passive internal mode (spring, gear/cam, tendon, or additional loop), with its rest/ratio/limits/force law if applicable.
4. Passive travel limits and the intended collision policy at the linkage interfaces.

Stop before full-X2 integration. The two research scripts have disposition `KEEP-AS-TOOL` for reproducing this diagnostic only; no production path depends on them.

## Evidence and Reproduction

Evidence is in `/home/lzy18001500226/projects/RobotSim/reviews/issue-46/x2-omnipicker-closed-loop-dynamics-20261017-run01/`. It contains the three MJCF variants, static scans, JSON/CSV/JSONL traces, raw surface audit, side-by-side GIF, and four key frames. Raw video/trace output is not committed.

See `REPRODUCE.md` for the exact MuJoCo 3.3.6 commands. The experiment modifies no pinned vendor files, prior review baselines, bottle, or full X2 workcell.
