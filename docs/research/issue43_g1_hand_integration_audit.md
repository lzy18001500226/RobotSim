# Issue #43: G1 actuated-hand integration audit

## Scope and pinned baseline

Read-only source audit. RobotSim is pinned to `unitree_mujoco`
`1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`; its
`unitree_robots/g1/g1_29dof.xml` has 29 actuated body joints plus the free
root (`nq=36`, `nv=35`, `nu=29`). The terminal `left_wrist_yaw_link` and
`right_wrist_yaw_link` each have a fixed rubber-hand visual mesh, but no hand
joints, actuator, or hand collision model. The hand visuals are disabled for
collision (`contype="0"`, `conaffinity="0"`).

The current pinned checkout contains files named like Unitree finger meshes,
but they are not the same mesh revisions as the compatible G1-with-hand
source below: SHA-256 checks for the palm, thumb, middle, and index meshes
showed differences between `unitree_mujoco@1eb6642` and
`unitree_ros@5994d4f`. Do not combine one revision's kinematics with the
other revision's same-named meshes without validation.

## Candidate comparison

| Candidate and exact source | License for code/assets | DoF, joints, actuation, contact and coupling | Fit without replacing G1 body |
|---|---|---|---|
| **Unitree Dex3-1**, `unitreerobotics/unitree_ros@5994d4faef0a9cadd3287f8de0199a67eeb2a259`; integrated reference `robots/g1_description/g1_29dof_with_hand_rev_1_0.xml`, standalone `robots/dexterous_hand_description/dex3_1/dex3_1_{l,r}.urdf` | Unitree repository BSD-3-Clause for source/model and meshes. Keep the Unitree copyright, license/disclaimer in source and binary distributions; do not imply Unitree endorsement. | 7 revolute DoF per hand; 14 bilateral. Names: `{left,right}_hand_thumb_{0,1,2}_joint`, `{left,right}_hand_middle_{0,1}_joint`, `{left,right}_hand_index_{0,1}_joint`. The G1 MJCF uses independent `<motor>` actuators; torque bounds are 2.45 Nm at thumb 0 and 1.4 Nm at the other hand joints. No tendon/equality coupling. Palm and finger segment meshes have collision geoms; thumb 1 also has a box collision. There are no named fingertip sites/geoms. | **Yes.** The source MJCF puts the palm/fingers under the existing `*_wrist_yaw_link`. One-hand graft adds 7 joints; both hands add 14. This is the most complete official MJCF hand source. |
| **HumanoidMimicGen G1 three-finger hand**, `NVlabs/humanoidmimicgen@d82844dcec242c82d6b82628ccc45933c3ad5cbd`; `humanoidmimicgen/wbc/robot_model/model_data/g1/g1_29dof_with_hand.urdf` | HMG NVIDIA-authored source is Apache-2.0; its `THIRD_PARTY_NOTICES.md` identifies the G1 description/meshes as Unitree BSD-3-Clause. Preserve both applicable notices. | Same 7 revolute joints per hand and Unitree mesh family. URDF has inertias, visual/collision meshes and limits, but no MuJoCo actuator definitions, tendon/equality constraints, or fingertip sites. HMG's WBC carries hand joint targets; the URDF alone is not the WBC or an actuator interface. | **Yes, after conversion**, but it is URDF-only and brings no MJCF actuator definitions. It offers no advantage over using the official Unitree MJCF hand subtree directly. |
| **MuJoCo Menagerie Unitree G1 with hands**, `google-deepmind/mujoco_menagerie@4d038b3feae26ec82b46a4d586379114012a8ac7`; `unitree_g1/g1_with_hands.xml` | The model folder has its own Unitree BSD-3-Clause `unitree_g1/LICENSE` covering its G1 model/assets. Retain that folder's notice. | Same 7 DoF per hand and joint names. It provides 14 named `<position>` actuators, default `kp=500`, `dampratio=1`, and inheritrange; these gains are a model default and need tuning for RobotSim. Hand collision is represented by mesh geoms and the thumb-1 box. No named fingertip sites, tendons, or equalities. | **Yes.** It is a complete MuJoCo example from which to extract only a hand subtree and matching meshes/actuators. Do not replace RobotSim's full body with the Menagerie full-body MJCF. |
| **Unitree Dex1-1**, `unitreerobotics/unitree_ros@5994d4faef0a9cadd3287f8de0199a67eeb2a259`; integrated `robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf` | Unitree BSD-3-Clause. Same notice conditions as Dex3. | 2 prismatic hand DoF each (4 bilateral): in the standalone URDF `Joint1_1`, `Joint2_1`; in the G1 composite, `left/right_dex1_finger_joint_{1,2}`. Opposite finger axes give a pinch; no mimic/equality relation. The composite URDF supplies visual and collision meshes, but no MuJoCo actuators or fingertip sites. | **Yes.** The composite attaches a fixed hand base to `*_wrist_yaw_link` at xyz `[0.0415,0,0]`. It has fewer DoF than Dex3, but converting the URDF and supplying actuator/control behavior is additional work. No Dex1 hand command example/interface was found in the pinned SDK2/ROS2 sources. |
| **LEAP hand**, `google-deepmind/mujoco_menagerie@4d038b3feae26ec82b46a4d586379114012a8ac7`; `leap_hand/{left,right}_hand.xml` | The hand folder includes its own MIT license for model and assets. Preserve that license. | 16 revolute DoF per hand. Order: `if_mcp, if_rot, if_pip, if_dip`; `mf_mcp, mf_rot, mf_pip, mf_dip`; `rf_mcp, rf_rot, rf_pip, rf_dip`; `th_cmc, th_axl, th_mcp, th_ipl`. Sixteen position actuators. Many named collision boxes and named fingertip contact geoms `if_tip`, `mf_tip`, `rf_tip`, `th_tip`; no tendon/equality coupling. | Can be mounted to a G1 wrist, but is not a Unitree hand: palm orientation/mount pose and 16-channel control need adaptation. It is not a smaller G1 augmentation. |
| **Unitree Inspire DFQ/FTP** (additional official option), `unitreerobotics/unitree_ros@5994d4faef0a9cadd3287f8de0199a67eeb2a259`; `robots/g1_description/inspire_hand/{DFQ,FTP}_left_hand.urdf` and G1 combined URDFs | Unitree BSD-3-Clause at repository root. | 12 revolute joints per hand. DFQ names in URDF order: `L_thumb_proximal_yaw`, `L_thumb_proximal_pitch`, `L_thumb_intermediate`, `L_thumb_distal`, `L_index_proximal`, `L_index_intermediate`, `L_middle_proximal`, `L_middle_intermediate`, `L_ring_proximal`, `L_ring_intermediate`, `L_pinky_proximal`, `L_pinky_intermediate` (joint names use `_joint`). Six joints declare URDF `mimic` relationships. Mesh collisions are included; no MJCF actuator block or fingertip sites in these URDF files. | Possible to attach using the official combined G1 URDF, but URDF mimic joints need explicit MuJoCo equalities/control mapping. More joints and coupling work than Dex3/Dex1; not an M0 shortcut. |

## Dex3 order and supported command interface

The Unitree `unitree_ros` G1-with-hand MJCF actuator declaration order is:

- Left: thumb 0, 1, 2; middle 0, 1; index 0, 1.
- Right: thumb 0, 1, 2; index 0, 1; middle 0, 1.

The Unitree SDK's Dex3 command order is the same for both hands but is
different from the left MJCF actuator order: `thumb_0, thumb_1, thumb_2,
index_0, index_1, middle_0, middle_1`. Map by explicit joint name; never
assume MuJoCo actuator index equals DDS slot. SDK2's
`example/g1/dex3/g1_dex3_example.cpp` uses seven `MotorCmd` entries per hand
with `mode`, `q`, `dq`, `tau`, `kp`, and `kd` fields. It publishes to
`rt/dex3/{left,right}/cmd` and subscribes to
`rt/lf/dex3/{left,right}/state`. The pinned `unitree_ros2@668d1ec5` provides
`unitree_hg/msg/HandCmd`, `HandState`, and a ROS example with topics
`/dex3/{left,right}/cmd` and `/lf/dex3/{left,right}/state`. The ROS2 Dex3
example target requests C++20.

These are separate hand channels, not extra elements in the G1 body
`LowCmd`/`LowState` 29-joint map. The pinned RobotSim `unitree_mujoco` model
and bridge do not contain the hand joints or this `HandCmd` hand path; adding
the hand MJCF does not automatically add DDS-to-MuJoCo hand control.

## Exact attachment and M0 recommendation

**Recommend one right Dex3-1 hand for the first bottle/cylinder M0.** It adds
one physically modeled, independently actuated three-finger hand, keeps the
RobotSim body unchanged, and has both a same-family G1 MJCF source and a
documented Unitree command schema. Use the hand subtree from the official
Unitree Rev. 1.0 MJCF at the pinned source URL below; the Menagerie model is a
useful independent MuJoCo comparison. A single Dex1-1 is the lower-joint
alternative only if the M0 deliberately accepts a simple pinch and does not
need a currently documented Unitree hand command path.

In RobotSim's pinned `g1_29dof.xml`, graft beneath `right_wrist_yaw_link`.
The existing visual rubber hand is at local `[0.0415,-0.003,0]`; the official
Dex3 palm uses that same hand datum. The integrated source MJCF's digit roots
are in the wrist-yaw frame: thumb `[0.067,-0.003,0]`, middle
`[0.1192,-0.0046,-0.0285]`, and index `[0.1192,-0.0046,0.0285]`. Replace or
disable the visual-only rubber-hand geom to avoid a duplicate palm. Keep the
hand chain's collision geoms and inertias with its matching source meshes.
There are no source fingertip sites; add RobotSim-owned named fingertip sites
only if M0 tests/controllers need them.

The smallest hand-only MJCF delta is:

1. Reference/add the matching source palm/finger mesh assets and hand default
   parameters under the license noted below; do not combine meshes from
   `unitree_mujoco@1eb6642` with the Rev. 1.0 hand MJCF because tested same-name
   mesh hashes differ.
2. Replace the right visual rubber-hand geom with the collision-enabled palm
   and add the three finger chains: thumb 3 joints, index 2, middle 2.
3. Add seven actuators by joint name. If keeping the official MJCF motor model,
   drive bounded effort/PD outputs; if using a MuJoCo position actuator for a
   scripted M0, tune and bound its gains/force. Do not copy the Menagerie
   `kp=500` default without measurement.
4. Keep all existing 29 body actuators first/in the same explicit name map;
   do not change the body `LowCmd` list. A right hand adds 7 scalar joints and
   7 actuators; bilateral Dex3 adds 14. With the floating base, expected model
   dimensions become `(nq,nv,nu)=(43,42,36)` for one hand or `(50,49,43)` for
   two. Joint insertion under the left/right wrist changes later qpos/joint
   numeric addresses, so consumers must resolve joints by name.

## Redistribution and integration risks

- Unitree source and Menagerie `unitree_g1` hand assets are explicitly marked
  BSD-3-Clause. Redistribution requires retaining the source notice/license
  and reproducing it in binary documentation; Unitree's name cannot endorse a
  derived product. The RobotSim third-party lock also cautions that not every
  robot asset's redistribution status should be inferred from a repository
  root license. Before checking meshes into RobotSim, confirm the exact asset
  grant and add its source/license record. Keeping a pinned ignored vendor
  checkout is a different distribution choice from committing copied meshes.
- HMG's wrapper/license does not turn its Unitree meshes into Apache-only
  assets; its own third-party notice lists those robot files as BSD-3-Clause.
- Dex3 has no explicit tendon/equality coupling or named fingertip site; add
  names only as RobotSim-owned metadata, not as an upstream claim.
- Adding joints changes `nq`/`nv`, model state snapshots, and right-side
  numeric joint addresses. It must not silently reinterpret the existing 29
  body command/state slots.
- The body stays at 29 DoF, but MuJoCo `nu` becomes 36 for a single hand or 43
  bilateral. Test code and state contracts that assumed `nu=29` need a
  hand-aware contract before enabling this overlay.
- `unitree_mujoco` at the current pin does not simulate Unitree hand DDS.
  Unitree's hand message and topic surface is documented separately; using it
  for a future hardware path is not evidence that the local simulator bridge
  already supports it.
- No simulation, hardware, or licensing counsel review was performed in this
  audit; model compatibility here is a source/frame analysis.

## Pinned source links

- [RobotSim lock file](https://github.com/lzy18001500226/RobotSim/blob/11aebeb60a1d3c1c2f580c8f4d6bd6907581fd33/third_party/LOCK.md)
- [Pinned 29-DoF Unitree MuJoCo model](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1/g1_29dof.xml)
- [Unitree G1 Rev. 1.0 with-hand MJCF](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml)
- [Unitree hand URDF and meshes](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/dexterous_hand_description/dex3_1)
- [Unitree repository BSD-3-Clause license](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/LICENSE)
- [Unitree Dex1-1 integrated G1 URDF](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf)
- [Unitree Inspire DFQ G1 hand URDF](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/inspire_hand/DFQ_left_hand.urdf)
- [Unitree SDK2 Dex3 example](https://github.com/unitreerobotics/unitree_sdk2/blob/63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36/example/g1/dex3/g1_dex3_example.cpp)
- [Unitree G1/Dex3 joint order](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1/g1_joint_index_dds.md)
- [Unitree ROS2 hand message](https://github.com/unitreerobotics/unitree_ros2/blob/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88/cyclonedds_ws/src/unitree/unitree_hg/msg/HandCmd.msg)
- [Unitree ROS2 Dex3 example](https://github.com/unitreerobotics/unitree_ros2/blob/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88/example/src/src/g1/dex3/g1_dex3_example.cpp) and [its C++20 target declaration](https://github.com/unitreerobotics/unitree_ros2/blob/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88/example/src/CMakeLists.txt)
- [HumanoidMimicGen G1 hand URDF](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/wbc/robot_model/model_data/g1/g1_29dof_with_hand.urdf)
- [HumanoidMimicGen Unitree asset notice](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/THIRD_PARTY_NOTICES.md)
- [Menagerie G1 with hands and per-model license](https://github.com/google-deepmind/mujoco_menagerie/tree/4d038b3feae26ec82b46a4d586379114012a8ac7/unitree_g1)
- [Menagerie Unitree G1 BSD-3-Clause license](https://github.com/google-deepmind/mujoco_menagerie/blob/4d038b3feae26ec82b46a4d586379114012a8ac7/unitree_g1/LICENSE)
- [Menagerie LEAP hand and MIT license](https://github.com/google-deepmind/mujoco_menagerie/tree/4d038b3feae26ec82b46a4d586379114012a8ac7/leap_hand)
- [Menagerie LEAP hand MIT license](https://github.com/google-deepmind/mujoco_menagerie/blob/4d038b3feae26ec82b46a4d586379114012a8ac7/leap_hand/LICENSE)
