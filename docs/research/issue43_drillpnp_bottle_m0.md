# Issue #43: DrillPnP to bottle/cylinder M0

## Scope and evidence

This is a source-level reuse analysis of `NVlabs/humanoidmimicgen` at
`d82844dcec242c82d6b82628ccc45933c3ad5cbd` (2026-09-23). Local02's successful
`LMDrillPnP90` WBC-goal playback is user-provided evidence; this report does
not claim an independent replay. No RobotSim production code or model was
changed.

## Source ownership

| Concern | Exact owner | Behavior |
|---|---|---|
| Environment registration | `humanoidmimicgen/locomanipulation/envs/base.py`: `RETAINED_LOCOMANIPULATION_ENV_NAMES`, `LocoManipulationEnvMeta`; `envs/__init__.py` imports `locomanip_pnp.py` and checks registration | `LMDrillPnP90` is explicitly retained. |
| Task and scene | `humanoidmimicgen/locomanipulation/envs/locomanip_pnp.py::LMDrillPnP90` | `_get_objects()` creates two source/target tables and one movable object; `get_object()` exposes the movable body's alias as `bottle`; the instruction correctly still says drill. `task_config()` identifies the object then the target table. |
| Initial object pose | Same `_get_objects()`; sampler applied by `humanoidmimicgen/locomanipulation/utils/scene/scene.py::Scene.reset` | `name="obj"`, `static=False`, scale 1.0; source-table-relative x `[-.28,-.24]`, y `[-.05,.05]`, yaw ±`.05π`. The sampler writes the object's free-joint qpos and qpos0. |
| Object geometry/contact | `humanoidmimicgen/locomanipulation/models/assets/objects/omniverse/locomanip/powerdrill_b01/model.xml` | Visual drill and battery meshes; collisions are simplified box/capsule geoms, including a narrow handle-like box and separate battery collision. It exposes `bottom_site`, `top_site`, and `horizontal_radius_site`. The `SceneObject`/MJCF wrapper injects a free joint for nonstatic objects. |
| Demonstration/task outcome | `humanoidmimicgen/dataset_playback.py`; criteria in `locomanipulation/utils/scene/success_criteria.py` | Playback reports both whether success occurred at any step and whether the final step succeeds. The source HDF5 has one source episode per task and stores scene XML on the demo. |

`LMDrillPnP90._get_objects()` places static `table_origin` from
`factory_ergo_table/model.xml` at nominal `[1.2, 0, 0]`, with ±.02 m planar
sampling and rotation `[-π,-π]`. Static `table_target` uses the same table model
at `[0,1.2,0]`, ±.02 m, and rotation `[-π/2,-π/2]`. The drill object is
sampled relative to `table_origin`; the task's `bottle` alias is only a task
name and does not alter the drill geometry.

## Success predicate

`_get_success_criteria()` is an `AllCriteria` over:

1. `IsStatic(obj)`: linear speed ≤ 0.002 m/s and angular speed ≤ 0.005 rad/s.
2. `IsUpright(obj, threshold=0.95)`: object world z-axis upright test, not
   symmetric.
3. `IsGripperFar(obj, threshold=0.1)`: both left and right end-effector sites
   must be farther than 0.1 m from the object's root body.
4. `IsInContact(obj, table_target)`: MuJoCo/RoboSuite object-table contact.

This is a stable final placement check, not a complete pick-and-place proof:
it does not assert source contact, lift height, source-table departure, carry,
or a grasp event. A push/fall onto the target could satisfy it. M0 evidence
should separately confirm a positive lift and transfer before final release.

## Trajectory, observations, and contact coupling

The playback loader reads each episode's `wbc_goal/navigate_cmd`,
`teleop_cmd/base_height_command`, `wbc_goal/wrist_pose`, and
`wbc_goal/target_upper_body_pose`. `G1DecoupledWholeBodyPolicy.set_goal()`
passes the 31-value upper-body pose and base-height command to the upper-body
policy, and `navigate_cmd` to the lower-body policy. Its action writes the
recorded upper-body pose directly into configured robot joint indices. The
31 values include waist/arms and 14 hand joints. The recorded `wrist_pose` is
loaded but this policy's `set_goal()` does not use it.

The RoboSuite environment is created with `use_object_obs=True`, so its raw
observations can contain object fields. `SyncEnv.observe()` then constructs a
robot-focused observation (q/dq, base, wrist, camera, and declared privileged
keys) rather than forwarding arbitrary raw object fields. In either case,
`G1HomiePolicyV2.compute_observation()` feeds joint state, floating-base
attitude/angular velocity, commands, and action history to the lower policy;
it does not read object pose. Thus the sequence is a fixed recorded body/hand
target, not an object-relative grasp controller.

The hand closure/preshape and the actual contacts are part of a human-recorded
drill demonstration. Reusing that sequence on a bottle with similar grasp
diameter, pose, mass, and friction is plausible because object pose is not an
explicit policy input, but it is not guaranteed. A cylinder lacks the drill's
handle/battery contact shape, so the narrow handle grasp may fail or slide.
Keep the bottle contact dimensions near the grasped drill section and validate
contacts, lift, transfer, and final settling in MuJoCo. The exact recorded
`navigate_cmd` values were not available in this source review; if M0 is to be
fixed-base, verify they are zero or suppress base motion explicitly rather
than assuming the full demo is stationary.

## Dataset replay trap

`dataset_playback.py::load_hdf5_dataset()` requires `demo.attrs["model_file"]`
and returns it with the frames. The playback loop calls `env.reset_to()` with
that XML; `wbc/envs/locomanipulation/sync_env.py::SyncEnv.reset_to()` reloads
it into RoboSuite before restoring the recorded MuJoCo state. Consequently,
changing only the class's `mjcf_path` still restores the drill from the source
HDF5.

For an M0 experiment, use a **derived local copy** of the HDF5 with the
embedded scene XML attribute changed to the cylinder scene, preserving the
robot, free-joint count/order, and recorded state layout. Alternatively, add
an opt-in playback setting that skips embedded model XML and ensure the task
environment has an equivalent topology before applying stored states. Do not
modify the published source dataset.

## Minimal adaptation: A vs B

**Recommend Option A for the smallest disposable M0.** Add a new cylinder
MJCF asset; in `LMDrillPnP90._get_objects()` point `name="obj"` to it and
change `_get_instruction()` to bottle/cylinder wording. Keep both table
samplers, object sampler, `get_object()` aliases, task/subtask logic, and the
success predicate. Do not overwrite the original drill asset. The cylinder
asset should keep the expected root/body structure and the three site names;
update site positions to the cylinder's real bounds. Include one dynamic free
joint through the existing wrapper and a physical collision cylinder, not a
visual-only mesh. Keep state topology identical for source-state replay.

This is fewer upstream touchpoints than **Option B**, but temporarily changes
the meaning of the named `LMDrillPnP90` benchmark. Choose B only if the drill
task must remain unchanged: add `LMBottlePnP90` in `locomanip_pnp.py`, add it
to `RETAINED_LOCOMANIPULATION_ENV_NAMES` in `envs/base.py`, preserve the
`envs/__init__.py` registration/import invariant, add its task-name mapping in
`scripts/evaluate_policy_example.py` (and applicable train/config maps), and
make the dataset's `script_config.task_name` select the sibling. B still needs
the embedded XML replay fix and the same object model.

## Exact touch list for Local02

For the minimal HMG experiment, touch only:

1. New MJCF under
   `humanoidmimicgen/locomanipulation/models/assets/objects/omniverse/locomanip/`
   (a distinct bottle/cylinder directory).
2. `humanoidmimicgen/locomanipulation/envs/locomanip_pnp.py`, class
   `LMDrillPnP90`: object `mjcf_path` and instruction only.
3. A local derived source-demo HDF5's `data/demo_1` `model_file` attribute;
   no published dataset/source commit is required. If this cannot be handled
   as a data derivative, the alternative source edit is an explicit opt-in
   skip-model-file option through `dataset_playback.py` and
   `SyncEnv.reset_to()`.

For an actual RobotSim M0, treat HMG as an experiment/reference, not a
runtime dependency. RobotSim's pinned G1 model is 29 DoF and has no actuated
hands, while HMG's recorded upper-body target has 31 channels including 14
hand joints. It cannot be copied directly; RobotSim would need its own simple
arm/end-effector command mapping or a fixed sequence compatible with its own
actuators. Keep the RobotSim object dynamic and validate physical contact,
lift, carry, release, and settling.

## Risks and reuse boundary

- The upstream task's success predicate is weaker than a demonstrated grasp.
- Fixed goals may depend on drill-handle contact geometry, and may include
  base commands; cylinder replay and fixed-base behavior need measurement.
- State replay requires identical joint/free-joint topology even if XML
  geometry changes.
- HMG's main code is Apache-2.0, while its retained loco-manipulation source
  and asset tree are documented as RoboCasa-derived MIT; its lower-body ONNX
  weights are separately licensed under NVIDIA Open Model License. The
  external source-demo dataset has its own distribution terms. Review exact
  notices before copying assets, weights, or data into RobotSim.
- Do not vendor the HMG framework, RoboSuite stack, HDF5 dataset, lower-body
  weights, or hand assets wholesale. Do not add Unity, perception,
  navigation, VLA, or locomotion for this M0.

## Pinned source links

- [Task and task registration import](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/envs/locomanip_pnp.py)
- [Retained environment registry](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/envs/base.py)
- [Scene reset and object free-joint state](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/utils/scene/scene.py)
- [Drill MJCF](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/models/assets/objects/omniverse/locomanip/powerdrill_b01/model.xml)
- [Success criteria](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/utils/scene/success_criteria.py)
- [Gripper distance helper](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/locomanipulation/utils/object_utils.py)
- [Dataset loading and model XML restore](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/dataset_playback.py)
- [WBC goal routing](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/wbc/policy/__init__.py)
- [Lower-body observation inputs](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/wbc/policy/g1_homie_policy.py)
- [HMG notices and weights/data licensing](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/THIRD_PARTY_NOTICES.md)
