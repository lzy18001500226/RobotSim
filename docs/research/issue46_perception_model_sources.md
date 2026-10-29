# Issue #46 perception-model source audit

This record pins inspection sources for the RGB-D perception workstream. It is
not an installation instruction, runtime dependency declaration, or approval
to redistribute model weights.

## Pinned local references

| Component | Local reference checkout | Revision | Intended boundary | License / prerequisite |
| --- | --- | --- | --- | --- |
| FoundationPose | `reference/NVlabs/FoundationPose` | `a1b694b83e633c2cb6115b9063d940a687759392` | RGB-D instance pose, evaluated only after valid camera intrinsics, depth and optical TF are available | NVIDIA Source Code License; non-commercial research/evaluation use; weights and CUDA extensions required |
| SAM 2 | `reference/facebookresearch/sam2` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Box/point-prompted object mask refinement between detection and RGB-D pose | Official SAM 2 source and checkpoint; research runtime only pending deployment/license review |
| Ultralytics YOLO | PyPI `ultralytics==8.3.0` in an isolated cache runtime | package version `8.3.0` | Generic 2-D detector providing an optional box prompt to SAM 2 | AGPL-3.0 package; not a RobotSim runtime dependency or deployment decision |
| AnyGrasp SDK | `reference/graspnet/anygrasp_sdk` | `b8eaafc9eca7babd5208e7a5ade3c561060be4c5` | Candidate 6D parallel-jaw grasps from camera-frame point clouds | SDK requires a machine feature-ID license issued by the upstream registration process |
| GraspNet baseline | `reference/graspnet/graspnet-baseline` | `280c215129f759ed8649cb4e89fc5dfee55f4f80` | Open research fallback for RGB-D point-cloud candidate generation while AnyGrasp license is absent | Upstream states non-commercial redistribution/use conditions; weights and CUDA operators required |
| GraspNet API | `reference/graspnet/graspnetAPI` | `eb57dd2092d8dbe05312a29c3d0c22f3226efbfc` | Candidate serialization and post-processing used by the open baseline | Separate upstream API checkout; used only in the isolated baseline runtime |

## Input contract before model inference

The perception adapter must reject frames unless all are present and timestamp
compatible:

1. RGB image and RGB `CameraInfo`.
2. Depth image and depth `CameraInfo`.
3. Optical-frame transform to `base_link` at the image timestamp.
4. Depth-unit/encoding validation and an object mask or point-cloud region.

The X2 community reference expects the RGB-D topic family under
`/aima/hal/sensor/rgbd_head_front/`. Its use is a reference contract only;
the installed RAICOM runtime must be probed read-only before adoption.

## OmniPicker filter boundary

Neither FoundationPose nor AnyGrasp directly commands X2. Any candidate must
be transformed into the docking-state base frame and pass, in order:

1. OmniPicker side-grasp orientation and aperture feasibility.
2. Dual-pad/static-collision feasibility for the selected STL model.
3. Whole-body IK with official standing state, feet support and torso limits.
4. Physical MuJoCo contact/lift validation.

This avoids presenting a generic parallel-jaw candidate as an executable
OmniPicker grasp.

## Local research-runtime status (not a deployment qualification)

The three perception stages have separate, reproducible research outcomes:

1. **FoundationPose:** a CUDA 11.5-based official image initially selected
   unsupported Ada `sm_89` for its nvdiffrast JIT.  The isolated launch wrapper
   reports the binary-compatible `sm_86` target only for that JIT build; it
   does not edit source or weights.  The official one-frame mustard example
   completed on the RTX 4060 and wrote a 4x4 camera-frame pose.
2. **SAM 2:** the official tiny 2.1 checkpoint ran on CUDA 12.4 / PyTorch 2.5
   with the official image-predictor API.  Given the public example's truth
   box strictly as a model smoke-test prompt, its produced mask reached IoU
   0.9702 against that example's mask.  This is not a detector evaluation.
3. **YOLO:** the isolated GPU runtime executes, but both YOLOv8n and YOLOv8x
   misclassified the small/partial mustard example.  Therefore that sample is
   not accepted as a reliable detector-to-SAM input.  Live X2 camera data or
   a target-appropriate detector must pass its own detection gate.

For the accepted `X2-OmniPicker-STL` upright 70 mm cylinder, a static
SIMULATION_ONLY camera-overlay diagnostic also exercised the model interfaces
on an X2 scene.  With a truth-box SAM2 smoke-test prompt, SAM2 IoU was 0.7536
and FoundationPose produced a pose, but its raw centre error was 36.08 mm.
It is therefore not qualified for IK.  The task is known to use an **upright
cylinder of known 300 mm height**, so a separate, constrained depth refinement
used only its visible horizontal top plane and reduced the static centre error
to 0.104 mm.  This geometric method has an intentionally narrow claim:
known upright cylinder, visible top plane, calibrated optical transform; it is
not a general-object pose estimator and it does not make the truth-box prompt
into a detector result.

The first official GraspNet `checkpoint-kn.tar` URL returned an upstream
Google Drive quota refusal during retrieval.  AnyGrasp also remains unavailable
until its upstream license registration issues a machine license.  Neither
condition authorizes a fabricated candidate: the maintained fallback is
geometry-derived candidate generation followed by the existing OmniPicker
filter, static collision screen, and whole-body IK.  A learned GraspNet
candidate generator remains `NOT RUN` until an upstream-provided checkpoint is
available.
