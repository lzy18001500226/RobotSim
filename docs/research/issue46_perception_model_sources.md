# Issue #46 perception-model source audit

This record pins inspection sources for the RGB-D perception workstream. It is
not an installation instruction, runtime dependency declaration, or approval
to redistribute model weights.

## Pinned local references

| Component | Local reference checkout | Revision | Intended boundary | License / prerequisite |
| --- | --- | --- | --- | --- |
| FoundationPose | `reference/NVlabs/FoundationPose` | `a1b694b83e633c2cb6115b9063d940a687759392` | RGB-D instance pose, evaluated only after valid camera intrinsics, depth and optical TF are available | NVIDIA Source Code License; non-commercial research/evaluation use; weights and CUDA extensions required |
| AnyGrasp SDK | `reference/graspnet/anygrasp_sdk` | `b8eaafc9eca7babd5208e7a5ade3c561060be4c5` | Candidate 6D parallel-jaw grasps from camera-frame point clouds | SDK requires a machine feature-ID license issued by the upstream registration process |
| GraspNet baseline | `reference/graspnet/graspnet-baseline` | `280c215129f759ed8649cb4e89fc5dfee55f4f80` | Open research fallback for RGB-D point-cloud candidate generation while AnyGrasp license is absent | Upstream states non-commercial redistribution/use conditions; weights and CUDA operators required |

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
