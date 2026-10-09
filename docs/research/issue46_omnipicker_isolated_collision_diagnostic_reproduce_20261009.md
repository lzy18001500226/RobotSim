# Reproduction and Evidence Index

This packet is a read-only synthesis of pinned source and completed diagnostics. The source audit used the clean AgiBot checkout at commit `575cc6b988f976c23550e0db85aa1e5475d3652d`. No model or physics run was created for this closeout.

The only new generated artifact is an orthographic visualization and mesh-metric JSON for the original narrow/wide loop STLs and their convex hulls. It reads the source checkout and writes only to the specified new evidence directory. The exact command used for this run was:

```bash
/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_omnipicker_loop_hull_projection.py \
  --vendor-repo /tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d \
  --output-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-collision-diagnostic-20261009/loop-geometry
```

That output directory now exists, and the script refuses to overwrite it. For another run, use a new empty output directory such as `.../loop-geometry-rerun`.

## Existing evidence reused

The reports below contain the exact commands and raw run identities. They were not rerun for this packet.

```text
Source-faithful static diagnostic:
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_root_cause_abc.py --phase x2 \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/root-cause-abc-20261008/x2-geometry-8

Historical wrist visual/collision comparison:
/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/root-cause-abc-20261008/overnight-frame-collision-20261008/frame-1/result.json

Historical six-piece decomposition comparison:
/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/root-cause-abc-20261008/x2-geometry-8/result.json
```

To verify the pinned vendor identity and that the checkout was unmodified:

```bash
git -C /tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d rev-parse HEAD
git -C /tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d status --short
sha256sum /tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/meshes/right_wrist_roll_extend_link.stl
sha256sum /tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/meshes/right_wrist_roll_link.stl
sha256sum /tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/meshes/narrow_loop_Link.stl
sha256sum /tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/meshes/wide_loop_Link.stl
```

## Packet contents

- `REPORT.md`: verdict, source facts vs unknowns, reused counterfactual results, static/physical gate status, and maintainer evidence request.
- `wrist_visual_collision_projection.png`: original source wrist visual/collision STL projections from the prior audit, unmodified.
- `loop-geometry/loop_source_vs_convex_hull_projections.png` and `loop-geometry/loop_hull_metrics.json`: geometry-only projection and mesh metrics from the pinned source assets; no model collision assets are written.
- `x2_decomposed_tcp_centered_closeup.png`, `x2_decomposed_tcp_centered_side.png`, and `x2_decomposed_tcp_centered_closure_profile.csv`: prior diagnostic six-piece run, unmodified.
- `wrist_visual_collision_comparison.json`, `source_collision_result.json`, and `runtime_identity.json`: raw historical geometry/runtime evidence, copied without modification.
- `SHA256SUMS`: hashes of the copied packet files.

No physical X2 rollout was executed; therefore this packet contains no X2 contact-force evidence or X2 manipulation video.
