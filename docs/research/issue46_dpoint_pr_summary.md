# Issue #46 — D-Point Mapping and Legal-Range Preflight

This Draft PR adds a source-traceable D-point audit, an isolated candidate preflight, and a repeatable MuJoCo 3.3.6 geometry/kinematics checker. The approved D-site positions are applied only to a derived candidate under governed `reviews/`; official URDF/STL files and model behavior remain unchanged.

- The four corrected D sites close in actual MuJoCo `mjData.site_xpos` within `0.000308 mm` across 11 source samples per branch from candidate q=0 to narrow q=-0.10 rad. Official roots remain inside their original limits, the candidate rate residual is `2.78e-16 rad` max, and root mimic residual is zero.
- The four collision-pair checks compare MuJoCo convex-hull distances with the original official STL triangles. Narrow root/loop hull contact is a confirmed proxy artifact; wide root/loop source containment remains unverified due a nonmanifold source STL. Both distal/loop source triangle surfaces intersect at every sampled pose, away from the D centers.
- `mj_step` was **not run**. The physical gate blocks on source triangle intersections and gross convex-hull contacts; no contact pairs were disabled and no meshes replaced.
- A local, clearly labeled non-physical kinematic telemetry preview and complete per-sample JSON/CSV evidence are recorded under `reviews/issue-46/dpoint-legal-motion-20261021T113855Z/`. These ignored artifacts are not committed.

Focused analysis: [`issue46_dpoint_legal_motion.md`](issue46_dpoint_legal_motion.md). Prior coordinate derivation and exact proposed site/sign mapping: [`issue46_omnipicker_dpoint_alignment.md`](issue46_omnipicker_dpoint_alignment.md) and [`issue46_dpoint_coordinate_proposal.diff`](issue46_dpoint_coordinate_proposal.diff).

The next gate is a source-supported collision disposition: an independently reviewed source-derived convex decomposition for root/loop proxy contacts, plus manufacturer clarification or authoritative collision geometry for the distal/loop STL crossings. Full commercial travel and grasp readiness remain unverified.
