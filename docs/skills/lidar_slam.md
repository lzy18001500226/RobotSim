# Technical note: LiDAR Localization

Design principles:
- sensor simulator outputs ROS 2 point clouds
- scan pattern is configured, not hard-coded
- G1 and X2 use the same bridge implementation with different profiles
- ground truth is stored for error evaluation
- SLAM/localization is a replaceable backend

Potential G1 route:
MID360-like point cloud -> LiDAR odometry / FAST-LIO family -> map / odom.

Do not couple the whole project to one SLAM package.


## X2 grasp-integration scope (2026-11-07)

The prior Issue80 scan-registration math is reused in
scripts/research/issue46/perception/lidar_scan_registration.py, pinned to Git
70a84cd. A fixed-landmark synthetic test proves that current-scan to previous-scan
registration is the **positive** sensor pose increment. The old caller's inverse
update must not be reused. Body pose composition also includes both measured
sensor extrinsics, so waist motion and the sensor lever arm do not masquerade
as base translation. These are offline frame tests only.

The current full-robot grasp implementation still uses native base truth.
RGB-D object estimation does not establish LiDAR base localization. In particular,
feeding the simulator's truth-derived MC odometry into a LiDAR/odom fusion loop
does not satisfy the truth-ablation requirement. Preserve the older map and
ray-caster evidence as read-only; verify map geometry, optical axes, local-frame
point clouds, timestamps and actual localization error before a live handoff.
No SLAM/Nav2 work is authorized by this math correction.
