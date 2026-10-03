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
