# Workflow: Add LiDAR Localization and Navigation

## Start simple
Use simulation ground truth only as an evaluation reference.

## Virtual LiDAR contract
The simulator should publish:
- `sensor_msgs/msg/PointCloud2`
- IMU if required by the estimator
- a stable sensor frame and TF

## Robot-specific sensor profiles
- G1: MID360-like profile
- X2: RoboSense E1R-like profile

The ray-casting implementation should be generic. Sensor geometry, FoV, scan pattern, rate, and noise belong in YAML configuration.

## Localization progression
1. Ground-truth pose for system integration.
2. LiDAR odometry / mapping.
3. Saved-map localization.
4. Navigation.

Do not let the task manager depend directly on a specific SLAM implementation.
