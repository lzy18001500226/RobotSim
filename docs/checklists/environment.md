# Environment Checklist

## Host and WSL

- [ ] Windows 11 host available
- [ ] WSL2 Ubuntu 22.04 available
- [ ] Project located in the WSL Linux filesystem
- [ ] `~/projects/RobotSim` is the active repository
- [ ] WSLg GUI applications can open
- [ ] OpenGL hardware acceleration uses the NVIDIA GPU for the Unity experiment

## Git and GitHub

- [ ] Git repository initialized on branch `main`
- [ ] GitHub `origin` remote configured
- [ ] GitHub SSH authentication works
- [ ] SSH over port 443 works if port 22 is blocked
- [ ] No machine-specific secrets are tracked

## Docker

- [ ] Docker Desktop WSL integration works
- [ ] `docker version` works inside WSL
- [ ] `docker run --rm hello-world` works
- [ ] NVIDIA GPU is visible inside a CUDA container when required

## ROS 2

- [ ] ROS 2 Humble baseline selected
- [ ] ROS domain configuration is documented
- [ ] Required ROS dependencies are reproducible through apt / rosdep / Docker

## MuJoCo

- [ ] A MuJoCo release is selected and pinned
- [ ] The pinned release is recorded in project documentation
- [ ] MuJoCo can load a minimal model
- [ ] MuJoCo can step physics without fatal errors
- [ ] The selected MuJoCo native library matches the Unity plugin version when the embedded Unity integration is used

## Unity

- [ ] Unity Hub launches under the selected development environment
- [ ] Unity authentication works
- [ ] A compatible Unity Editor version is installed
- [ ] Unity can start with hardware-accelerated OpenGL under WSLg
- [ ] A minimal 3D project runs stably before adding MuJoCo
- [ ] MuJoCo Unity integration is tested separately before adding G1

## Robot Backends

- [ ] Unitree G1 upstream source versions are pinned
- [ ] G1 model loads in the selected MuJoCo baseline
- [ ] AgiBot X2 remains a planned second backend
- [ ] Vendor files are not modified in place

## Reproducibility

- [ ] Third-party commits are recorded in `third_party/LOCK.md`
- [ ] Machine-specific Unity files remain under ignored local paths
- [ ] Generated build artifacts are ignored
- [ ] Setup steps are documented
- [ ] A clean-machine or clean-container reproduction path exists
