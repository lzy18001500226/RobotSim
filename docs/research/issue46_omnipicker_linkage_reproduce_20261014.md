# X2 OmniPicker Linkage Candidate Reproduction

This is a SIMULATION_ONLY fitted-pin constraint diagnostic. It is not a source-faithful physical mechanism validation.

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-linkage-physics-20261014
/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  scripts/research/issue46_omnipicker_linkage_constraint_validation.py \
  --source-xml /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-g2-omnipicker-compatibility-20261013/x2_g2_omnipicker_review.xml \
  --source-urdf /home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf \
  --g2-xacro /home/lzy18001500226/.cache/robotsim/upstreams/genie_sim_robot_model/robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro \
  --attempt-id df89d796-4673-445f-b003-d65f6dc689e5 \
  --out /home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-linkage-physics-20261014-df89d796-4673-445f-b003-d65f6dc689e5 \
  --samples 101
```

Expected status: `FAIL_STATIC_CONSTRAINT_FEASIBILITY`, with 101 scan samples, maximum fitted-site residual about `0.120083 mm`, and `mj_step_calls=0`. Do not add `--dynamics`: the static closure gate has not passed.
