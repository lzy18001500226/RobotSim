# Issue #46：原生 X2 OmniPicker 抓取恢复阶段报告

**结论：PARTIAL；静态抓取端点仍被源碰撞几何阻挡，未进入动力学。** 本轮按有限几何假设检查了瓶身和瓶盖目标。没有找到同时满足双指接触精度、OPEN/闭合全程无非目标碰撞及零初始自穿透的源几何端点，因此没有运行指爪接触物理、全臂 IK、路径、抬升或释放。

## 复现身份

- RobotSim 分支：`research/issue46-omnipicker-grasp-frame-20261012`；执行时基线 HEAD：`71cadfbafdcd6b4c6e581fde0aa36c719cc69615`。本报告和审计脚本随后作为研究变更提交。
- X2 来源：[`AgibotTech/agibot_x2_urdf`](https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d)，固定 commit `575cc6b988f976c23550e0db85aa1e5475d3652d`，Mulan PSL v2；模型 `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`，SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`。
- canonical 瓶 helper SHA-256：`41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`；瓶身直径 70 mm、总高 244.5 mm、总质量 0.570 kg。瓶身、肩部、瓶颈和瓶盖保持原始 canonical 几何与物理参数。
- Python `3.10.12`；MuJoCo Python/native `3.3.6`；native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`。
- 最终审计脚本 SHA-256：`6b90124493e8e5ed2122f7afadf44cfd8e0a1160b5e9611098056d90d4c8a5b9`。本轮没有改官方 URDF、源网格、joint range、mimic、过滤、瓶子或物理参数。
- 隔离 fixture 固定在官方 `right_wrist_yaw_link` 根；仅用于静态 endpoint / collision 几何检查。没有 physics step（所有候选 `physics_steps=0`）。

## 已验证结果

### 瓶身目标的前序证据

沿用本轮前已生成的同一 pinned source / canonical 瓶证据，不重复执行：

| 假设 | 指垫到瓶身距离 | wrist-roll 最近碰撞 | 结果 |
|---|---:|---:|---|
| 瓶身中心、手指长轴竖直、0° 方位 | 约 −0.001 / −0.044 mm | 瓶身 −42.441 mm | wrist 已在 OPEN 状态穿入瓶身 |
| 瓶身中心上移 60 mm | 窄侧 −51.318 mm，宽侧 −0.031 mm | 瓶身 −41.411 mm | 腕碰撞且一侧严重穿入 |
| 瓶身中心上移 100 mm | +4.014 / +4.421 mm | 肩部 −35.956 mm | 没有指垫接触，腕仍穿入肩部 |
| 瓶身中心、手指长轴 ±90° | 窄侧 −22.427 mm，宽侧约 −0.004 mm | 瓶身 −51.172 mm | wrist 碰撞 |

这些是有限候选的失败，不构成全局不可行证明。来源和 mesh / compiled-hull 细节见 [前序 grasp-frame audit](issue46_native_omnipicker_grasp_frame_audit_20261012.md)。

### 瓶盖侧夹的新增几何候选

目标从编译后的 canonical `bottle_cap` 圆柱导出：半径 20 mm、直径 40 mm、中心世界位置约 `[0.300, 0, 1.036] m`。对源 jaw gap 的开合样点按 40 mm 交点插值，并将夹爪闭合中点对准瓶盖中心；瓶身和其余三个瓶体 geom 仍是碰撞障碍。四个小范围朝向均检查 OPEN、闭合端点和 101 个闭合样点：

| 方位 / 手指长轴倾角 | 指垫到目标瓶盖最近距 | 首要非目标阻挡 | OPEN / 闭合 |
|---|---:|---|---|
| 0° / 0° | −0.0298 / +0.0276 mm | wrist-roll `geom_6` ↔ `bottle_cap`，−25.238 mm | 不清 / 不清 |
| 0° / +90° | +0.0878 / +0.0276 mm | wrist-roll `geom_6` ↔ `bottle_shoulder`，−37.033 mm | 不清 / 不清 |
| 0° / −90° | +0.0878 / +0.0276 mm | wrist-roll `geom_6` ↔ `bottle_cap`，−25.550 mm | 不清 / 不清 |
| 90° / 0° | −0.0298 / +0.0276 mm | wrist-roll `geom_6` ↔ `bottle_cap`，−25.238 mm | 不清 / 不清 |

指垫精度门槛为每侧绝对 signed distance `<= 0.020 mm`；上表最近距均未满足双侧门槛，不能记作接触。以代表性 0° / 0° corrected-metric 输出为准，实际编译 jaw surface gap 是 `40.000013 mm`，不是把目标直径当作实测 gap。driver/follower 为 `−0.265167541 / +0.265167541 rad`，mimic residual `0 rad`，官方源限位通过。

另外在 OPEN 状态，`right_wrist_yaw_link/geom_2` 与 `right_wrist_pitch_link/geom_4` 自碰穿入 `39.217 mm`。该结果此前也出现在源隔离子树审计；没有添加排除。它是单独的有效性问题，不将其误报为瓶子接触。

### 几何解释和边界

- 四个瓶盖候选的首个瓶—机器人阻挡均是原始 pinned OmniPicker 的 `right_wrist_roll_link` 编译碰撞 mesh；不是视觉网格。前序 mesh inventory 已记录视觉文件 `right_wrist_roll_extend_link.stl` 与碰撞文件 `right_wrist_roll_link.stl` 不同，且 MuJoCo 对 collision mesh 使用编译凸包。本轮没有用 visual mesh 替代 collision mesh，也没有删碰撞。
- 正交方位或手指轴倾转不能解除 wrist-roll 对瓶盖/肩部的碰撞；在高位瓶身候选中，pad 与目标错开 4.0–4.4 mm，而 wrist 仍穿入肩部 35.956 mm。证据支持“当前有界几何假设被腕碰撞包络阻挡”，不支持全局源 gripper 不可抓取的结论。
- canonical bottle 的四个碰撞 geom 挂在同一刚性物体上，没有独立瓶盖螺纹/脱落自由度。因此即使未来瓶盖 pinch 静态通过，仍需明确它不能验证瓶盖结构或螺纹承载；必须再以目标任务和物理接触验证为准。
- 因 OPEN clearance、全闭合通道和零初始自穿透未通过，受控物理闭合、接触力、load transfer、50 mm 抬升、全 X2 IK 与 path 全部 `NOT RUN`。没有接触/抓取/抬升 PASS。

## 自动检查和 reproduction

已运行 `py_compile`、`git diff --check`、审计脚本 `--help`。每个静态候选校验 pin 与资产哈希、source axes / range / mimic、源 limits、OPEN clearance、全闭合非目标碰撞、端点 pad 距离和自穿透；候选 JSON 与 101 样点轨迹已保留。由于静态端点失败，没有运行 IK 或物理验收测试。

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -B scripts/research/issue46_native_omnipicker_grasp_frame_audit.py \
  --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
  --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robosuite-bottle-20261011/robotsim_reference/canonical_manipulation_assets.py \
  --output /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-native-omnipicker-grasp-recovery/<new-empty-dir> \
  --azimuth-deg 0 --finger-tilt-deg 0 --grasp-height-offset-mm 0 --target-bottle-geom bottle_cap
```

代表性更正测量的完整 JSON、101 点闭合轨迹、isolated URDF、source/compiled mesh inventory 和图片在：

`C:\Users\HP\Desktop\RobotSim\reviews\issue46-native-omnipicker-grasp-recovery\attempt-e04e0754-1034-4a6e-8249-1fc0760a16c8-cap-a0-t0-corrected-metric\`

其它三个瓶盖候选及前序瓶身高度/倾角候选均保存在同一 review 目录的命名子目录中。每个目录包含 `result.json`、`endpoint_geometry.json`、`closure_geometry_trace.jsonl`、`experiment_commands.txt`、mesh inventory、URDF 和多视图 PNG；没有物理 MP4，因为没有运行物理步。

## 阶段判定 / 后续依赖

- Native isolated source-faithful grasp endpoint：本轮有限候选 `FAIL`；全局可行性 `UNPROVEN`。
- Bilateral contact / physical lift / unsupported hold：`NOT RUN`。
- Full-arm IK / collision-aware approach：`NOT RUN`，静态端点先决门失败。
- 当前唯一明确阻挡：源 `right_wrist_roll_link.stl` 对瓶盖/瓶肩的编译碰撞包络，与已测试的源 pad 接触姿态重叠；另有 yaw/pitch 39.217 mm 初始自碰。
- 最小后续动作：由维护者或 AgiBot 资料确认 visual/collision wrist 网格差异是否对应真实硬件外廓，并确认相邻 wrist yaw/pitch overlap 是否为预期模型表示。未核验前不应以视觉网格替代、削薄碰撞体、增加 collision exclusion 或继续全臂 IK。若需诊断 proxy，应另行获批并与 source-faithful 结果分开。

本研究脚本 disposition：`KEEP-AS-TOOL`（离线静态几何审计）；不由生产运行时代码依赖。PR #58 保持不变，不合并。
