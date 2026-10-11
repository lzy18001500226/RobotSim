# Issue #46：原生 X2 OmniPicker 隔离抓取帧审计

**结果：当前两个有依据的源几何端点均失败；尚无可进入物理闭合或全臂 IK 的端点。** 这不是“所有姿态都不可行”的证明。本次停在静态几何门，未运行动力学、全臂 IK、瓶子抬升或放置。

## 来源与复现身份

- RobotSim 基线：`aab885ccddf68b1743b63d96f8f343699be544b2`，分支 `research/issue46-omnipicker-grasp-frame-20261012`。
- AgiBot 来源：[`AgibotTech/agibot_x2_urdf`](https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d)，commit `575cc6b988f976c23550e0db85aa1e5475d3652d`，Mulan PSL v2；模型 `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`，SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`。
- Canonical 瓶：70.0 mm 直径、244.5 mm 高、0.570 kg；helper SHA-256 `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`。
- Python `3.10.12`、MuJoCo Python/native `3.3.6`；native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`。
- 本次研究脚本 SHA-256 `2994a968ed1667afd4732d8549e897f19a2537538d8153b70a1824ddb20eb29d`。
- 每次候选从官方 `right_wrist_yaw_link` 子树建立固定根隔离模型；保留来源关节层级、安装变换、轴、限位、mimic 和网格。没有改源模型或碰撞过滤。

复现命令（把输出目录换成一个新的空目录即可）：

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -B scripts/research/issue46_native_omnipicker_grasp_frame_audit.py \
  --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
  --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robosuite-bottle-20261011/robotsim_reference/canonical_manipulation_assets.py \
  --output <new-empty-dir> --azimuth-deg 0
```

同一命令以 `--azimuth-deg 90` 运行第二个正交候选。已完成输出位于本机 `C:\Users\HP\RobotLab\evidence\issue46-native-omnipicker-grasp-frame-20261012\attempt-3cf77655-03ca-4946-bdd9-ae3986d935ad-final3-a0/` 与 `...-final3-a90/`。各目录含 `result.json`、`endpoint_geometry.json`、101 点 `closure_geometry_trace.jsonl`、mesh/hull 清单、抽取的 URDF、命令和多视图 PNG。

## 碰撞几何与视觉几何

| 位置 | 官方 URDF 视觉网格 | 官方 URDF 碰撞网格 | 源 STL 尺寸（局部轴） |
|---|---|---|---|
| `right_wrist_roll_link` | `right_wrist_roll_extend_link.stl`，SHA `94c4c7b3…397657de` | `right_wrist_roll_link.stl`，SHA `f6ed0a2a…d487c72` | visual `83.700 × 45.151 × 78.367 mm`；collision `99.633 × 71.043 × 204.628 mm` |
| `R_hand_narrow3_Link` | `narrow3_Link.stl` | 同一文件 | `52.953 × 27.672 × 26.000 mm` |
| `R_hand_wide3_Link` | `wide3_Link.stl` | 同一文件 | `52.920 × 27.612 × 60.378 mm` |

腕 visual STL SHA-256：`94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`；腕 collision STL SHA-256：`f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`。

腕部 visual 与 collision `<origin xyz/rpy>` 在 pinned URDF 中均为零；上述 raw STL 尺寸为文件局部坐标轴。编译 mesh 由 MuJoCo 转换后的坐标包围尺寸另列，不能直接按各轴与 raw STL 排序对比。

MuJoCo 的 collision mesh 编译件为 `right_wrist_roll_link`，`contype=1`、`conaffinity=1`；编译碰撞凸包记录 716 个独立顶点、1,352 个面，编译局部包围尺寸约 `61.736 × 101.367 × 204.584 mm`。原碰撞 STL 有 17,811 个独立顶点与 35,822 个三角面。视觉件独立保留：31,400 个源顶点、62,884 个三角面；仅用于画面的凸包/网格显示不会改物理碰撞。

**准确的阻挡网格是 collision `right_wrist_roll_link.stl`，不是 visual `right_wrist_roll_extend_link.stl`。** 官方 `X2-Ultra_simple_collision.urdf` 的该腕部 visual 与 collision 都引用 `right_wrist_roll_link.stl`（URDF SHA `f1de13da59ad9484c8940c8574705c75c0b0bd445b7954ae222639f33086f452`），所以它没有提供一个不同的、更短的腕部碰撞包络作为现成替代。本审计没有替换 OmniPicker 碰撞体。视觉/碰撞网格差异是已验证事实；其是否代表不同硬件外壳、装配变体或保守安全包络，仍未经官方 CAD/实物尺寸验证。

## 运动学、抓取帧与两个候选

- OmniPicker 安装：`R_omnipicker_joint`，`right_wrist_roll_link` → `R_omnipicker_base_link`，`xyz=[0.00196, 0.00035, -0.062] m`、`rpy=[3.14,0,0]`。
- 主动夹爪关节 `right_claw_joint`：轴 `[0,0,-1]`，源限位 `[-1,0] rad`。从动 `R_hand_wide1_joint`：同轴，源限位 `[0,1] rad`，官方 mimic multiplier `-1`、offset `0`；本次采样残差为 `0`，两关节均未越界。
- 源子树共有 13 个关节：以上两个为唯一活动 revolute/mimic 对；其余 narrow/wide 两侧的 2/3/4 段与 loop 链在该 URDF 中均为 fixed joints。两侧 distal mesh 分别是 `narrow3_Link.stl`、`wide3_Link.stl`，并非共享几何。
- 两个候选是以 compiled jaw 关闭方向相对瓶子水平截面作 `0°`、`90°` 正交摆放；指垫长向与瓶轴对齐。瓶子/夹爪位置直接由这两个源几何方向构造，没有随机种子或全臂搜索。
- 两候选隔离根世界位置分别为 0° `[0.298603,0.183975,1.041792] m`、90° `[0.116025,-0.001397,1.041792] m`；源 wrist pitch/roll 关节角为 0 rad，原始旋转矩阵和全量 joint state 保存在各自 `result.json`。
- OPEN 实测 jaw gap `103.091 mm`。达到 70 mm 瓶径的源限位内闭合点在 aperture ratio `0.580189`，`right_claw_joint=-0.580028504 rad`、`R_hand_wide1_joint=+0.580028504 rad`，compiled jaw surface gap `70.000293 mm`。
- 两个 compiled 指尖最近点距离：窄侧 `-0.001151 mm`，宽侧 `-0.043720 mm`。它们是微小几何穿入，不是由物理接触力证实的双侧抓取。两个表面最近点高度分别约 `876.999 mm` 与 `854.952 mm`，高差 `22.047 mm`；须在定义任务 TCP 时显式考虑真实 pad 表面，不能把腕部/基座原点当 TCP。
- pinned URDF 没有命名 TCP、`ee_link` 或 `grasp_frame`。报告中的诊断帧只是两个 closest-point witness 的中点，位置约 `[0.300021, 0.010732, 0.865976] m`（0° 候选），并非 AgiBot 定义的 TCP。

## 首个硬阻挡

| 候选 | OPEN 最小腕/瓶间距 | 接触开度最小非指垫间距 | 101 个闭合采样中腕碰撞 |
|---|---:|---:|---:|
| `0°` | `-42.440641 mm` | `-42.440641 mm` | 101/101 点均为同一距离 |
| `90°` | `-42.440641 mm` | `-42.440641 mm` | 101/101 点均为同一距离 |

碰撞对两候选均为 `right_wrist_roll_link/geom_6 ↔ m0_bottle/bottle_body`。0° 候选 witness 约为腕 `[0.268417, 0.015084, 0.900699] m`、瓶 `[0.306712, -0.003211, 0.900699] m`。瓶子在 OPEN 已与腕部编译碰撞体重叠 42.441 mm，闭合不会解除；因此当前 endpoint、OPEN 和闭合走廊均不具备无非抓取碰撞的条件。

另有一项需要单独追踪的静态结果：隔离姿态中 `right_wrist_yaw_link/geom_2` 与 `right_wrist_pitch_link/geom_4` 存在约 `39.217 mm` 几何穿入。没有报告该静态状态的力值；它不是瓶子抓取证据，也尚未判定为设计允许的相邻连杆重叠还是转换问题。

## 参考工作流与结论

- [MoveIt Task Constructor pick-and-place](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html)：分阶段构造预抓取/抓取帧、approach 和 close；规划场景可允许特定手指—目标物接触，但不代表物理仿真要关闭指垫接触。
- [Isaac Sim manipulator setup](https://docs.isaacsim.omniverse.nvidia.com/latest/learn/setup_a_manipulator.html) 与 [Grasp Editor](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/grasp_editor.html)：显式配置 manipulator/gripper、DOF、open/closed configuration、抓取帧和 collider，再通过物理仿真验证；本审计相应地先从真实 pad colliders 推导帧与开度。
- [MuJoCo mesh/collision computation](https://mujoco.readthedocs.io/en/3.3.5/computation/)：mesh 碰撞按凸包处理；渲染几何和编译碰撞表示必须分别检查。这里用 `mj_geomDistance` 检查已编译几何，并额外渲染 convex-hull overlay。

两个方向都因同一腕滚转碰撞体失败，所以停止在静态隔离门；不执行物理 close、全 X2 IK 或 arm path。此结论只覆盖这两个有界假设，不代表全局不可行。下一步应先由维护者/供应方确认 wrist visual 与 collision mesh 是否属于相同硬件装配，并核实碰撞 STL 的物理包络；在确认前不应擅自替换碰撞网格或继续全臂优化。

**物理验收：** bilateral contact physics `NOT RUN`；50 mm free-body lift + 1 s unsupported hold `NOT RUN`。本报告不是抓取 PASS。

**视觉证据：** 每个候选目录均有 `source_visual_only.png`、`compiled_collision_hull_only.png`、`source_visual_collision_overlay_front.png`、`source_visual_collision_overlay_side.png` 和 `source_visual_collision_overlay_contact_witness.png`。最后一张只将瓶子的渲染 alpha 设为 0.30，以显示 pad/wrist closest-point witness；碰撞与动力学未改。两候选的 `result.json` 和见证图 SHA-256 已在任务记录中校验。

关键证据 SHA-256：0° `result.json` `96bc4e38c3e0abcd449b39b8a2e815585bd71acb3ffed694c98acbd9666e020c`；90° `result.json` `5c169ddcc99d3c9826288dd87f448a4424bd6d93af13da705c5797a5b8aaaa72`；0° witness PNG `e62e695264e9d777656a1d967fd9638d96c65c9f24188d7f598bee6b838ef0b3`；90° witness PNG `4a65107aefd9199cc8be98bc638fdfe57bd2e5c165f742dc2f12bda3871dc0a4`。

**研究工具 disposition：** `scripts/research/issue46_native_omnipicker_grasp_frame_audit.py` 暂记 `KEEP-AS-TOOL`（离线几何审计）；不得由生产运行时依赖。静态门未通过，后续里程碑可据维护者决定调整 disposition。
