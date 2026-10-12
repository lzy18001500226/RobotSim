# Issue #46：X2 抓取参考实现与碰撞根因检查

日期：2026-10-12
研究分支：`research/issue46-omnipicker-grasp-frame-20261012`
诊断前 RobotSim HEAD：`24ddf8dbdbe70dcdeefbbaf955db2947caebef6e`
结论范围：参考实现审计 + 两个有依据的原版几何候选；未运行全臂 IK 或接触动力学。

## 结论

这两个候选都能在原版关节限位内让窄、宽指腹到达 70 mm 瓶身，但原版右腕滚转碰撞体及两侧手指环路同时穿入瓶身。最近距离分别为腕部 `-39.9062 mm`、窄侧环路 `-27.6692 mm`、宽侧环路 `-27.6799 mm`。此处为 MuJoCo 编译碰撞几何上的静态有符号距离；不是接触力或动力学结果。

源 STL 与编译碰撞表面的双向采样距离均为纳米量级，包围盒也一致到纳米量级。因此，本组证据不支持“MuJoCo 凸化单独制造了 40 mm 障碍”；相反，源碰撞表面采样本身已深入瓶身。腕部视觉网格明显小于其碰撞网格，但没有官方 CAD/机械尺寸证明这是错误、保守安全包络还是不同硬件变体，不能自行替换。

判定：**SOURCE-FAITHFUL 静态端点/路径：本次两个候选均 FAIL；物理接触与 50 mm 抬升：NOT RUN。** 这不是对所有腕向和所有抓取构型的全局不可行证明。

## 参考实现比较

| 来源与固定版本 | 源码实际实现 | 本任务复用/边界 |
|---|---|---|
| [AgiBot X2 grasp demo](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/tree/26d3d45bc466139cbf6a124439e23f3df6eb8d2c)，`26d3d45bc466139cbf6a124439e23f3df6eb8d2c`，[Apache-2.0 license](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/blob/26d3d45bc466139cbf6a124439e23f3df6eb8d2c/LICENSE) | `grasp_planner.py` 从目标点构造 grasp / pregrasp，固定工具局部 `+Z` 对齐世界/机身抓取轴，绕轴滚转保持自由；当前姿态、撤退姿态及肩/肘/腕若干小扰动作为连续 IK 种子。采用 120 mm standoff、15 mm 笛卡尔接近步长、分段 20 mm 抬高，目标 60 mm（最小 40 mm）。`grasp_executor.py` 按准备、开爪、接近、闭爪、抬高、后撤阶段通过 ROS 2 / AimDK 命令执行，手命令发布率 50 Hz。 | 复用“轴约束 IK + 几何定义的 standoff + 连续种子 + 分段 approach/lift”的方法。没有照搬 `gripper_reach=0.11 m`、`grasp_x_offset=0.05 m` 或 `grasp_plane_z=0.262 m`：这些偏移依赖 demo 的工具帧与模型版本。demo URDF 有 63 links、62 joints，但 **0 visual / 0 collision geoms**；其 IK 成功不能证明碰撞可行或物理抓取。 |
| [MuJoCo Menagerie Robotiq 2F-85](https://github.com/google-deepmind/mujoco_menagerie/blob/0059d4335f8156206f63a35662313385f7ad6d74/robotiq_2f85/2f85.xml)，Menagerie `0059d4335f8156206f63a35662313385f7ad6d74`，[model license](https://github.com/google-deepmind/mujoco_menagerie/blob/0059d4335f8156206f63a35662313385f7ad6d74/robotiq_2f85/LICENSE)，BSD-3-Clause | 分开视觉网格与碰撞网格；指腹碰撞用两块零质量 box pad；只排除指定相邻 linkage/body 对；左右 driver 由 fixed tendon 分力，follower 由 equality 耦合；单一 tendon actuator 有 `forcerange=-5..5 N`、`ctrlrange=0..255`，仿射 position/velocity feedback；椭圆摩擦锥 `impratio=10`，指腹摩擦 `.7/.6`，pad contact `priority=1`。 | 作为“有效小面积指腹 collider、精确列出 linkage exclusion、单控制输入带动耦合爪”的建模范例。没有把 Robotiq 几何、执行器或接触参数移植给 OmniPicker，也没有用其抓取结果代替 X2 验收。XML 参数本身不构成对本瓶子的经验成功证明。 |
| [robosuite](https://github.com/ARISE-Initiative/robosuite/tree/5ce6643f3092639d08f7b0f90ed1c6a84f50552c)，`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`，[MIT license](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/LICENSE) | `BottleObject` 通过 wrapper 添加 free joint（damping `.0005`）；XML 用同一网格分离视觉与碰撞用途，视觉 geom 不参与接触，另一个 geom 参与物理，密度 `50`，摩擦 `.95/.3/.1`。XML sites 给出的高度端点为 `-0.082/+0.075 m`。 | 复用“视觉与碰撞职责分开”的资产组织方式作为审查参考。该瓶不是 RobotSim 规范目标：RobotSim 仍保留 70 mm × 244.5 mm、0.570 kg 的四段瓶身；不将 robosuite 默认密度/瓶子替换为规范对象。 |
| [GraspNet baseline](https://github.com/graspnet/graspnet-baseline/tree/280c215129f759ed8649cb4e89fc5dfee55f4f80)，`280c215129f759ed8649cb4e89fc5dfee55f4f80`，[research license](https://github.com/graspnet/graspnet-baseline/blob/280c215129f759ed8649cb4e89fc5dfee55f4f80/LICENSE)，学术/非营利非商业研究许可 | 网络解码 grasp score、width、approach、angle、depth、rotation、center、tolerance；碰撞过滤器先体素化场景点云，再用固定手指尺寸的局部 box 区域统计碰撞比例和 approach corridor。 | 仅借鉴输出量与逐候选 corridor 检查的分层思路。点云 box/IoU 过滤不是原版网格精确相交、关节限位检查、力闭合检查或 MuJoCo 动力学。许可也不支持不经审查纳入商业产品。 |
| [MuJoCo 3.3.6 collision / mesh 文档](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#asset-mesh) 与 [collision filtering 文档](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html#collision-detection) | MuJoCo mesh collision 使用凸包；非凸物体可由多个凸 geom 组成。默认过滤 parent-child body 碰撞，除非特殊 world 关系或显式过滤属性改变。 | 用于分别核查编译碰撞距离与 parent-child 过滤；不把“凸包”推断成当前 wrist 侵入的原因，也不把离线分解当作可替换的源碰撞模型。 |

上表固定版本的 X2 源模型来自 [AgiBotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d)，commit `575cc6b988f976c23550e0db85aa1e5475d3652d`，Mulan PSL v2。源 URDF：`X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`，SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`。

## 本轮厘清的旧阻挡

1. 抓取帧不再由整手/腕网格主轴估计：开度轴来自两侧 distal link 在真实开合运动中的相对位移，横向中心来自闭合到 70 mm 开度时两块真实指腹的 compiled witness points。因此确认了“指腹可达”这一子问题。
2. 之前报告的 yaw/pitch `-39.217 mm` 不能作为完整 X2 自碰撞结论：隔离夹具改变 parent-child 过滤关系，当前已与真实手-瓶碰撞分开。
3. 当前 wrist/bottle 阻挡不支持归因于 MuJoCo 凸包：在同一目标 pose 上，源三角面采样和编译表面采样几乎重合。
4. 已定位本次两个构型共同的首个几何门槛：原版 wrist collision geom 和两个 loop 几何体侵入 canonical 瓶身。未通过该门槛，所以不运行闭爪动力学或抬升。

## X2 demo 与 pinned source 的差异

| 项目 | demo URDF | RobotSim 固定官方 URDF | 差值/影响 |
|---|---:|---:|---|
| 右 OmniPicker mount xyz | `(0, 0, -0.046) m` | `(0.00196, 0.00035, -0.062) m` | X/Y +1.96/+0.35 mm；Z 后退 16 mm；两者 RPY 均约 `(pi,0,0)`。 |
| 右腕 pitch 原点 xyz | `(0,0,-0.0818) m` | `(0,0.0005,-0.0823) m` | Z 差 0.5 mm，Y 差 0.5 mm。 |
| 右腕 pitch 限位 / effort / velocity | `±0.558 rad / 4.8 / 4.188` | `±0.523599 rad / 6 / 15.708` | 轴同为 Y，但限位及执行器元数据不同。 |
| 右腕 roll 限位 / effort / velocity | `[-0.724, 1.571] / 4.8 / 4.188` | `[-0.724312, 1.509710] / 6 / 15.708` | 上限差约 0.0613 rad，执行器元数据明显不同。 |
| 右腕 roll 质量 | `0.303847 kg` | `0.107 kg` | 约 2.84 倍差异；不能拼接两版的惯量或运动参数。 |
| visual/collision | demo OmniPicker URDF 未含 visual/collision geometry | 官方模型有完整 visual 与 collision 几何 | demo IK 只可作为帧/路径方法参考。 |

两版都用右 claw 作为主动坐标、`R_hand_wide1_joint` 作为 mimic follower；源限位分别为主动 `[-1,0] rad`、从动 `[0,1] rad`，mimic multiplier `-1`、offset `0`。demo 的源文件及参数链接：[`grasp_planner.py`](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/blob/26d3d45bc466139cbf6a124439e23f3df6eb8d2c/demo/src/x2_grasp/x2_grasp/grasp_planner.py#L75-L118)、[`grasp_executor.py`](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/blob/26d3d45bc466139cbf6a124439e23f3df6eb8d2c/demo/src/x2_grasp/x2_grasp/grasp_executor.py#L101-L177)、[`solver.py`](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/blob/26d3d45bc466139cbf6a124439e23f3df6eb8d2c/demo/src/x2_grasp/x2_arm/solver.py#L296-L335)、[`unified_grasp.yaml`](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/blob/26d3d45bc466139cbf6a124439e23f3df6eb8d2c/demo/src/x2_grasp/config/unified_grasp.yaml#L106-L127)、[demo URDF](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/blob/26d3d45bc466139cbf6a124439e23f3df6eb8d2c/demo/src/x2_grasp/urdf/x2_ultra_plus_omnipicker_omnipicker.urdf)。官方模型源位置见 [pinned X2 URDF](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1028-L1068) 及 [右手安装/主动关节](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1337-L1385)。

## 几何候选与碰撞根因

使用本地 research checker 从实际编译后的 `R_hand_narrow3_Link` / `R_hand_wide3_Link` 指腹接触见证点导出 jaw axis 与抓取帧；不使用手掌原点或 PCA。世界抓取目标为规范瓶身中心 `(0.300, 0, 0.8775) m`，开爪预抓 standoff `120 mm`，接近每步 `15 mm`。每个候选检查 9 个 approach 样本、101 个闭合样本及 1001 个 source-limited jaw aperture 样本。

| 指标 | 候选 1：正面 approach | 候选 2：侧向 approach |
|---|---:|---:|
| Approach 方位 / 世界方向 | `0°`, `+X` | `-90°`, `-Y` |
| 隔离夹爪 root 世界位置 | `(0.0535783361, 0.0010523474, 0.8667302272) m` | `(0.3010523474, 0.2464216639, 0.8667302272) m` |
| source-limited 接触开度 | `70.00000775 mm` | `70.00000775 mm` |
| 右主动 / mimic 角度 | `-0.580025374 / +0.580025374 rad` | 相同 |
| OPEN 两侧指腹间隙 | `17.9168 / 17.9115 mm` | 相同至数值精度 |
| 闭合后指腹到圆柱表面的有符号距离 | `-0.00554 / -0.00012 mm` | 相同至数值精度 |
| 腕/环路到瓶身的最小距离 | 腕 `-39.9062 mm`；窄环 `-27.6692 mm`；宽环 `-27.6799 mm` | 数值相同至浮点精度 |
| 机器人到桌面的最小距离 | `+10.6542 mm` | `+10.6542 mm` |
| source joint limits | PASS | PASS |
| 整体静态 grasp/path | **FAIL** | **FAIL** |

### 根因判别表

| 问题 | 结论 | 证据/限制 |
|---|---|---|
| 39.217 mm wrist-yaw / wrist-pitch 是否为完整 X2 的真实 active self-collision？ | **否，测量来自隔离 fixture 拓扑。** | 官方 URDF 是 yaw→pitch 的 parent-child；MuJoCo 默认滤除此类接触。隔离模型把 yaw 子树根化到 world 后，该对不再是 parent-child，静态 contact candidate 才出现 `-39.217 mm`。不是手-瓶阻挡证据。没有添加 exclusion。 |
| 原版腕碰撞 mesh 是否与瓶体交叠？ | **是，静态几何强证据。** | `right_wrist_roll_link/geom_6` 对 `m0_bottle/bottle_body` 最小有符号距离 `-39.9062 mm`；源 collision STL 全顶点及逐三角面心采样中 `49,737/143,288 = 34.711%` 落在瓶身圆柱体内部。该采样不是精确布尔体积计算。 |
| 主要阻挡是否由 MuJoCo 凸包额外引入？ | **本候选中不支持。** | 编译 hull 顶点/面心对源面样本的双向最近邻 p95 分别约 `2.59e-9 / 3.15e-9 m`，最大约 `4.10e-9 m`；世界 AABB 差为纳米量级；源与 hull 位于瓶身内的比例约 `34.71% / 34.77%`。这是表面采样证据，不是精确 mesh boolean 证明。 |
| 腕视觉 / collision 几何是否一致？ | **不一致，原因未知。** | 官方 URDF visual 使用 `right_wrist_roll_extend_link.stl`，collision 使用 `right_wrist_roll_link.stl`，两者 origin 为零、默认 scale 为 1；视觉 AABB 尺寸约 `83.700×45.151×78.367 mm`，collision 约 `99.633×71.043×204.628 mm`。没有权威 CAD/机械外壳尺寸可判断 collision 是保守包络、不同装配还是资产问题。 |
| grasp frame 是否找到功能接触面？ | **指腹几何层面是；完整端点不是。** | frame 由实测 distal pad witness line 及 jaw 闭合轴构建，两个 pad 都可在 source limits 内触到 70 mm 瓶身；但 wrist 与两条 loop 同时侵入，故不能据此称完整可行 grasp。没有采用 20 µm 的通用成功阈值。 |
| source-faithful 端点全局可行吗？ | **未证明。两个候选 FAIL。** | 正面和侧向两个有物理意义的方向都受同一非抓取碰撞阻挡；不推断其他 wrist roll/pitch、抓取高度或不同机械包络均不可行。 |
| 全身右臂 IK / 物理抓取 / 接触力 / 抬升是否测试？ | **NOT RUN。** | 由于末端刚性 wrist subtree 已在瓶身处被阻挡，本任务停止于隔离手腕的静态几何门槛；未启动全臂 IK。静态检查仅 `mj_forward`，`physics_steps=0`、瓶体 qpos writes `0`。没有接触力、动力学摩擦、承载或抬升数据，也未生成动态视频。 |

官方碰撞 STL SHA-256 为 `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`；视觉 wrist STL SHA-256 为 `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`。源与编译表面在诊断的坐标系中相符；源几何是否符合真实硬件外壳仍未知。

## 凸分解诊断的处置

一次独立 CoACD 离线诊断以原 wrist collision STL 为输入，固定参数为 threshold `0.001 m`（real_metric）、preprocess auto/res `50`、resolution `2000`、MCTS `20`、iterations `150`、maxdepth `3`、PCA off、merge on、no decimation、seed `0`。生成 489 个 watertight 凸片，原始网格非 watertight。源表面→分解表面最近邻 p95/max 为 `2.160/5.413 mm`，分解表面→源表面为 `7.387/14.185 mm`；部分反向误差可能来自分解内部面，且分片体积之和不能作为 union 体积（约为输入 trimesh 报告体积 1.2804 倍，包含重叠）。导出后摘要后处理失败，**未形成可验证的并集、未加载成 MuJoCo 模型、未进行接触测试，明确拒绝作为替代碰撞模型**。原始 STL 和 489 个分片作为 DIAGNOSTIC ONLY 保存于 `collision_decomposition_diagnostic/`。

此前的六片 proxy 曾扩大 loop 包络且仍阻挡瓶体；facet proxy 曾有 1 s 双侧接触但发生约 5.003 mm 横向逃逸且未抬升。这些均不是 source-faithful 成功证据，也不是当前任务的新动力学实验。

## 状态、环境与复现

- RobotSim 诊断前 HEAD：`24ddf8dbdbe70dcdeefbbaf955db2947caebef6e`；研究脚本 SHA-256：`2807c846e79da1fbe58316b44ab62ee83d81c0a94067592378910ac1b037139d`。
- X2 upstream SHA：`575cc6b988f976c23550e0db85aa1e5475d3652d`；MuJoCo Python/native `3.3.6`；Python `3.10.12`；WSL2 Ubuntu 22.04 x86-64。
- canonical helper SHA-256：`41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`。
- 11 项既有 SIMULATION_ONLY exclusions 保持；未增加新 exclusion、未改源碰撞几何/限位/瓶子物理参数。
- 精确命令见各候选 `experiment_command.txt`。示例：

```bash
MUJOCO_GL=egl /tmp/robotsim-issue46-coacd-venv/bin/python -B \
  /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011/scripts/research/issue46_reference_frame_static_check.py \
  --x2-root /tmp/robotsim-issue46-reference-study-20261012/x2source \
  --canonical-helper /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-x2-reference-physics-recovery-20261012/inputs/canonical_manipulation_assets.py \
  --output /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-x2-reference-physics-recovery-20261012/candidate1_review \
  --candidate-id 1 --approach-azimuth-deg 0.0
```

候选 2 使用相同命令，把 output 改为 `candidate2_review`、`--candidate-id 2 --approach-azimuth-deg -90.0`。检查器以预期几何 gate FAIL 返回非零状态；两份 `result.json` 均记录 `status=FAIL`，这不是程序异常，也不是物理测试失败。

永久证据目录：`C:\Users\HP\Desktop\RobotSim\reviews\issue46-x2-reference-physics-recovery-20261012\`

- 每候选：`result.json`、`static_geometry_trace.jsonl`、`experiment_command.txt`、生成的 isolated URDF、开爪 overview、front/side 接触几何及 front/side 碰撞 overlay。
- 原腕 collision mesh 与 CoACD 489-piece 分片：`collision_decomposition_diagnostic/`，明确仅供几何诊断。
- `SHA256SUMS.txt` 包含两候选 JSON/JSONL、front/side annotated overlay 与源 collision STL 的哈希。
- 源模型使用 X2 官方 Mulan PSL v2；未把 vendor 资产复制到 RobotSim 跟踪目录。

## 下一步

先取得官方 wrist-roll 外壳 CAD/机械尺寸或 AgiBot 对 `right_wrist_roll_link.stl` 碰撞包络用途的说明，确认它和视觉 `extend_link` 的差异是有意设计还是资产错误。确认前不应再枚举相似腕角，不应把 CoACD 分片或 visual mesh 升格成源模型。若维护者批准诊断碰撞代理，再单独建立 `SIMULATION_ONLY` 对照；任何代理成功仍须与源模型结果分开报告。

**目前结果：两个受限 source-faithful 静态候选 FAIL；真实接触、自由瓶承载、50 mm 抬升均 NOT RUN；没有全局不可行结论。**
