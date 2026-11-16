# X2 桌前左右换位：组合输入调研

日期：2026-11-12。Issue #46 / Draft PR #106。范围是已有官方来源与已执行轨迹的只读调研；没有新 `mj_step`、MC运动或模型/步态参数调整。

## 决定

优先验证**同一条速度消息中的前进＋侧向组合启步，随后降低前后分量、保留侧向分量**。这是有官方接口依据、尚未物理验证的假设。保持约50cm的骨盆到桌边观察基准；以实际进入右手可抓区域为终点，不追固定毫米级落点。不要把后退绕行设为必经阶段，也不要把多个发布者当作速度叠加器。

## 官方来源与现有复用

| 核对项 | 已确认内容 | 来源 |
|---|---|---|
| 三轴组合 | 一条 McLocomotionVelocity 含 forward/lateral/angular；正方向分别为前、左、逆时针 | [官方走跑接口](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/locomotion.html)；交付 `aimdk_msgs_ws/src/aimdk_msgs/interface/robot/mc/motion/msg/McLocomotionVelocity.msg:7-17` |
| 启步与维持 | 静止启步与运动维持不同，启动取决于组合输入；在线单轴侧移门限量级高于前进，但随模型/版本变化 | 同上；不能把在线0.5–0.7m/s参考直接用于当前桌前试验或突破原速度上限 |
| 可执行组合示例 | 官方示例给出(0.2,0,0.5)前进并左转；不保证纯角速度原地启步 | [官方MuJoCo指南](https://x2-aimdk.agibot.com/zh-cn/latest/sim_contents.html)；[Python示例](https://x2-aimdk.agibot.com/zh-cn/latest/example/Python.html) |
| 输入源 | 连续命令按输入源仲裁；零命令可能清零，不能当作多个源会相加 | [官方仲裁规则](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/MC_control.html) |
| 既有桥接 | 已有Twist三轴到同一MC消息的映射；原社区默认侧向上限0，小非零输入还有最小值抬升 | `reference/AgiBot-Community/raicom2026-agibot-X2/src/race_task/race_task/cmd_vel_bridge.py:30,165-209,231-239`；实际试验应核对已覆盖的启动参数及DDS输出，不假设默认值仍生效 |
| 既有点导航 | 有转向、前进、末端转向；不等于已具备面向桌子的平行换位 | 同仓 `service_pose_nav.py:239-315` |
| 不同模式 | 本地SD、LD、STEP分别映射cpgtelecon、foundationwalk、cpgstep；STEP是越野，不是原地踏步能力保证 | 交付 `mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/action_setting.yaml:32-57`；SDK `McAction.msg:34-35` |

官方在线是AimDK_X2 1.1.0，不能直接视为交付MC的精确规范。本地manifest记录MC/sim压缩包为v1.0.0-20260522，消息SDK来自1.1.0/faf4c945。赛事参考固定在c3ffe0d4980e6b7193170cc4f7f29aea680294cc，社区固定在a97761820b398b6cab12b632ce9f313c394eeb7e；两份参考checkout均只读且干净。

现有试验还复用了Issue #80的SIMULATION_ONLY FoundationWalk配置覆盖：膝关节action scale 0.25→0.19、min_run_velx 0.5→0.1；未改ONNX。它不是完整厂商默认配置，也不是本轮新增调参。以后比较模式须固定并记录该覆盖，不能隐去。

## 既有实测：停止后的实际位移

单位cm，正前/正左以动作开始的骨盆基体朝向投影；不是直接把世界Y当机体左。统计包含动作与停止尾段。

| 运行 / 记录 | 前后cm | 左右cm | 朝向变化deg | 退出原因 |
|---|---:|---:|---:|---|
| 20261112T050231Z / task_region_navigation_1_result.json | +5.31 | +0.37 | +1.26 | TASK_REGION_NO_MEASURED_PROGRESS |
| 20261112T125412Z / task_outer_0_0_result.json | +1.36 | -12.11 | +0.18 | LEFT_CHECKED_OUTER_BOX |
| 20261112T130124Z / task_outer_0_0_result.json | -19.76 | +4.49 | +3.72 | 无 |
| 20261112T130124Z / task_outer_0_1_result.json | -3.39 | +1.78 | +10.57 | LEFT_CHECKED_OUTER_BOX |

关键解释：

- STEP的侧向试验实际DDS侧向约0.074–0.120m/s，约5.44秒；最终左移约0.37cm，同时前移约5.31cm。不是成功的受控横移。
- 用户看到的STEP右移约12.11cm确实存在；首条非零DDS消息却为(-0.16196,0,0)，主要请求后退。这时基体朝向约±1.4deg，不能仅用侧身坐标投影解释。
- DEFAULT的另一后退段实际后退约19.76cm、左移约4.49cm。两轮输入时长、停止流程不同，**不是单因素模式A/B**；不能直接宣布STEP有厂商缺陷。
- 所查4段DDS在各自非零时间跨度内没有穿插全零包；不是证明整个系统永远无零源干扰。DDS观察和选中输入源仍不是MC内部策略执行回执。
- 离线隔离容器读取MC与当前消息SDK的C introspection：顶层消息均96字节，五字段名称/类型/顺序/offset相同(0,48,72,80,88)。未发现顶层前后/左右字段互换；这不是全部嵌套ABI或MC内部算法验证。
- RL debug采集工具虽被准备，但这些失败段缺少对应obs/action轨迹。不能编造“策略内部实际收到了哪个数”；下一次应复用现有只读采集器，从运动前开始记录。

## 为什么不优先用纯前进S弯

理想单轮车几何下，前进速度v、角速度先+ω后−ω、两段转角均θ，可恢复朝向，但侧移为2(v/ω)(1−cosθ)，前进为2(v/ω)sinθ。

例如v=0.12m/s、ω=0.40rad/s，想侧移12cm，理想结果会同时前进36cm。起始距桌50cm会变成14cm，还未计算手脚扫掠。**这是离线几何计算，不是X2运动预测或实际MC测试**，足以说明“前进＋转向”并不天然适合紧贴桌前的平行换位。空间和净位移都经过验证后，它仍可作为备选。

## 一个最小后续验证方案（未执行）

1. 沿用当前完整X2、夹爪和已审核源限制，从约50cm、面桌、无主动转腰/侧倾的稳定站姿开始。一个速度owner、一条三轴完整消息；保留真实反馈、新鲜度、源限位/effort、碰撞及停止检查。
2. 优先交付LD/LOCOMOTION_DEFAULT做有界方向检查，不把LD的后退记录当成侧移PASS。短段纯侧向输入作为基准；若不启步，再保持同一消息/源，加入小前进分量。
3. 示例候选(vx,vy,ω)=(0.12,±0.12,0)，合成平移约0.170m/s，仍低于既有0.20m/s命令上限。它只是待审测试输入；桥接后的实际DDS总量也必须满足上限，不能被最小值抬升。
4. 一旦确认迈步，连续调整vx比例、保留vy，观察能否维持正确侧向净运动；不要在这两段之间切站立/更换source，从而丢失要研究的维持状态。没有方向性改善就停止，不做增益/步态参数扫。
5. 记录基体与世界位移、桌法向距离、身体朝向、MC模式/源、实际停止尾程及现有RlMcDebug；对有效方式再补反方向检查。根据RGB-D重观测和实际可抓区域决定何时停，不要求复现一个名义移动距离。
6. 只在方向、距离带、停止和完整模型碰撞门通过后接回三代表点抓取；本轮调研不授权实际运动。

未知项：组合启步能否有效增强当前交付策略的侧向响应；在现有速度约束内能否持续向左右移动；前后串动和停止误差有多大；不同腰姿/负载是否改变响应。还没有证据支持线性叠加系数、精确横移API或11点任务PASS。

## 复现与验证

只读审计入口：`scripts/research/issue46/native_runtime/lateral_motion_source_audit.py`；单位测试：`python3 -B -m unittest scripts.research.issue46.native_runtime.test_lateral_motion_source_audit -v`。

完整已执行argv、4份输入SHA、审计JSON、官方页面快照、隔离消息元数据探针及命令在本轮canonical reviews目录；不需要启动MC或MuJoCo。最终本地路径、manifest和SHA记录在该目录的REPORT_ZH.md。分析工具处置RETAIN_RESEARCH_DIAGNOSTIC；不作为运行控制器依赖。
