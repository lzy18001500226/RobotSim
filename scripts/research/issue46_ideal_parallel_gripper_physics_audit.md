# Issue #46：理想平行夹爪 run05/run06 摩擦与受力审计

## 结论

**理想夹爪物理基线仍为 FAIL，失败门为 load transfer。** run05 的 16.78 mm 是短暂抬升，随后圆柱滑落；run06 将 50 mm carriage smoothstep 从 0.60 s 放慢到 2.00 s 后，圆柱最高仅升 0.0276 mm，整个 1 s suspended-hold 仍与桌面接触。慢速试验排除了“抬升过快是主要原因”，但不排除速度是 run05 短暂抬升的次要影响。

主要物理机制是：**两侧保持双边法向接触，但接触切向力远未达到名义摩擦锥上限，圆柱相对指垫持续滑动，手部实际向上受力不足以卸载桌面。** 名义 `mu*N` 是上界，不是求解器必须施加的力。MuJoCo 软接触允许非最大耗散摩擦；本模型使用 pyramidal cone，因而非零滑动时也可能出现未饱和的切向力。现有证据支持这一解释，但没有记录真实积分步的 solver 统计，故不能据此证明“求解器每步都完全收敛”。

此外确认了一个**受力汇总缺陷**：旧 runner 对每个接触的世界 Z 力先做 `max(0, Fz)` 再求和，隐藏了同类支撑接触之间的向下分量，高估了净手部支撑。原始 trace 中的逐接触带符号力没有损坏。已在研究 runner 中改为同时输出净力、向上分量和向下分量，并标注接触力评估时刻；未改模型、接触参数或控制器，也未重跑完整物理序列。

## 原始证据与身份

| 项目 | run05 基线 | run06 慢速 lift |
| --- | --- | --- |
| 输出目录 | `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-ideal-parallel-gripper-20261011-run05\` | `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-ideal-parallel-gripper-20261011-run06-slow-lift-2s\` |
| 运行记录中的 RobotSim SHA | `079dc2f8955448da13329fb3d3ac37c9fd359711` | `0510a742c698efa29c7fe0202d225524fed3ab0d` |
| 原运行器 SHA-256 | `2122679aaf76e93dc537ae2a2cf8852bf8553160701a4a827ee53473b724d837` | `2bd791103bab29af63a145a83013fc62e5bd4884aa36d98c12c138eaba9af807` |
| result SHA-256 | `1ce89d482679dc30e24efedf1c4ba28bdb209d764b7a20764122b8ebb87d1a0d` | `bf161505b8bd325108844f0007695d58c1ea204fdf2844110a0c93a2fa2fbb24` |
| physics trace SHA-256 | `e1b89e4a43957d5a4f03024eba8cc5613404fb10f3e42d529e6846dde8d2d150` | `3bed5cdb90f651e05fb8560a145fd71e23eccb5f626eb9825e5a05468463e770` |
| 模型 XML SHA-256 | `b44a29746c1a7871236f47b96ca6d086ea5f6533ebcbe8014dea6490a2ebf24e` | 同左 |
| 视频 SHA-256 | `e9363d85bf6b7638880b4ae380e4cfcb7bb9d139b121a11f84f09af692129422` | `94c048b43099670f2e0fbda749dcf73bac20ad70c3d087fbe717cbc1384c4b9d` |

两次运行均为 MuJoCo 3.3.6 / Python 3.10.12 / WSL2；native `libmujoco.so.3.3.6` SHA-256 为 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`。圆柱是 70 mm × 244.5 mm、0.570 kg freejoint，重力 9.81 m/s²，几何摩擦参数 `[1.4, 0.02, 0.001]`。run06 只改变 lift smoothstep 时长；run05/run06 原始结果及视频均未修改。

## 接触模型、力坐标与数值证据

编译模型为 timestep 0.002 s、`implicitfast`、Newton、80 次迭代上限、tolerance `1e-10`；所有相关接触 `condim=4`，`cone=pyramidal`。实际 pad/cylinder pair friction trace 为 `[1.4, 1.4, 0.02, 0.001, 0.001]`；table/cylinder trace 也记录同一有效向量。默认接触参数为 `solref="0.02 1"`、`solimp="0.9 0.95 0.001"`（编译值 `[0.9, 0.95, 0.001, 0.5, 2]`）。双指对瓶身每侧每步一个 contact point，桌面接触另计。

`contact_rows()` 使用 MuJoCo `mj_contactForce` 的接触坐标系六维力/力矩，以 `contact.frame.T` 转到世界系，并按瓶身是 geom1 或 geom2 修正符号。对保存的逐接触瓶身受力求和后，hand + table 的世界 Z 合力与瓶重一致到约 0.002 N 以内；未发现力反号或重复计数。两侧接触法向夹角在 run06 lift 中均值约 `179.99996°`，不是两指同向推挤造成的横向合力问题。

接触力是在 `mj_step` 积分前的 forward dynamics 中求出的，而记录的位姿/速度是在积分后读取的，因此原 trace 的 `time_s` 与力/接触几何存在一个 2 ms 样本错位。以下切向相对速度用前一记录状态与当前接触快照配对，避免把该错位误当成精确同步数据。MuJoCo 文档说明 `mj_contactForce` 返回接触坐标系力，并指出软接触可能不满足最大耗散；pyramidal 与 elliptic cone 也会形成不同软接触动力学：[计算与接触模型](https://mujoco.readthedocs.io/en/3.3.6/computation.html)、[API 函数](https://mujoco.readthedocs.io/en/3.3.6/APIreference/APIfunctions.html)。

### 带符号重算结果

表中 hand/table 为瓶身所受接触力的逐步世界 Z 分量，按各阶段全部步数平均；downward 是负数。upward 与 downward 分开列出，净力等于二者之和。

| 阶段 | run | table 接触步 | hand 净 N | hand 向上 N | hand 向下 N | table 净 N |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLOSE | 05 / 06 | 350/350 | 0.02194 | 0.28036 | -0.25842 | 5.56924 |
| 桌面 HOLD / preload | 05 / 06 | 500/500 | 0.02166 | 1.11398 | -1.09232 | 5.57053 |
| LIFT | 05 | 108/300 | 4.36745 | 4.49428 | -0.12682 | 1.22735 |
| LIFT | 06 | 963/1000 | 3.22221 | 3.36456 | -0.14235 | 2.36921 |
| SUSPENDED_HOLD | 05 | 295/500 | 3.29059 | 3.45568 | -0.16509 | 2.29915 |
| SUSPENDED_HOLD | 06 | 500/500 | 0.62265 | 1.32381 | -0.70116 | 4.96956 |

run05 旧报告写的 suspended hand support `3.456 N` 是逐 contact 正向分量之和；净手力应为 `3.291 N`，差 `0.165 N`。run06 hold 旧字段 `1.324 N` 也高于净值 `0.623 N`，差 `0.701 N`。这修正了读数，但没有改变失败归因：run06 `0.623 + 4.970 = 5.592 N`，圆柱重量为 `5.592 N`，桌面仍承担绝大部分载荷。

run06 各 pad 的平均瓶身世界力向量：

| 阶段 | 左 pad `(Fx,Fy,Fz)` N | 右 pad `(Fx,Fy,Fz)` N | 每侧 Fn / 局部切向力范数 N |
| --- | --- | --- | --- |
| 桌面 HOLD | `(4.808, 0.015, 0.015)` | `(-4.814, -0.015, 0.007)` | `4.808/1.106`, `4.813/1.105` |
| LIFT | `(4.839, 0.010, 1.610)` | `(-4.858, -0.010, 1.612)` | `4.839/1.755`, `4.858/1.756` |
| SUSPENDED_HOLD | `(4.852, 0.015, 0.308)` | `(-4.854, -0.016, 0.315)` | `4.852/1.011`, `4.854/1.016` |

run06 LIFT 的切向滑动速度均值/p95/p99/max 为 `23.095/39.653/44.435/51.301 mm/s`；SUSPENDED_HOLD 为 `4.762/12.816/16.194/18.753 mm/s`。按 condim4 pyramidal 的局部摩擦棱锥 L1 利用率计算，run06 LIFT 每接触点均值约 `0.264`、p95 `0.456`、p99 `0.547`、最大 `0.599`，明显未到边界。run06 hold 均值约 `0.150`、最大 `0.323`。名义圆锥上限 `μFn` 约 `6.8 N/侧`，对称静态承重最低需约 `2.796 N/侧`竖直分量；实际 lift 只有约 `1.61 N/侧`，hold 约 `0.31 N/侧`。摩擦上限没有被实现为实际力。

## 控制器、滑移与求解器

- 指垫 position servo：`kp=1000 N/m`、`kv=20 N·s/m`、力限 `±8 N`；目标闭合行程保持 `0.022 m`。run06 HOLD 法向力约 `4.85 N/侧`，实际平均 actuator force 也约 `4.85 N`，最大约 `5.06 N`；无饱和。位置误差约 `4.85 mm`，符合接触载荷下有限刚度 position servo 的弹性偏差。lift 开始时目标没有变松，未见 preload 释放或 follower/耦合现象。
- Lift servo：`kp=800 N/m`、`kv=60 N·s/m`、力限 `±25 N`。run06 carriage 最大速度 `0.03769 m/s`，目标 smoothstep 最大速度 `0.0375 m/s`；平均绝对 tracking error `9.75 mm`、最大 `12.48 mm`，最大 actuator force `8.00 N`，没有触及 25 N 限值。runner 计算并记录目标速度，但 position actuator 没有单独的目标速度前馈；这解释了部分 carriage tracking lag，但 carriage 实际上升约 `46.1 mm`，瓶身相对 carriage 滑移 `49.85 mm`，不能解释瓶身几乎未离桌。
- run05 实际瓶身峰值上升 `16.78 mm`，suspended-hold 有 `295/500` 步桌面接触；run06 峰值仅 `0.0276 mm`，hold 有 `500/500` 步桌面接触，底面最终仍低于桌面约 `0.876 mm`。run06 放慢后接触继续存在但相对滑动更小，静态竖直摩擦仍不足以接替桌面载荷。
- 既有离线快照审计对所有 run05/run06 trace 状态重建后执行 `mj_forward`，并清零 solver warmstart；run05 最多 5 次 Newton iteration，run06 最多 4 次，均未接近 80 次上限，最终 scaled-gradient norm 最大分别约 `7.34e-15`、`1.17e-14`。这说明已记录状态的重建求解容易收敛，但不是原始积分步骤的 solver stats；原运行器没有保存每一步的 solver 统计，因此不能宣称已逐步排除求解器问题。

## 修正、验证与边界

研究运行器 `scripts/research/issue46_ideal_parallel_gripper_baseline.py` 现用 `signed_vertical_force_summary()` 同时记录净值、正向和负向分量；phase/result 字段改成明确的 `net_force_world_z_N`，悬空门槛也使用桌面净力。trace 新增 `contact_force_evaluation_time_s`。没有更改物理模型、摩擦、重力、执行器、轨迹或验收阈值。

验证：

- 三项回归精确命令（需在本仓库根目录执行）：

  ```bash
  PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages \
  /usr/bin/python3 -m unittest scripts.research.test_issue46_ideal_parallel_gripper_baseline -v
  ```

  结果：3 项通过（空接触、相反方向分量、support class 分别求和）。
- 两个研究脚本 `py_compile`：通过。
- MuJoCo 3.3.6 MJCF 编译及单步 smoke：通过；首次无瓶身接触时统计为零，模型时间 `0 → 0.002 s`。
- `git diff --check`：通过。
- 没有重跑 run05/run06 全序列；既有原始结果、JSONL、MJCF、MP4 均保持原 hash。

历史 run05/run06 物理实验使用各自 `result.json` 中的 `reproduction_command`。run06 将 `LIFT_50MM` 从 0.60 s 改为 2.00 s；模型和其他物理/控制参数哈希相同。

本工作区分支头为 `0c6ad3ef7c27eacd458c79a90bfaa8f6c7abdb4e`，比远端领先 6 个既有本地提交。此次 runner/test/report 是本地未提交修改；未改分支、worktree 元数据、PR #58 或远端。

## 唯一建议的下一步

仅做一次**固定初始状态的短时切向剪切 A/B 诊断**：从同一 run06 bilateral-hold 快照恢复 robot-free fixture、自由圆柱和夹爪状态；保持瓶子、质量、几何、摩擦、重力、执行器、目标与 solver 数值参数不变，只在对照模型中把 `cone` 从 `pyramidal` 换成 `elliptic`。对两者施加同一小段竖直 carriage reference，记录同步的 solver stats、逐 pad 瓶身世界力、实际切向摩擦锥利用率和相对滑动。该试验只判断摩擦锥/软接触实现对未饱和滑移的影响，不是 50 mm grasp acceptance，也不授权进入 native X2 Phase B/C。
