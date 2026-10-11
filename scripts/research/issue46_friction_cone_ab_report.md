# Issue #46: ideal parallel-gripper friction-cone A/B

## 结论

在这次固定台架、短时垂直剪切中，`elliptic` 相对 `pyramidal` 提高了实际手部竖直支撑并降低了指垫相对滑移；但圆柱最大只升高 **0.936 mm**，且 **97.4%** 的采样仍有桌面接触，没有形成悬空保持。该结果说明接触响应对 cone 形式敏感，不能证明锥近似或求解器是 run06 失败的唯一根因，也不是抓取通过。

## 受控条件与身份

- A=`pyramidal`，B=`elliptic`。两份 MJCF 除 `<option cone>` 外相同；编译拓扑、质量/惯量、几何、摩擦、关节和执行器数组逐位相同，重力、步长、积分器、solver、迭代上限、容差和 `impratio` 相同。
- 从 run06 `BILATERAL_HOLD_ON_TABLE` step 1100、`t=2.2 s` 恢复；A/B 的 qpos/qvel/ctrl SHA-256 均为 `fba7f223bb7045d89ae26dc23084bdb57b2e0b1353593a800480220f7c18620b`。恢复位置和四元数误差为 0；刚体 twist 最大误差 `6.94e-18`；清零 warm-start/外力后调用 `mj_forward` 重算派生状态。
- 每个 cone 运行 500 步（dt=0.002 s）：10 mm 升降台 smoothstep 参考，0.5 s ramp + 0.5 s hold；指垫目标各 0.022 m。free cylinder 为直径 70 mm、高 244.5 mm、质量 0.570 kg；无运行中瓶体 qpos 写入、weld、mocap 或外加物体力。
- MuJoCo 3.3.6；Python 3.10.12；native `libmujoco.so.3.3.6` SHA-256=`b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`。模型：timestep 0.002、gravity `[0,0,-9.81]`、`implicitfast`、Newton、80 iterations、tolerance `1e-10`、impratio `1.0`。接触为 condim 4、solref `[0.02,1]`、solimp `[0.9,0.95,0.001,0.5,2]`；每个已编译接触 friction 为 `[1.4,1.4,0.02,0.001,0.001]`。

## 量化结果

下表 ramp 是升降台开始运动后的前 0.5 s。力为逐接触 `mj_contactForce` 经 MuJoCo 接触坐标架变换到**作用在瓶体上的世界坐标力**；手部竖直支撑是左右指垫各一次有符号 world-Z 力之和。

| Ramp 指标 | A 金字塔 | B 椭圆 | B 相对 A |
|---|---:|---:|---:|
| 手部竖直支撑均值 | 1.761 N | 2.576 N | +0.815 N / +46.3% |
| 桌面竖直支撑均值 | 3.830 N | 3.017 N | -0.813 N |
| 左/右指垫法向力均值 | 4.859 / 4.873 N | 5.572 / 5.588 N | 增加 |
| 左/右指垫切向力幅值均值 | 1.195 / 1.185 N | 2.427 / 2.448 N | 增加 |
| 左/右指垫有符号世界 Z 力均值 | +0.887 / +0.874 N | +1.290 / +1.286 N | 增加 |
| 左/右相对切向滑速均值 | 10.969 / 10.924 mm/s | 5.660 / 5.684 mm/s | 约降低 48% |
| 圆柱最大上升 | 0.283 mm | 0.936 mm | 仍远低于 50 mm |
| 桌面仍接触的样本 | 250/250 | 240/250 | B 有 10 个 ramp 采样无桌面接触 |

在后续 0.5 s hold 中，手/桌面支撑均值分别为 A `1.203/4.389 N`、B `3.463/2.126 N`；双指接触两者均为 100%。全程 B 有 487/500 桌面接触样本，A 为 500/500；B 的 13 个无桌面接触采样最长即使连续也仅 26 ms，不能算悬空保持。B 圆柱最后相对初始高度仅 `+0.396 mm`；A 为 `+0.066 mm`。

逐接触检查未把平均利用率当作约束活跃判据：

- 指垫实际 cone 利用率：A ramp 左/右最大 `0.385/0.360`；B 最大 `0.902/0.839`，峰值分别在 step 220 (`t=2.640 s`) 和 step 216 (`t=2.632 s`)。B 的指垫仍未达到本诊断归一化指标 1.0，但左指已到约 90%。
- 桌面接触是关键例外：B 的 full-condim4 利用率 `>=0.99` 在 ramp/hold 分别出现 77/250、73/250 个接触样本，峰值约 1.0（step 185、`t=2.570 s`；step 256、`t=2.712 s`）。A 桌面峰值为 ramp `0.756`、hold `0.664`。
- 每样本实际接触为两枚指垫接触及桌面接触（B 桌面偶尔丢失）；没有额外接触对。contact position、局部力、世界力、滑速和每个 EFC row 均逐步保存在 JSONL。MuJoCo contact frame 的第 1 轴是法向，力变换按接触 frame 转成 world 后以作用在瓶体的方向记号保存，符合[官方计算文档](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html)。
- MuJoCo 记录的指垫 contact rows 在椭圆锥下每接触 4 行、金字塔锥下每接触 6 行，观测行状态均为 `mjCNSTRSTATE_QUADRATIC`。此状态表示约束代价处于 quadratic 区域，不等同于摩擦锥边界标志；这里分别另算单接触 cone 利用率，且结果 JSON 明确标记它不是 solver active flag。[MuJoCo 3.3.6 类型文档](https://mujoco.readthedocs.io/en/3.3.6/APIreference/APItypes.html)列出 `SATISFIED`、`QUADRATIC`、`LINEAR*` 和 `CONE` 的不同含义。

## 求解器与执行器

- Newton 求解步迭代数 p95/max：A `2/3`，B `4/6`；记录的 solver gradient 最大值 A `2.088`、B `4.499`。两种配置都无 MuJoCo warning、状态非有限值或执行器饱和。B 迭代较多，但当前模型 `enableflags=0`，故 trace 中的 `solver_fwdinv=[0,0]` 不是 forward/inverse 收敛证明；现有记录也不足以将差异归因于 solver 不收敛。
- 指垫各自目标为 22 mm，实际位置约 `16.3–17.2 mm`；平均绝对 tracking error A 约 4.85 mm、B 约 5.54–5.69 mm；最大力分别低于 8 N 限值。
- 升降台目标从约 10 mm 到 20 mm，但 ramp 末实际位置 A=`11.627 mm`、B=`9.541 mm`（起始均 `6.072 mm`）；hold 末 A=`15.316 mm`、B=`12.699 mm`。两者都未完成 10 mm 参考运动且均未触及 25 N 力限。相同命令下的实际机构轨迹不同，是后续分离执行器跟踪与接触本构时必须控制的变量。

## 工程判断与下一步

本次 A/B 显示 cone 选择改变了软接触响应和载荷分配；椭圆锥获得了局部改善，但不能解决瓶体抬升/悬空保持失败。它足以排除“二者在此 fixture 中完全等价”，不足以证明原金字塔锥是 run06 失败的主因，也不足以证明 Newton 求解行为异常。MuJoCo 3.3.6 文档明确说明两种 cone 会产生不同软接触动力学，且其接触力锥是按 condim 定义；本实验的逐接触力与此一致，但不是对真实材料的标定。

唯一建议的下一项：做一次同一预先记录的**实际**升降台 qpos 轨迹下的配对小幅垂直剪切，对两个 cone 分别记录末端求解梯度/残差、逐接触力锥利用率和滑速；同时保留原执行器限值，另用无物体的同命令短检查标明执行器跟踪误差。该测试是机制隔离诊断，不是抓取验收。未通过独立台架真实 50 mm 抬升和 1 s 悬空保持前，不进入原生 X2 几何/抓取阶段。

## 复现

先前已提交的有符号接触力仪器修复：`8916b32`；A/B runner：`19c97d6`。以下实际实验命令使用固定 run06 证据；runner 会拒绝覆盖已存在的输出目录，复跑时应传入一个新的空目录。

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011
SITE=/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages
PYTHONPATH="$PWD:$SITE" \
LD_LIBRARY_PATH="$SITE/mujoco" \
MUJOCO_GL=egl \
/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_friction_cone_ab.py \
  --source-dir /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-ideal-parallel-gripper-20261011-run06-slow-lift-2s \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-friction-cone-diagnostic-20261011

/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/analyze_issue46_friction_cone_ab.py \
  /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-friction-cone-diagnostic-20261011 \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-friction-cone-diagnostic-20261011/analysis.json
```

原 run06 输入 SHA-256：MJCF `b44a29746c1a7871236f47b96ca6d086ea5f6533ebcbe8014dea6490a2ebf24e`；trace `3bed5cdb90f651e05fb8560a145fd71e23eccb5f626eb9825e5a05468463e770`；result `bf161505b8bd325108844f0007695d58c1ea204fdf2844110a0c93a2fa2fbb24`。逐步 A/B trace 哈希分别见 `steps_pyramidal.jsonl` 和 `steps_elliptic.jsonl` 的证据 manifest/result。

本次产物 SHA-256：金字塔 MJCF `13cec48b5d9bf2b68b8622461c334eabdc800e92c0ff1f60975cc5f03168a75b`；椭圆 MJCF `abdd0ff11b17e12af4aa75d7d5a204b33605bf2e10de5e1262187d3d5f12611c`；金字塔 trace `6f86376959ac38d1cb3d87a6d4c5a4539984e27bde4ad3e5dab72aefd45b3e70`；椭圆 trace `16bd7cb9c561ef9429043e0824794373f1b9a04f8b52b641c84e502f0e5163b3`；result JSON `552ca487984f2f10ab62409ea57a345e78d70fbdf4062de5dcabd7149848ccfb`；post-analysis JSON `ba672a3fe763e757a9e32f2c20d3c16a154595237a20140bf170e2c1c636ac84`。物理 runner SHA-256=`8ec95d61948aab94e85279099e2d882dbb0a97df8a04394e60f2cac00edfea94`；只读分析器 SHA-256=`5f9cd8e7f41199cf2cbb27e0543a8afed63c826c751eed5bdce2c47447031d78`。

## 证据目录

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-friction-cone-diagnostic-20261011/`

包含两种 MJCF、逐步 contact/force/solver/actuator JSONL、同状态快照、result JSON 和只读 post-analysis JSON。run05/run06 原目录未修改。所有结果均为独立 ideal parallel-gripper 台架的 SIMULATION_ONLY 诊断，不是 X2 原生抓取验收。
