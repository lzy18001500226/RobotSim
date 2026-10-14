# Issue #46：X2 OmniPicker 连杆几何根因审计

## 结论

**BLOCKED — 当前拟合销位候选没有来源依据的被动运动学，不能据此做机械闭环或硬件可行性结论。未找到有证据支持的固定修正，因此没有改动模型。**

本任务只做离线几何及编译模型静态检查。对 MuJoCo 3.3.6 使用 `mj_forward()`；`mj_step()` 数为 0；未运行瓶子、接触动力学或抓取。数值上的 `0.01 mm` 闭合值是既有诊断阈值，不是供应商公差。

证据目录：`/home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-linkage-root-cause-20261014-597a9717/`。该证据目录在主 RobotSim checkout 中，作为本地审阅产物保留；代码和本报告位于隔离研究分支。

## 根因判断

| 问题 | 观察 | 判断 |
|---|---|---|
| 坐标/安装变换错误 | 官方 X2 固定连杆模型与插入被动铰链、令其为零的拟合候选，在 5 个开合点、8 个 link frame 上比较；相对基座最大位置差 `1.63e-14 mm`、方向差 `2.1e-6°`。 | 本次检查的 link origins、安装变换及根关节映射一致，未发现坐标复制错误。此检查不验证推算出的被动轴。 |
| 销位残差 | 全行程 101 点的残差方向和大小不变；被动角 Jacobian 数值秩为 2，残差完全落在可动子空间外。 | 该候选的三维 pin-center 误差不能由当前三根被动转轴的转动消掉。销中心来自 STL 孔特征拟合而非 CAD 基准；与拟合误差同量级。无法区分 STL 拟合偏差、真实间隙/装配基准偏差或候选轴定义错误。 |
| 过度手指旋转 | 候选的 `narrow3` 在 `u=0.50` 达 `+2.3896 rad`；相邻 `0.01 rad` 主动角步出现 `0.7588 rad` 的窄侧末端跳变。 | 主要是欠约束候选的分支选择/连续性问题。connect 约束的被动角 Jacobian 秩 2，且源 URDF 没有这些被动 DOF、限位或运动律；最小二乘延续不是硬件机构定律。 |
| `14.201 mm` narrow2/loop 重叠 | 在候选 `q=-0.5` 时 MuJoCo 编译 hull 接触距离 `-14.200609 mm`；同一候选 link transforms 下，原始 STL 三角面存在至少 20 个交线见证。官方固定连杆源模型同一主动角下无该 pair 活动 hull contact，也无 STL 三角面相交。 | 不是“只有凸包造成”的假阳性；候选分支确实令网格表面相交。但交叉仅在未验证的候选被动角下出现，不证明官方真实机构会发生干涉。凸包穿透值不能当作三角网格的真实穿透深度。 |
| 最小固定修正 | 没有官方 pivot datum、轴向定义、被动限位、杆长/间隙和装配公差可用。 | 不调整 pivot、轴、限位或 equality 参数。任意修正会拟合某条分支而非恢复来源可追溯的机构。 |

## 固定闭环残差

| 侧 | 残差范数范围（101 点） | 世界坐标残差分量（约） | 被动 Jacobian 秩 | 最大残差在可动子空间外 |
|---|---:|---|---:|---:|
| narrow | `0.087641564370302–0.087641564370330 mm`；峰峰值 `2.76e-14 mm` | `(+0.087633990, -0.001152224, +0.00000037) mm` | 2 | `0.087641564 mm` |
| wide | `0.120082976478543–0.120082976478573 mm`；峰峰值 `2.98e-14 mm` | `(-0.120072598, +0.001578730, -0.00000051) mm` | 2 | `0.120082976 mm` |

轴方向核验解释了残差为何几乎不随开合变化：候选把 narrow2/narrow3/narrow-loop 及 wide2/wide3/wide-loop 六根轴都设成各自 link frame 的 local `+Z`。在 `q_driver=-0.5 rad` 的编译姿态下，它们全部映射到世界方向约 `(0.999913575, -0.013146967, 0.000004246)`；101 点中候选轴两两夹角最大数值为 `0°`。这只是候选的建模假设；官方 URDF 把这些 link 连接定义为 `fixed`，没有相应的被动转轴可供核对。

在相同假设下，narrow 与 wide 销位残差分别与该轴平行到全行程最大 `1.06e-5°` 和 `1.54e-5°`；垂直于轴的残差最大仅 `1.63e-8 mm`、`3.24e-8 mm`。101 点被动角 Jacobian 的奇异值范围为：narrow `σ1=0.03569–0.04623`、`σ2=0.01554–0.02763`、`σ3=4.47e-15–2.40e-12 m/rad`；wide `σ1=0.03984–0.04593`、`σ2=0.01607–0.02667`、`σ3=2.92e-15–2.24e-12 m/rad`。因此当前轴构造的可动子空间只有二维，残差沿轴方向的分量无法由这些被动转角改变；近零 `σ3` 是这种局部平面运动的数值证据，不是厂商机构规格。

候选销位用主轴方向的 STL facet-normal 圆柱拟合得到，不是 CAD 测量。所选 distal pin 特征拟合 RMS：narrow3 `0.1300 mm`、narrow loop `0.1216 mm`、wide3 `0.1175 mm`、wide loop `0.1178 mm`。它们与闭合残差同量级，说明当前差值不足以判定真实机械不兼容，但也不足以通过精确闭环验证。

残差的近似固定方向来自连接点可达位移子空间缺少一个方向，而非随着开度增长的误差。优化器成功收敛只表示找到了局部最小值；它不表示三维销位重合。

## 轴、行程与运动分支

固定源参考：`AgibotTech/agibot_x2_urdf`，commit `575cc6b988f976c23550e0db85aa1e5475d3652d`，`X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`，SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`。该 URDF 的 `right_claw_joint` 范围 `[-1,0] rad`、轴 `(0,0,-1)`；`R_hand_wide1_joint` mimic 系数 `-1`。narrow2/narrow3/loop 和 wide2/wide3/loop 的关节类型为 `fixed`。这些 fixed joint 上的 axis/limit XML 标签不创建被动转动 DOF，也不构成可用的机械被动限位。

拟合候选将六个 fixed child/loop 关系改成无范围的 local-Z 被动 hinge，插入 root mimic equality 和两个 distal connect equality；销位取自 STL 拟合，且没有真实连杆长度/销孔配合基准。故候选的 6 个被动角只是满足有限约束的一个优化分支，并非源 URDF 的运动定义。101 点的主动 root `right_claw_joint` 均在官方 `[-1,0] rad` 范围内，mimic 目标按源系数 `-1` 生成；候选被动 hinges 是 `limited=false`，源模型没有对应的被动 joint limits，因此不能报告“所有候选关节均满足来源限位”。

| 开口 `u` | root q（rad） | 候选 narrow2 / narrow3 / loop（rad） | 候选 wide2 / wide3 / loop（rad） | `narrow3` 与 `wide3` frame 角差 | 编译 distal carrier gap |
|---:|---:|---|---|---:|---:|
| 0.00 | 0.00 | `-0.0199 / +0.0381 / -0.0117` | `-0.0370 / +0.0646 / -0.0217` | `0.540°` | `-0.058 mm` |
| 0.25 | -0.25 | `-0.3323 / +1.0074 / +0.2990` | `+0.4664 / -1.9104 / -0.4592` | `150.069°` | `38.365 mm` |
| 0.50 | -0.50 | `-0.5168 / +2.3896 / +0.9649` | `+0.5123 / -1.9880 / -0.8354` | `110.855°` | `59.173 mm` |
| 0.75 | -0.75 | `+0.1627 / -0.6414 / +1.2637` | `+0.4548 / -1.2013 / -1.0178` | `101.288°` | `82.463 mm` |
| 1.00 | -1.00 | `-0.4447 / +0.8172 / +1.3080` | `+0.5909 / -1.3492 / -1.3346` | `179.384°` | `88.416 mm` |

该 carrier gap 并非真实 pad-to-pad 间距；相对 link frame 角也不是 pad-face 法线夹角。当前 STL 没有可验证的功能指腹面语义，因此不能从这两个 proxy 宣称“能夹 70 mm 瓶”或“开口/闭合姿态正确”。

### 不要混淆两个候选数据集

较早 Local03 aperture viewer 的 `X2 SIMULATION_ONLY candidate` 是另一个静态对照扫描：carrier gap 在 `u=0/.25/.5/.75/1` 为 `0.307/35.637/58.013/76.353/89.376 mm`，terminal frame 角为 `96.551/90.921/103.599/117.601/129.774°`。其中 70 mm 只和 carrier gap 代理比较；full-open frame angle 比官方 X2 大约 `18.96°`、比 G2 大约 `37.40°`。

本报告五点 fitted-candidate 表直接取自后续 `constrained_candidate.xml` 和已保存 101 点静态扫描，两套数据在闭合端 gap 为 `+0.307 mm` 与 `-0.058 mm`，不可当成同一次状态或同一条唯一运动曲线。差异符合候选被动解不唯一/分支依赖的风险，不能据此做零位修正。

三模型既有 aperture 对照数据如下。X2 源 q 为 `(-u,+u)`，G2 mimic q 为 `(-u*pi/4,+u*pi/4)`，拟合候选对照使用自身拟合关节解。

| u | X2 源 carrier gap / frame angle | G2 源 carrier gap / frame angle | 较早 X2 fitted candidate carrier gap / frame angle |
|---:|---:|---:|---:|
| 0.00 | `0.059 mm / 96.634°` | `0.070 mm / 98.580°` | `0.307 mm / 96.551°` |
| 0.25 | `38.528 / 90.242°` | `28.755 / 93.561°` | `35.637 / 90.921°` |
| 0.50 | `62.553 / 90.768°` | `51.775 / 90.739°` | `58.013 / 103.599°` |
| 0.75 | `84.504 / 98.098°` | `69.255 / 90.331°` | `76.353 / 117.601°` |
| 1.00 | `103.155 / 110.816°` | `84.908 / 92.371°` | `89.376 / 129.774°` |

所有间距与角度均为原静态脚本的 collision carrier/frame proxies，而非经确认的抓取面测量。G2 使用独立官方 Xacro：`AgibotTech/genie_sim_robot_model` commit `6d5b2159806ae0e64b432b638e9ee288d8bddc8f`，`robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro`，SHA-256 `b20d8691b03a256ad3481380b71fd1e1eaa11495673f4bf7fd7377f67238d183`。G2 仅作拓扑/运动参考，不是可替换的 X2 机构定义。

## STL 交叉与凸包检查

`q_driver=-0.5 rad` 的 fitted-candidate 分支：`x2cand_x2geom_4_collision`（narrow2）与 `x2cand_x2geom_8_collision`（narrow loop）在 `mj_forward()` 后产生 1 个 active contact，signed hull distance `-14.200609 mm`。对相同 link transforms 变换二进制 STL 顶点并做三角面对三角面窄相测试，142 个邻近候选三角面对中至少 20 个存在表面交线。故“完全由 convex hull 造成”被排除；但 hull 数值是凸包穿透代理，不是原始网格的 penetration depth。

同一 `q=-0.5 rad`、源 `wide1=+0.5 rad` 的官方 X2 固定 child/link 对照：`x2src` narrow2/loop 无活动 hull contact，原始 STL 三角面检查无相交。这个对照说明干涉是当前 fitted passive branch 引入，而非原始固定装配在该状态的证据。它仍不代表真实 X2 的动态被动连杆轨迹，因为 URDF 没有把这些 link 定义成被动铰链。

Mesh inventory：narrow2 STL 31,926 三角面，loop STL 10,302 三角面；两者无边界边，narrow2 有 2 条非流形边、loop 为 0。该表面相交测试支持“在候选变换下网格相交”的存在性，不计算精确实体重叠体积，也不判断这些相交在真实机械组装中是否被允许。

## 处理决定与下一步

- 没有改官方 X2/G2 URDF、mesh、limits、碰撞过滤或之前的拟合候选。
- 没有做动力学、viewer 动画、瓶接触或抓取。
- 不提交交互 viewer：目前无来源依据的“修正候选”可供公平比较。
- 唯一建议：先获得 X2 OmniPicker 的权威 pivot center/axis、杆/销实际连接关系、被动 joint 限位/传动规律及装配间隙资料（CAD、图纸或可追溯实测）。再用这些参数替换 STL 估点，并重跑同一静态全行程闭合门。没有这些信息前不再调 pin、轴、offset、equality 或分支初值。

## 五项问题结论

综合根因分类为 **D（组合）**：坐标复制/根关节映射错误在本次对照中未发现；当前失败由来源未定义的平行被动轴假设与 STL 拟合销位共同决定，残差大小又与 STL 拟合误差相当。尚不能断言原始 X2 硬件几何不兼容。

1. **残差为何恒定：** 在这份候选中，拟合残差几乎完全沿假设的共同 hinge 轴，转动坐标不能消除轴向销位差；其数量级与 STL 孔特征拟合 RMS 相当。现有证据不能把 mesh 拟合/基准误差和真实装配间隙区分开。
2. **手指为何过度外翻：** 候选被动转轴无源定义、无范围且闭环欠约束，最小二乘延续在局部分支间跳变；`narrow3` 最大相邻点跳变 `0.7588 rad`，`wide3` 为 `0.4253 rad`。在 `u=1` 时，后续 constrained candidate 的 terminal-frame 角 proxy 为 `179.384°`，而较早 aperture candidate 为 `129.774°`；X2/G2 静态参考分别为 `110.816°/92.371°`。候选数据来自不同静态解，不能拼成单一运动曲线。它不能证明官方手指应复制 G2 姿态，但说明当前候选外翻来自未验证的运动分支。
3. **14.201 mm 是什么：** 对当前拟合分支的相同 STL 姿态检出至少 20 个三角面交线，故不是只有凸包产生的假相交；官方 fixed-child 对照无交线。不过它只说明该推算分支确实自相交，不说明真实 X2 机构会如此运动。
4. **能否做固定修正：** 不能有依据地修正。缺真实 pivot/axis/装配基准，改 pin 或 axis 会拟合假设而非恢复来源定义；不对候选做修正。
5. **能否进入约束动力学：** **BLOCKED**。被动运动学尚无来源支持，静态闭环 residual 不为零，且拟合分支在若干姿态出现网格相交；不应以当前候选启动受约束动力学，更不能开始抓取。

所需补充的最小机械资料：每个被动铰链/销中心的三维坐标和轴向（明确测量坐标系）、各窄/宽连杆两端销距与装配关系、loop link 与哪些销/孔构成真实闭环、被动关节硬限位或确定的传动关系，以及销孔配合/间隙或可追溯装配公差。STL 拟合点不能替代这些 CAD datum。

## 复现与来源

完整命令、依赖及输入 SHA-256 见证据目录 `README.md`。主要复现程序：[`issue46_omnipicker_linkage_root_cause.py`](../../scripts/research/issue46_omnipicker_linkage_root_cause.py)。输入包括已保存的 Local02 101 点 static scan、Local03 aperture CSV、源 X2/G2 比较模型以及此前只读归档的 STL；完整重跑调用 `mj_forward()` 并输出 CSV/JSON/PNG，不调用 `mj_step()`。

```bash
env MUJOCO_GL=egl /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  /home/lzy18001500226/robotsim-issue46-omnipicker-linkage-root-cause-20261014/scripts/research/issue46_omnipicker_linkage_root_cause.py \
  --model /home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-linkage-physics-20261014-df89d796-4673-445f-b003-d65f6dc689e5/constrained_candidate.xml \
  --source-reference-model /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-g2-omnipicker-compatibility-20261013/x2_g2_omnipicker_review.xml \
  --static-scan /home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-linkage-physics-20261014-df89d796-4673-445f-b003-d65f6dc689e5/static_scan.json \
  --aperture-csv /home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-joint-aperture-20261013-run02/aperture_scan.csv \
  --source-mesh-dir /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-variant-b-full-x2-geometry-feasibility-20261012/source_snapshot/meshes \
  --reference-images /home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-joint-aperture-20261013-run02 \
  --out /home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-linkage-root-cause-20261014-597a9717/refined_diagnostics
```

最终分析器 SHA-256：`7b57751842416294683b9305666a0ed7752614445550c85bd046ef84d7c7690f`。静态重跑和 `python3 -m py_compile` 均通过；未执行动力学。完整最终 JSON、CSV、图像在证据目录 `refined_diagnostics/`；初次静态结果仍原样保留在证据根目录。

任务分支：`research/issue46-omnipicker-linkage-root-cause-20261014`。基线为 Local02 `30a00c92ada270a49799cdf680a6f147f700031a`。本报告本身不改 PR #67。
