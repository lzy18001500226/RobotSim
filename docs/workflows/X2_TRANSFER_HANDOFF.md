# Local03 → Local02：X2 两桌搬运交接

更新：2026-11-10。只需先读本页，不用加载旧聊天/session。
本次仅保存检查点，不合并 main、不删除工作树、不启动仿真。旧 Goal 保持暂停。

## 分支与工作树

- 工作树：/home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030
- 分支：codex/x2-autonomy-task-20261031
- 交接前实现 HEAD：f43a6da2649dc93f9124ff576281c64269269877
- [PR #106](https://github.com/lzy18001500226/RobotSim/pull/106) 保持 OPEN / Draft；
  [Issue #46](https://github.com/lzy18001500226/RobotSim/issues/46) 保持 Open。
- #80/#81 是历史导航任务，不要切回旧导航工作树。
- 实查共3个工作树：项目根目录、agents-skill-routing、上述X2工作树。其余不动。
- 不合并 main 的原因：22厘米整段未完成，当前配置的 START 复现未收尾，
  PR 的 Agent infrastructure checks 仍有失败。新线程可以直接复用本工作树。
- Git 操作不用于修改/删除旧 session；会话卡顿原因本次未作性能诊断。

## 已验证的实际进度

证据根目录：/home/lzy18001500226/projects/RobotSim/reviews/issue-46/

| 目录 | 实际结论 |
| --- | --- |
| raicom-stl-xvfb-native-reset-stand-20261110T053739Z | 30厘米固定桌前开始：真实相机抓取、抬升、收臂、后退/转向、走到第二桌、完整停稳、松爪撤手及桌面独立支撑2秒全部 PASS。failure=null、清理确认成功。采用明确标记的 MULTICCD 数值配置。START→取瓶桌为 NOT_RUN。 |
| raicom-stl-xvfb-native-reset-stand-20261110T035240Z | 22厘米真实相机抓取/抬升 PASS：圆心误差1.263毫米，最小抬升58.770毫米，悬空保持1秒，桌面力0；独立4027帧源关节/姿态审计 PASS，右爪最大1.864Nm。尚无22厘米恢复正常站姿、搬运或第二桌放置通过结果。 |
| raicom-stl-xvfb-native-reset-stand-20261110T060108Z | 当前配置最后一次从 START 复现被中断：有MC/native日志，但没有 result.json，不能报 PASS 或已清理。外层目录为 live-retreat-turn-human-review-command-20261110T060002Z。 |
| raicom-stl-xvfb-native-reset-stand-20261109T221003Z | 较早配置完成过一次从 START 到第二桌；不能代替当前 MULTICCD 配置的完整复现。 |

041253Z实际物理放置通过但父进程结束确认超时；042535Z、044427Z放置稳定门失败，
历史结果保留。不要把它们改写成完整运行 PASS。
详细原因见[现有执行报告](../research/issue46_truth_grasp_execution_findings.md)。

## 当前配置，别重新发明

- 已审核X2/OmniPicker STL、装配、开合机构不变。真实厂商MC和MuJoCo3.3.7，
  x86 Humble/AimDK通信；不是硬件验证，也不宣称未修改厂商默认资格。
- 22厘米抓取截面为距瓶底12厘米，即COM上方1厘米。
- 侧身是转腰yaw，不是左右侧弯。用转腰＋前弯＋必要的小幅原生下蹲，
  roll保留小幅实测平衡值。
- 沿用kp6、原2Nm夹爪上限、内部frictionloss0、既有22块源派生包胶代理、
  elliptic/impratio10、NoSlip0及局部摩擦参考0.002s。
- 力控目标参数必须沿用 --carry-grip-force-margin 15.0，不是“15%力度”；
  H220曾误用默认1.5，导致承重不足。
- --multiccd 是单独的数值对比配置：enableflags0→16。427项编译数组保持一致；
  不改网格、质量、材料、增益、源关节/执行器限制或停止阈值，不添加支撑。
  native必须实测加载16。053739Z已通过该配置整段固定桌前搬运。
- 相机是任务蓝色、竖直圆柱RGB-D几何估计，不能称为已完成通用YOLO/6D识别，
  光学标定也不能当作厂商真机标定。
- 22厘米用桌外抬手＋高位接近。中间姿态按当前pelvis相对坐标检查；
  最终抓取仍对齐新鲜相机的世界坐标瓶心。冗余IK须预留控制补偿余量。
- 安装仿真位置模式的effort前馈行为有实测差异，22厘米分段路径用了受限
  bias/kp适配；不得直接当真实硬件接口合同。冻结增益不能在实时tick重复解析YAML。
- read_fresh_native_row返回完整状态；require_fresh_sample只返回新鲜度判定。
- 复用已验证配置的开合证据和IK初值，实际相机、姿态、到位、承重和停止门仍重算。
- 参数以现有 reviewed_carry_step_forward.json 为准：departure0.12、
  STEP直行SDK目标0.20m/s、compact turn上限0.60rad/s、独立导航/制动根部上限0.60，
  站立仍0.20。这些是现有实验配置，不是厂家源限制或继续提速授权。

## 下一步

1. 恢复Docker连接后先核对旧容器及进程，不从中断最后一帧接续控制。
2. 收尾当前MULTICCD配置的完整START复现；固定桌前整段已PASS，不重做CAD、
   圆拟合、同样库存审计或无变化的开合测试。
3. 22厘米：抓起/收臂后保持夹持，原生恢复身高，将腰部交回实际MC输出，
   完整站稳后才接回既有搬运与第二桌释放流程。
   不能直接删协调腰carry禁止门，也不能让机器人带未验证的低姿态直接走。
4. 前两阶段收敛后，做“只知桌位→相机找未知瓶→可达则抓→必要时有限站位调整
   →停稳重新观察→抓取搬运”。不开展语音、新Nav2/SLAM或新本体。

沿用原route manager、point-goal、M1桥、AimDK/ROS与原MC，不建第二套框架。
核心入口为 scripts/research/issue46/ 下的 grasp/live_retreat_turn_review.py、
stl_approved_circle_native_trial.py、stl_approved_circle_ik.py，
以及 native_runtime/xvfb_native_reset_stand.py、stl_hand_startup.py、
native_body_height.py、native_carry_placement.py。

native_runtime/native_grasp_stand_restore.py 是保留下来的未接入草稿：
当前无运行调用者，只做过Python/生成脚本语法检查，没有真实ROS/持瓶恢复验证，
所需子脚本尚未接线，不能当成已实现功能调用。
处置为 PORT-TO-C++：先保留隔离研究原型，实际流程合格后再迁移生产，不被生产包导入。

## 复现入口与环境

以下命令本次未执行。是待确认的完整START入口，不是当前配置已稳定复现声明。
WSL终端：

    cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030
    DISPLAY=:0 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46/grasp/live_retreat_turn_review.py --to-delivery-table --from-start --multiccd

不加 --from-start 为053739Z通过的固定桌前范围。
默认显示Windows只读实时MuJoCo视图；真实物理/MC在WSL容器。
--no-viewer只供无人审核诊断，不能冒充可视入口；--check-only仍依赖Docker。

PowerShell直接传参数，避免嵌套bash引号：

    wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec env DISPLAY=:0 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46/grasp/live_retreat_turn_review.py --to-delivery-table --from-start --multiccd

22厘米成功诊断准确命令见证据根目录下
h220-command-margin-grip-lift-live-20261110T035154Z/command.json；
它是后台Xvfb诊断，不是22厘米整段可视入口。

暂停后的最后检查：WSL Docker集成不可用，Windows LinuxEngine管道未找到，
没有本任务的WSL启动器进程。旧容器名由真实启动记录确认：
robotsim-i46-xvfb-0261110t060108z。守护进程不可达时不能确认它已被移除；
本次不启动Docker服务或新仿真。接手时重新检查。

## 保留本地状态

交接前4个tracked修改及40个untracked文件已逐个备份并记录哈希到：
reviews/issue-46/local03-to-local02-handoff-20261110T085712Z/
其中 local_worktree_inventory.json、preexisting_tracked_changes.patch、
local_worktree_backup/ 可恢复检查点；站姿恢复草稿随交接提交保留，
其余原有文件不被本次提交打包，也没有删除。

部分未跟踪历史助手是本地研究入口依赖。仅clone远端分支不等于复现本机；
Local02应复用同一工作树，不能运行git clean/reset去“整理”。
实验输出保留canonical reviews/issue-46，不写Desktop Robot。
Windows证据入口：
\\wsl.localhost\Ubuntu-22.04\home\lzy18001500226\projects\RobotSim\reviews\issue-46
