# X2 两桌搬运：当前接续点

更新：2026-11-11，22cm完整START物理验收已通过。本页是当前任务记录，旧失败保留在原reviews，不改写历史结果。

## 目标、工作树和边界

- 工作树 `/home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030`
- 分支 `codex/x2-autonomy-task-20261031`；本轮开始HEAD `03c3f15a6dcbc3a3171dce944704f9c2dc46d3c1`。
- Issue #46保持Open；PR #106保持Draft。不得合并main、新建工作树或清理现有文件。
- 完成D70/H220mm/0.200kg圆柱：START→取瓶桌→新鲜RGB-D抓取→>=50mm真实抬升→收臂
  →恢复站姿→退开转身→第二桌→实际落桌→松爪撤手→独立支撑>=2秒，随后交人工审核。
- 沿用用户已审核X2+OmniPicker STL、source关节/effort/速度、碰撞、摩擦、质量、机构和安装。
  没有改手、被动连杆、耦合或厂商资产。无瓶子qpos写入、焊接、隐藏支撑或左手辅助。
- 真实物理为原Docker内厂商MC+MuJoCo3.3.7；host3.3.6用于静态IK和复核渲染。
  本地只读Windows viewer不发物理/运动命令。优先终端和后台渲染，不使用桌面Computer Use。

## 已测结果

证据根 `/home/lzy18001500226/projects/RobotSim/reviews/issue-46/`。

| 运行目录 | 结果 |
| --- | --- |
| raicom-stl-xvfb-native-reset-stand-20261110T110500Z | 30cm完整START→第二桌释放放稳PASS；不重复。视频native-transfer-recorded-review-20261110T111339Z/full_x2_actual_transfer_review.mp4。 |
| raicom-stl-xvfb-native-reset-stand-20261110T235520Z | 22cm固定桌前完整搬运/放置PASS，failure=null；HAL_PHYSICAL_LIFT、POST_LIFT_RETRACT、站姿恢复、LEAVE_TABLE_AND_TURN、SECOND_TABLE_BASE_ARRIVAL_AND_STOP、FINAL_COMPOSITE_STOP、PHYSICAL_TABLE_PLACEMENT、TWO_TABLE_TRANSFER_EXECUTION均PASS。START到取瓶桌NOT_RUN。 |
| raicom-stl-xvfb-native-reset-stand-20261111T004845Z | 22cm完整START走近/停稳、相机抓取、>=50mm抬升、收臂、恢复站姿、持瓶搬运、第二桌松爪撤手/独立支撑全部PASS；failure无，容器退出确认。 |

22cm235520Z：实际trace13332条，已越过旧12000条截断；第二桌独立支撑2.000000s，
最小桌面上向力1.961801N（物体重1.962N），最大手爪接触法向力0；实际开爪q=-0.99894，
撤手80mm。最终COM[.06745386,1.38316052,.65672510]，轴倾角0.01679deg，容器清理确认成功。
末帧后台复核 `native-run-end-capture-20261111T000011Z/overview.png`，不是新物理。

当前没有物理运行，停止新试验，交人工审核。22cm完整START证据004845Z已核对首帧实际base
[-1.5,-1.5,.65192435]。初次抬升最小59.003mm（最大60.837mm），上下向手部支持均值1.962279N，
取瓶桌力0，双pad最小约14N；恢复站姿后的全程COM峰值升高208.951mm。
持瓶底盘净位移2.56845m，到第二桌XY误差107.48mm、朝向误差0.26789rad；不为凑朝向额外转动。
松爪撤手80mm后桌面独立支持2.02秒，桌面最小上向力1.961884N、pad力0，最终瓶身倾角0.03736deg。

最终完整START视频：
`native-transfer-recorded-review-20261111T005652Z/full_x2_actual_transfer_review.mp4`，
SHA256 `43fc26622df7880672599401653eaca73f519ad9d8efba8af41b9830efb43fd3`。
它是原生3.3.7真实物理轨迹在同一完整模型上的复核渲染，不是新物理；1495帧/5fps，已完整解码。
原始trace SHA256 `e18e8081bbeb8a557e4316419ddac292cd2833fb5d423dbc7ac840a4dc96d8eb`。
测量汇总 `h220-full-start-transfer-closeout-20261111T010140Z/measured_result.json`，
SHA256 `acbd3c6cb14971c1f09c95f530c5a48eee9f168b28e9d96b347bc7b8d9beeb85`；Linux及Windows UNC读回一致。

## 复现命令

PowerShell（真实物理+只读实时GUI，004845Z已实际完成的完整START入口）：

    wsl.exe -d Ubuntu-22.04 --cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030 --exec env DISPLAY=:0 PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python -B scripts/research/issue46/grasp/live_retreat_turn_review.py --to-delivery-table --from-start --bottle-height-m .22 --multiccd

不加`--from-start`是235520Z通过的固定桌前范围；004845Z则独立完成了完整START。
WSL终端用相同worktree、env和python参数，不再套wsl.exe。
`--check-only`在物理之前执行现有route codegen/compile，不启动额外MC。
输出由path_governance分配到canonical reviews，禁止写Desktop Robot或C:/home。

## 当前任务配置及实际控制

- H220配置 `simulation/mujoco/scenes/raicom2026/config/reviewed_carry_step_h220_forward_no_roll.json`
  和`reviewed_h220_turn_forward_bend_pickup_functional.json`。
- 先实际抬小臂，再原生下蹲/腰姿，再前送。保留收臂，取消额外预设内旋stow；
  IK仍可做维持抓取方向所需的小幅关节补偿。
- kP6、2Nm爪源cap、既有材料和22块源派生包胶代理、源关节/effort不变。
  必须保留`--carry-grip-force-margin 15.0`，不是15%位置或电流。
- 保留已审核MULTICCD数值配置；不是新几何。冻结host scene SHA
  `404fad116e3432000709e4b0d899603c16a4b68f91548ea6252ee0ec94f6a030`。
- 持瓶先PELVIS_RIGID恢复native站高/MC腰；完整停稳后同一owner将实际TCP重表达为TORSO_RIGID。
  指令采用编译模型真实effort cap（腕2.2Nm等），不提高cap，不夹qpos。
- 直接后退，不做转角预热；中间朝向0.40rad内由后续行走修正。进入250mm任务区域先停，
  然后完整实测停稳及实际放置IK；不单独追末端角度。旧150mm/120mrad诊断保留。
- 90s导航deadline、原非零指令/速度/源限制、15s BRAKING上限不变；零指令停稳最多等12s。
- 第二桌原生降身30mm。放置采用实际FK工具截面向桌面投影，再一次有界桌面XY/手臂IK，
  调整<=150mm，保留桌边半径/倾斜几何余量。实际plan.config传给原owner；不载回旧固定目标。
  初始预选机器人右向50mm只是IK参考，不是每轮强制同一世界落点。
- 允许源范围内小角度倾斜接桌（最大计划12deg），实际力接触后才松爪，最终独立放稳仍实测。
  235520Z实际松手后自行竖稳到0.01679deg，不据倾角几何推理宣称物理PASS。
- 误差单位修正：axis_grasp_error后三维是0.1*(axis-goal)，真实角度为
  `2*asin(norm(e_axis)/.2)`；旧scaled .001约0.573deg。放置相对选定轴误差<=1deg、XYZ<=5mm。
  抓取/抬升/搬运旧数值门不变。旧“1mrad/1deg”部分注释换算有误，旧证据不改写，
  旧H220失败运行不能因此升级。新static plan和实际控制用一致真实单位。

## 已确认的观测截断根因

`native_grasp_audit.c`旧代码在累计12000条时继续真实mj_step但静默停止写trace/latest。
231041Z、232044Z、233127Z均精确12000条；30cm成功110500Z为11500条。
移除观测条数截断，仅保留sink检查；观测寿命由已有bounded运行进程决定。
不延长任何动作预算，不改变物理，不放宽200ms新鲜度。临时late-snapshot等待wrapper已撤掉。

`native-observer-12000-frame-regression-20261110T234818Z`以相同337头文件和C stub做对照：
旧12020次调用仅12000输出，新12020次调用全部输出，实际引擎调用次数相同；这是无物理回归。
235520Z真实13332条及004845Z真实15203条、两次完整放置PASS独立验证修复。新C源码SHA
`c450cff4c4c06f84e048fa7b5788c538ecb4cc20bfba80272965cdf4b9766f01`。
不能把这类截断报告当MuJoCo物理冻结或vendor缺陷。

## 源码、证据和收尾

- 现有脚本仍属隔离研究原型，处置PORT-TO-C++；生产代码不依赖scripts/research。
- 相机是任务蓝色竖圆柱的RGB-D几何估计，不是已验证通用YOLO/6D或真机标定。
- 原4个tracked及40个untracked文件快照/哈希在
  `local03-to-local02-handoff-20261110T085712Z/local_worktree_inventory.json`，未删除。
  同工作树另有Local03/人工infra/docs改动；提交前精确分离，禁止broad add/reset/clean。
- 观察截断修复已提交caf0e225167b02454003943f888db0c4598c07dc。完整任务Git/PR收尾以当前HEAD和PR106结构化closeout为准。物理验收已完成；人工审核PENDING，不merge。
- 部分失败片段 `native-transfer-recorded-review-20261110T232729Z/full_x2_actual_transfer_review.mp4`
  明标PARTIAL/SOURCE FAILED，不能替代最终完整视频。
- Windows证据UNC根：
  `\\wsl.localhost\Ubuntu-22.04\home\lzy18001500226\projects\RobotSim\reviews\issue-46`。

## START前置控制器生命周期

004845Z完整运行采用同一个bootstrap腰/臂owner：先接管并保持当前腰姿（零目标位移），再抬小臂，
再下蹲和弯腰。未接管阶段若MC腰输入年龄>200ms，暂停新发布，消息恢复到<=200ms后继续；
持续缺失>1s或存在未获腰owner的前臂请求仍失败。手臂轨迹不会用旧MC输入推进。
手动owner的目标/增益在新鲜源输入时捕获，后续依赖新鲜native物理反馈；未增改执行器/gain或源限制。
新建只读body observer在原5s准备窗口等待首次有效native状态，不能以ROS端点已发现代替数据新鲜。
所有实际物理、限位、速度、effort、碰撞、停止和200ms状态验收门保持。
