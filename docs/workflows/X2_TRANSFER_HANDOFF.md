# X2 两桌搬运：当前接续点

更新：2026-11-11，22cm完整START物理验收已通过。本页是当前任务记录，旧失败保留在原reviews，不改写历史结果。

## 当前Goal：提高验证效率并完成统一11点，再接LiDAR（2026-11-14）

2026-11-15运行恢复：经用户授权修复Docker/Ubuntu集成，Desktop升级4.99并恢复
Ubuntu集成，关闭Ubuntu重复的systemd-timesyncd。Windows QPC独立对照误差约7ppm；
35秒原启动门通过，无时钟步变，未放宽门限。证据
`reviews/issue-46/clock-recovery-docker499-20261114T182204Z/`，长期稳定性仍须运行监测。
随后真实5号启动在首次GetMcAction超时退出（193330Z）；未进入PREGRASP。
修正只读初始化轮询的一次超时直接退出缺陷：4秒总预算重取新鲜RUNNING状态，
非法响应/执行器故障立即失败，Set仍只发一次；外部10秒预算和物理门不变。
实际cf45725运行195105Z通过初始化/Reset/站稳，切换LOCOMOTION_STEP后Get回读
瞬时超时，未发行走指令。共享ensure_mode也仅在原5秒期限内重查只读Get，
不重发Set；持续失败照常停止。21项聚焦回归通过；重新冻结源版本再实测，旧失败不升级。

195848Z再次停在模式查询，未发行走指令。随后隔离ROS图（无MC、无MuJoCo）用
真实Humble/aimdk消息重现：Get服务已返回，但Humble `remove_node()`保留旧的
`Node.executor`引用；普通`rclpy.spin_once()`移除节点后，新增helper错误地把旧引用
当成已注册，导致回调超时。旧`spin_until_future_complete`同图通过。
修正为幂等`add_node`并仅撤销本调用新增的成员关系，原期限、单次Set和物理门不变。
`test_ros_mode_transport`真实生成消息/连续四次模式检查通过；原21项回归通过。
证据`reviews/issue-46/common-mode-readback-case5-20261114T195657Z/`中的
`generated_ros_current_result.json`、`generated_ros_legacy_result.json`及`real_ros_fixed.log`。
这证明本地ROS回调缺陷已修正，不代表MC实际行走或5号完整搬运已通过。

2026-11-15接续：先5号完整放置，再同版0/10，再其余八点；不新增点位专用搬运程序。
截图sim254.901的轨迹复核见
`reviews/issue-46/case5-route-heading-forensic-20261114T162907Z/REPORT.md`：
控制异常后的未刹停物理尾段不等于正常DELIVERY路线。旧0/10通过不能证明新版本5号通过。
本轮修复父运行器的失败收尾：若导航失败且本次停止未验证，立即停止本轮容器，
不先等待轨迹整理/持瓶owner退出；只读窗口以CARRY_ABORT替代误导性的CARRY_HOLD。
保留物理失败，禁止进入放置，未改模型、路线参数、源限制或时钟门限。
边界回归旧代码3项失败，修复后相关150项离线检查通过；尚无新增物理PASS。

本轮35秒启动观测FAIL，证据
`reviews/issue-46/infrastructure-preflight-20261114T171340Z/infrastructure_preflight.json`：
RAW/调整后时钟速率偏差38852ppm、两次墙钟步变约-497/-649ms；机器人/MC尚未启动。
不绕过启动门，不在本任务展开WSL时钟维修。后续先通过现有启动检查，再执行新批次；
修复与执行准备见`reviews/issue-46/common-transfer-case5-fix-20261114T172343Z/`。
5号、端点、其余八点和LiDAR版本轮均NOT RUN；Goal未完成。

用户授权先减少冗余准备、等待和无效全流程回归，再完成同一公共程序的11点完整抓放。
保持已批准模型、材料、源限制和物理安全门；不做持瓶手腕内旋。只改个别瓶位配置时验证
该点；公共运行逻辑集中修复后先验证0/5/10，冻结运行源码和配置后完成全部11点。
文档或纯离线测试变更不自行推翻物理测量；最终验收不能拼接不同运行版本的PASS。
目前历史11点抓取/抬升均有证据，历史完整搬运放置0和10通过；当前公共版本中点尚待复测。
上一轮中点抓取、站起、退开及独立停稳已通过，停稳后的时钟代际交接导致提前结束。
提交b167975修复仅限静止/零输入的短暂反馈重取，240项离线检查通过，动态回归NOT RUN。
上轮总墙钟397.425秒，物理264.841秒，原生主段实测约0.889倍实时；进入原生证据目录前
约71秒准备，新增分段计时定位重复编译，尚未宣称准备加速完成。

开发回归先中点，再代表端点，后其余8点。先完成真值定位版本全部11点；然后控制输入
替换为真实模拟LiDAR估计，再验11点，不能把离线估计检查当成物理导航通过。
沿用现有工作树、分支和reviews；不合并、不清理、不动其他工作树。
PR106保持Draft、Issue46保持Open。D70/H220mm/200g；570g、真机和SLAM均未验收。

## 最新接续：远处观察与斜向接近，11点抓取覆盖（2026-11-13）

本轮Goal已完成11个D70/H220mm/0.200kg瓶位的**逐点累计物理覆盖**：每点均由真实
START[-1.5,-1.5]出发，先在距桌800mm、较桌中线偏机器人右120mm的空间观察；
控制只读取新鲜模拟RGB-D瓶位，按当前实际站姿与源范围IK选择右手抓取区，
用既有MC前进/转向接近，停稳后重新观察，不以精确横移或名义坐标到达替代可抓判定。
左侧目标先粗略对准行进方向；近桌前向修正拆为有碰撞检查的40mm中间路点。
必要时先在宽敞处收小臂，再执行下蹲/转腰/前弯；不新增预设持瓶扭腕。

11点均有双pad实际接触、桌面卸载、>=50mm抬升与>=1秒悬空验收，随后收臂并持稳8秒。
各点持稳窗最小抬升58.856–59.394mm，平均向上支持1.962016–1.962557N，最大桌面反力0N。
**不同点对应完整保留的不同开发源快照；不是最终提交统一版本的11点重测，不是统计成功率。**
11点后退/搬运/第二桌放置均NOT RUN；旧单点搬运PASS不能代替本轮覆盖。
人工视觉审核、570g瓶和真机均NOT RUN。冻结模型、源限制、effort、原接触设置未改。
真实物理引擎为既有原生MuJoCo3.3.7；静态IK/复核渲染为3.3.6，两者不混报。

证据根：`reviews/issue-46/far-observe-diagonal-all11-20261112T172727Z/`。
先看`index.html`（11点滑块）、`all11_actual_grasp_review.mp4`、`REPORT_ZH.md`、
`all11_verified_measurements.json`与`matrix.json`；44条尝试及全部失败保持。
各point目录的`command.json`和`executed_source.tar.gz`绑定实际执行argv、HEAD和完整源哈希。
现有dirty基线依赖和绝对参考路径仍需保留；不是新机器安装包。不要解压覆盖当前工作树。

本机单点新增实测入口（新子目录，不覆盖旧证据）：

```bash
cd /home/lzy18001500226/robotsim-x2-stl-autonomy-integration-20261030
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python -B scripts/research/issue46/grasp/table_edge_grasp_trials.py --output-dir /home/lzy18001500226/projects/RobotSim/reviews/issue-46/far-observe-diagonal-all11-20261112T172727Z --case 5 --far-observe --visible
```

本轮在人工审核前停止：PR106保持Draft、Issue46保持Open，不合并main、不新建worktree、不清理文件。
研究处置：far_observe_approach、native_task_grasp_region和当前runner保留为研究消费者；
先审核实际11点视频，再在明确新任务中提炼稳定动作接口/代表点搬运回归，生产代码不得反向依赖研究脚本。

## 当前Goal：桌边8cm网格与实时相机（2026-11-12，本地时间）

沿用同一工作树、分支；输入HEAD f31b231。用户明确授权新Goal，不再使用上一轮5点/随机点计划。
D70/H220mm/0.200kg、原STL手爪/源限制/材料/求解配置冻结。表长约901.8mm，
中心留55mm桌边余量，世界Y从-1.787376734到-0.995593627，10个80mm常规点加远端，共11点。
位置只用于物理开始前初始化和评估；控制不得读取初始化编号/偏移。

- 默认观察位由桌子网格几何计算：pelvis XY[0.821382333,-1.391485180]、yaw0，
  距近侧长边500mm。37cm候选仅覆盖9/11；50cm离线RGB-D覆盖11/11。
- 全部11点：完整RGB-D XYZ估计→当前站位检查→现有少量备用站位→准备/伸手/靠近/
  闭合/抬升静态见证及33点导航几何采样通过。APPROACH最小约21mm，
  闭合/抬升的wide_loop到瓶体最小约1.48mm。不是连续步态或真实抓放PASS。
  新流程实际换位、抓取、第二桌放置11点全为NOT RUN。
- 新任务只扩展桌边搜索/行程范围（候选<=0.80m、导航根部位移<=0.90m），
  不提高物理速度/effort/关节门限。保留停稳后重新RGB-D判定、最多两次换站位。
- 第二桌此前实际停姿的朝向不变，向世界+Y推进80/140mm的独立离线放置计划通过，
  计划轴倾角约9deg、原12deg上限不变。任务路线关闭过早250mm停车触发，
  复用原实测刹停预测；实际放置IK、接触/卸载/独立支撑仍必须实测。
- 独立第一视角H.264连续视频流已实现，并以旧真实轨迹通过Edge解码（2.51s/25帧）；
  页面明确REPLAY / NO PHYSICS，尚未验证新实况直播。独立只读渲染、不发MC/ROS命令，
  不调用mj_step，最新单帧缓冲，不阻塞控制。任务可见运行会启动相机观察进程。
- 实况仿真环境BLOCKED：Ubuntu普通wsl --exec退出1，Docker日志E_UNEXPECTED/发行版挂载服务缺失。
  Docker Desktop29.8.1正常且无运行容器；现有SSH可进入相同Ubuntu、uid1000与此工作树，
  已实际WSL写入/Windows UNC读回PASS。尚未重启Ubuntu：已有其他SSH/终端可能被中断，
  正等待用户确认安全维护窗口。不得以Windows Docker误传Linux路径或权限绕过代替修复。
- 离线focused tests145 PASS；远端初始化/模型数组不变/START叠加PASS，check-only无Docker也PASS。
  本轮代码以本分支checkpoint为准；其他既有dirty/untracked均保留。物理未运行，不宣称任务完成。

证据/滑块审核页：
reviews/issue-46/table-edge-8cm-grid-20261111T181355Z/index.html；
完整说明REPORT_ZH.md、all11_static_rgbd_summary.json、first_person_stream_replay_current/browser_decode_result.json。
attempt_id 1f9167c0-7cc2-4a57-89ad-0179fc6d413f。

恢复环境后先运行中点5，再近端0/远端10真实完整START代表点；相同版本验证后再其余8点。
已有table_edge_grid_trials.py prepare/execute入口：冻结源代码与实际资产哈希，按5/0/10/其余点执行，
缺测量不补零、首失败停止、旧固定站位PASS不能替代新中线观察流程。提交后须重新prepare，旧manifest不会静默混用HEAD。
精确命令在本轮REPORT_ZH.md。PR106 Draft、Issue46 Open，不merge、不新建工作树、不清理文件。
生产化处置PORT-TO-C++，离线网格/观察工具KEEP-AS-TOOL；当前不开展雷达/SLAM替换。

### 恢复环境后的首个实测（2026-11-12）

用户已确认其他WSL工作保存退出并授权只重启Ubuntu-22.04；重启后exec、Docker socket/daemon、
WSLg、源码和76个资产哈希均恢复/匹配。首个点5实际运行
raicom-stl-xvfb-native-reset-stand-20261112T012251Z，START到中线观察位及完整停稳PASS，
实际根部XY[0.840159,-1.374849]，随后首帧RGB-D年龄240.135ms超过不变200ms门限而停止；
换位/闭爪/抬升NOT RUN，容器已正常清理。

保存状态离线计时复核：最初两帧493.7/230.0ms，其后中位约69.6ms。
修正仅丢弃过期图像、最多3秒取得新源帧，不改写时间戳，不放宽200ms、不重试原生/物理故障。
新增5项回归与原有检查共159 PASS。旧失败/旧batch保持；新版本重新prepare再执行点5/0/10。
本轮恢复的closeout attempt_id为1bc024dc-25ce-4e52-912c-38ae0c27c521，旧BLOCKED事件不覆盖。

### 新微调入口的时刻顺序修复

第二轮raicom-stl-xvfb-native-reset-stand-20261112T013657Z：相机两帧源年龄56/70ms通过，
新鲜RGB-D选择NEARBY_0；随后微调调用把读取状态前的tick传给PointGoalController，
新读到的样本可晚于tick而被判stale_state。纯离线测试用真实PointGoalController复现旧调用失败；
改为读取/校验状态后再取求值时刻，保留原样本时间戳，新增实际pose_age诊断。160项检查PASS。
手、模型、接触、关节/effort/物理速度和200ms新鲜度未改；尚未闭爪/抬升。
相机固定8793端口另有实测EADDRINUSE，现由系统分配独立loopback端口并在绑定成功后写URL，
不改相机标定。外部Edge自动启动被策略拒绝；提供Codex浏览器面板/手动URL，未绕过拒绝。
旧两轮原始轨迹和失败均保留，新版本重新prepare实际试验。

### 近桌纯转向漂移与任务平移方案

第三轮raicom-stl-xvfb-native-reset-stand-20261112T015605Z已通过修正后的时间语义，
但近桌ALIGN纯转向指令(linear=0,angular=0.4)使根部从约x0.815前漂至x1.068；
右narrow3在[1.32254,-1.41650,0.54487]碰桌沿，约-0.828mm/8.67N，物理门正确停止。
没有闭爪/抬升。原定点控制在末端还试图回正朝向，不能把固定锚点原地转向采样当作实际步态。

改用现有MC的body-X/Y平移通道，取消近桌pivot输入；沿用点目标的时效/帧/超时检查、
0.6高层位置比例、0.20m/s合成平移上限，角速度指令0。官方/社区桥已有lateral_velocity，
仅task_grasp_region启用max_lateral_velocity0.20；底层MC、模型、源关节/effort/物理速度不变。
粗停使用已有任务6cm位置容差；完整物理停稳后必须重新RGB-D/IK判断可抓，不要求名义最终朝向。
检查路径是实际观察姿态上的恒朝向平移，255个包络样本(±5cm位置、±0.20rad朝向)；
11点静态均通过，最小约27.3mm。运行中离开已检查包络先安全停稳并重观测，
真实碰撞/源限位等仍立即终止。旁向实际能力尚待本轮物理验证，静态不计抓放PASS。

### 首次新观察/调整/抬升实测与持瓶余量修正

raicom-stl-xvfb-native-reset-stand-20261112T025942Z：真实START、中线观察、两次安全暂停后重观测、
当前站位IK、实际双pad抓取、最小59.0378mm抬升、手部平均向上力1.962139N/桌力0、收臂和站起均PASS。
两个调整都因离开已检查平移管道主动停稳重观测，没有把名义点位误差当抓取失败。
开始后退后right_shoulder_roll实测0.064513rad超过源上限0.061；原限位门停止，搬运/放置未通过。

站起后目标roll约0.04455，余量约0.0167rad，实测动态偏差约0.020rad。
使用原7DoF IK在保存的根部/腰轨迹上作4个有界侧移反事实，4cm是最小检查候选，
预测roll余量约0.0885rad、全部arm余量同量级。原收臂13样本源范围/碰撞检查通过。
反事实几何仅用于离线预检，保持原实测物体-手相对姿态；初次混用理想手位姿与实测瓶位姿的
碰撞检查被单独标记并更正，没有在实际仿真加入刚性约束或屏蔽碰撞。

task_grasp_region的既有收臂平移由world[-.16,0,0]改为[-.16,-.04,0]，
即当前桌坐标负Y方向4cm；源模型、限位、增益、effort、摩擦和>=50mm门均不变。
没有新增预设扭腕。实际动态资格待新版本完整START试验。
163项离线检查通过；实际状态视频point05_actual_grasp_lift_partial_20261112T025942Z.mp4为1218帧/
10fps/121.8s、2x物理时间，已完整解码，标题保留final task FAIL。

### 原生观测同步写盘阻塞（修复中）

右移4cm后的raicom-stl-xvfb-native-reset-stand-20261112T035355Z再次通过抓取/抬升/收臂/站起；
后退约0.114m后进入TURN，源状态出现387.43ms墙钟间隔但仅推进20ms仿真时间。
记录自带计时显示previous_observer_io=367.13ms、max_physics_step=1.085ms、
max_between_step_calls=0.644ms；异常在观测同步I/O，不据此归咎MC或求解器。

依据runtime.md快控制与磁盘I/O隔离规则，原始JSONL改由64条/16MiB有界FIFO写盘；
物理线程只发布原时间戳的atomic latest，原200ms控制新鲜度不变。
不修改mjModel/mjData、步长、碰撞、力或控制输入；real mj_step仍调用一次。
正常退出排空；队列/写盘失败停止新状态发布，由既有新鲜度安全门终止；
成功运行要求原始记录不缺失，证据评估增加独立排空屏障，不能当作新鲜反馈。

精确337 ABI的counter stub注入350ms慢写盘：旧latest采样间隔370.38ms，新20.85ms，
两者120次基础调用/120条记录，无物理动力学调用。溢出测试明确报告失败/拒绝记录，
不允许把数据丢失包装成PASS。166项Python检查通过。真实动态资格待新版本实测。
已关闭本线程已结束且身份核验匹配的只读窗口；不处理其他任务进程。

## 后续空手运动标定：当前停在审核（2026-11-11）

新增有界标定没有取得最小精度资格：前后与左右转向各一条停稳响应；左移
在运动中因-0.790737777s系统/单调时钟映射跳变锁定，右移和重复项NOT RUN。
这不是夹爪/摩擦结论；新标定器的平滑/支撑相位停令协议也使小目标判定滞后。
不得把其29.5cm/35–48deg响应当作硬件最小精度或生产运动接口。此前H220搬运PASS保留；本轮不改变该结论。

当前没有MC/物理运行，不再为补齐表格继续运动或放宽时钟/停止检查。
审核入口：`reviews/issue-46/native-motion-capability-review-20261111T123841Z/index.html`，
其中REPORT_ZH.md、stop_decomposition.json、原生物理复核和实际状态视频已保存。
实际来源是120712Z、122716Z两个独立native会话；诊断代码RETAIN_RESEARCH_DIAGNOSTIC。
下一步先审测量/停令语义与时间边界，再单独授权补充试验；不恢复站位微调/抓取调参。

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
