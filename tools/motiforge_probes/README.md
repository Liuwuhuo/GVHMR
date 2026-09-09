# 视频源与支撑短窗诊断（非生产默认）

本目录仅在隔离 GVHMR Python 中运行，不是产品自动后处理。以下第一组是
2026-09-08 飞书固定机位案例的受控实验；该组三个脚本
没有被 backend、安装入口或 MotiForge core 导入，不改变普通视频推理。受试视频 SHA-256
为 `80fa6378001d92df61459105ef59cb2043d5c4c141ce71a383d7bd0cf0827f1f`，2834 帧、30 Hz。
脚本拒绝其他视频身份：人工帧标注不能直接迁移到其他输入。

## 输入与复现

先自行设置 `INPUT`（带脚底几何的完整 portable NPZ）、`KEYPOINTS`（同次推理的
`native/motiforge/preprocess/vitpose.pt`）、`MODEL`（获许可的 SMPL-X neutral NPZ）和
`OUTPUT`（不存在候选文件的实验目录）。在本仓库根目录运行，路径可来自任意受控存储。
模型、原视频、关键点及输出均不随代码分发，不隐式下载。

```bash
export PYTHONNOUSERSITE=1
python tools/motiforge_probes/prepare_geometry.py \
  --input "$INPUT" --keypoints "$KEYPOINTS" --model "$MODEL" \
  --output "$OUTPUT/geometry.npz"
python tools/motiforge_probes/fit_video_window.py \
  --input "$INPUT" --geometry "$OUTPUT/geometry.npz" --model "$MODEL" \
  --output "$OUTPUT/local.npz" --pose --freeze-toe \
  --nonpenetration-weight 2 --plane first-frame --local-support-only --iterations 160
python tools/motiforge_probes/add_verified_contacts.py \
  "$OUTPUT/local.npz" "$OUTPUT/local-annotated.npz"
```

随后切回主 MotiForge 环境，使用未修改的正式链路：

```bash
motiforge retarget "$OUTPUT/local-annotated.npz" -r adam_lite \
  --engine motiforge-mink --fps 30 --ground --foot-target-mode surface-static \
  -o "$OUTPUT/robot"
```

## 实际优化及边界

- 在原始视频的 `[450,660)` 范围对照；人工可见前掌支撑为右足 `[561,579)`、左足
  `[594,601)`（零基、半开区间），不是整脚平放或三维接触真值。真实跳跃 f498/499/553/554
  受保护，未知帧不标为腾空。COCO17 不提供脚趾观测。
- `prepare_geometry` 由首帧 global/incam 根关系构造固定 R/T、原 K 与实际 COCO12 观测，
  用完整 SMPL-X FK 复核 incam。此 R/T 是诊断估计，不是实测相机标定。
- `fit_video_window` 用真实二维重投影、根位移、腿局部旋转、二阶增量正则、每足一个
  连贯前掌材料 patch，以及可选整个足部 mesh 非穿地软项。冻结 shape、根朝向、上肢；
  `--freeze-toe` 不允许利用 body22 位置无法表达的脚趾局部旋转满足约束。
- `--plane first-frame` 与既有首帧地面 gauge 一致，仅适用于本例首帧确有足部落地。
  `--local-support-only` 只允许两支持段及各侧 5 帧余弦过渡修正；其他源帧逐值恢复。
  单点支撑没有固定足部朝向。序列化世界 pose 后重新进行完整 FK 与脚底表面计算，
  校验关节/精简皮肤计算与完整模型误差 <0.1 mm，不保留陈旧几何。
- incam、原静止概率、原 floor_correction_y 保留为原推理证据，不冒充精修后的相机解。
  增量、参数、限制、原输入身份写进 `experimental_video_window_fit`。
  `add_verified_contacts` **单独**把两个可见支撑窗的证据强度写为 .99，保存原概率；
  这不是自动接触估计器，也不是校准概率。现有 retarget 运动学护栏保持启用。
- 输出拒绝覆盖。所有候选仍需公共 Quality/Fidelity 和同一原目标参考评价，不应凭
  修改后的目标自洽评分宣称恢复真值。最终局部候选的最大片段人体修正约 97 mm/14°，
  不属于无损滤波。

## 结果与失败消融

Adam Lite 的两支持窗足底点最大 XY 漂移从约 70.4/90.2 mm 降到 7.0/4.9 mm。
局部窗口浮脚仍 4/210，全片 84→85/2834；唯一新增帧仅从 29.948 变到 30.100 mm，
跨过公共 30 mm 阈值。固定原目标 tracking p95 62.7→92.2 mm，物理等级仍 C。
因此只是值得人工回放的局部研究候选，**不是自动修复或生产验收**。

不能省略的反例：只加人工接触标注、原几何不变，机器人输出精确不变（速度护栏拒绝）；
只做二维重投影会让三维支撑漂移更大；允许整窗变化则出现新的抬高，float 84→99。
`--contact-weight 0`、不传 `--pose`、不传 `--freeze-toe`、
`--nonpenetration-weight 0`、`--plane window`、不传 `--local-support-only`
保留这些受控消融，不能当作推荐参数排列组合。源帧不变不保证机器人帧严格不变：
全片足部标定和时序求解仍可能传播小变化。

下一步应自动获得可靠的前掌/滚足/未知支撑证据，增加窗口边界与非接触区域的保持约束，
并回归不同视频；不能通过降低公共质量门槛或将所有脚强压地来验收。

## 2026-09-09：高度消融、背部支撑与相机一致性

下面三个独立脚本不绑定飞书案例，也不被生产流程导入。输入仍需是本 backend 的静态、
未镜像 portable artifact；所有输出拒绝覆盖原件，模型/视频/关键点不随源码分发。

### 拆开已有高度修正

```bash
python tools/motiforge_probes/ablate_source.py \
  --source "case-name=$PLAIN_PORTABLE" \
  --asset-root "$ASSET_ROOT" --output-root "$NEW_EXPERIMENT_DIR"
```

从原 plain position-only portable 的 world Y + floor_correction_y 重建 native，再分别运行
camera-only、floor-only，输出 plain/inputs 两层及报告。native 仍包括原生 GVHMR 后处理，
不是 raw network 输出。先重跑 combined 校正并要求误差 <1e-5 m；只允许 Y 平移，pose、
incam、confidence、水平坐标保持。脚底必须重新经过完整 SMPL-X 导出，不平移旧证据冒充重算。
五条站立/走停/跳跃/快速换步/躺起，共 20 组同参数对照，最大重建误差 <1.20e-7 m。
统一关闭修正导致走停浮起和躺起穿地/跳变，不作为修复方案。

### 人工双背部支撑（两个失败候选，仅供复现）

```bash
python tools/motiforge_probes/fit_back_support.py \
  --input "$SURFACE_PORTABLE" --keypoints "$KEYPOINTS" --model "$MODEL" \
  --window-start 12 --window-end 17 --support-start 14 --support-end 15.5 \
  --iterations 120 --surface-penalty mean --device cuda --output "$NEW_MEAN_NPZ"
```

以上时间只对应男性躺起开发片段，不能原样当其他视频标签。另一个候选**只**将
`--surface-penalty` 改为 `worst-frame`、使用另一输出文件；其余参数和权重不变。
女性留出原视频先独立确认支撑 [6.5,7.5) s，优化 [4.5,9.5) s，没有按模型结果修改标签。

- 人工确认原首帧站立、支撑窗骨盆后侧与胸背确实接地。plane 取原首帧脚面，R/T 取原首帧
  global/incam 根关系，K 和 shape 固定；它们是估计 gauge，不是实测相机/地面标定。
- shaped-neutral 空间按 skinning 权重及后方选两个静态材料 patch，先验证 +Y 躯干、+Z
  toe-forward、patch 后向/不重叠。只优化根和髋/膝/脊柱，不改变 toe/shape/原输入。
- 根平移范数 ≤0.65 m，根旋转每轴 ≤45°、局部每轴 ≤20°，**并非总旋转角 45°/20°**。
  目标为真实 COCO12、背部贴地、抽样全身+完整手脚非穿地、先验和平滑。ViTPose heatmap
  峰值不保证 ≤1；只把优化权重裁到 [0,1] 平方，观测本身不改。
- 0.5 s sin² 窗口过渡，窗口外和边界帧逐值恢复。序列化后全时间线 SMPL-X FK/脚面检查，
  独立完整 mesh 穿透评估；拒绝已有 body22 姿态证据，避免保留陈旧旋转。
- 原 incam/confidence/floor_correction_y 是旧推理证据；新几何证据写入
  `experimental_back_support_fit`。旧 foot_surface_export 是祖先记录，不是本次计算身份。
- 男性开发片段支撑重投影 p95 84.80→135.84/121.70 px，全 mesh 最大穿地
  70.19→259.00/147.36 mm。均方会稀释局部深穿透，逐帧最深惩罚减轻它却仍不能验收。
  两版都不接入自动流程；贴地或某个中位数下降不能证明动作恢复正确。

### 只读 world/incam 审计

```bash
python tools/motiforge_probes/camera_consistency.py \
  --input "$SURFACE_PORTABLE" --keypoints "$KEYPOINTS" --model "$MODEL" \
  --start 14 --end 15.5 --output "$NEW_AUDIT_JSON"
```

始终 CPU、不拟合任何变量，完整 FK 验证后，在同一实际 2D mask 比较原 incam 直接投影与
现 world 经首帧固定外参的投影，记录每帧隐含旋转偏差。男性支撑窗两者 p95 13.91/84.80 px，
提示固定相机下世界恢复与相机观测仍不一致。旋转偏差不是实测 camera motion；2D detector
不是 mocap 真值，首帧一致也不证明整段 gravity/extrinsic 正确。

新一轮仍先定位 incam/world 和重力/地面参考，再做下一轮有限候选，不继续增加本轮接触权重。
完整输入身份、固定参数、机器人统一门禁与留出结果维护在 MotiForge 的
`docs/video_source_comparison.md`。本目录不含训练新模型或机器人动力学后处理。
