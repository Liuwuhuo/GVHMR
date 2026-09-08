# 视频支撑短窗诊断（非生产默认）

这是 2026-09-08 飞书固定机位案例的受控实验。三个脚本仅在隔离 GVHMR Python 中运行，
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
