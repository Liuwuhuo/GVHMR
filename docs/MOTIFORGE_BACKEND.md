# MotiForge headless backend

`hmr4d.backends.motiforge` is the stable process boundary used by the sibling
MotiForge repository. GVHMR owns video normalization, tracking, image features,
visual odometry, model inference and portable SMPL-X/body22 export. MotiForge
owns orchestration, content-addressed caching, retargeting, quality gates and
dataset sinks.

The backend does not import MotiForge and never downloads weights. MotiForge
invokes it in this repository's Python environment using a versioned JSON
request/response protocol.

## Runtime check

### Explicit no-flight source projection (opt-in, 2026-09-16)

Request `options.assume_grounded` is a boolean, default false. When true, source
ground stabilization must be enabled. Both capabilities and doctor advertise
`assume_grounded: true`; clients must negotiate support, not silently drop it.
Fresh inference carries the flag through `_portable_prediction` and
`smplx_sequence.enrich_prediction` into `surface_ground.add_surface_ground`.

```bash
python -m hmr4d.backends.motiforge export-ground INPUT.npz \
  --output out/assume-grounded.npz --asset-root . --assume-grounded
```

The operator declares at least one foot touching the same zero-height flat floor
EVERY frame. Let `s` be current left/right SMPL-X foot surfaces and `c0` the
previous downward correction: use total `c=c0+min(s)`, translate world joints and
global translation by `c0-c`, and rebuild/validate the portable sequence. Pose,
shape, root orientation, horizontal motion and incam remain unchanged. Static
probabilities neither choose supports nor authorize this operation. Original
parameters/correction/surfaces and input provenance are retained.

This removes real flight; do NOT apply to jumping, airborne running, stairs or
object support. Exact per-frame contact cannot also guarantee a fixed correction
speed cap. v4 reports `removes_real_flight=true`, the actual correction speed and
`grounded_projection_fast_height_change` above 0.2 m/s. Root jitter remains a
review concern; this is not a pose/dynamics/sliding/collision solver. Original,
v2 and v3 caches can be re-exported after undoing their TOTAL correction, never
overwriting input. Reject reference-window combinations and repeated v4 exports.
Default/disabled behavior and real jumps remain unchanged when the option is off.

Six real-source regression inputs (2114 frames) preserve pose/XY/incam and produce
zero lower-foot height with FK error <2 micrometers. Three matched Adam SP/UMR
50 Hz pairs reduced >3 cm dual-foot elevation from 90.77/8.94/17.88% to zero;
quality C/C/C became B/C/C, not universal acceptance. One clip's slip increased
~20%, another retains ~43.5 mm self-penetration. The reported cz4Z clip's source
vertical acceleration p95 increased 3.42 to 8.90 m/s²; no smoothing improvement
is claimed. Full backend suite: 170 tests. Main integration: 1680 passed, 2 skipped.

### Flat-ground surface correction (2026-09-16)

After observation candidate selection, `smplx_sequence.enrich_prediction` calls
`surface_ground.add_surface_ground` before creating the portable sequence. CPU
SMPL-X FK supplies the same weight-selected feet as `export-foot-surface`, and
checks body22 against the portable joints (0.1 mm tolerance). The checkpoint's
static probability is trained from joint speed, not physical ground contact.
Both custom static-support anchor stages are now disabled. The earlier
joint-height percentile is NOT the final ground plane.

If `c0` was already subtracted and `s` is the current surface, the desired total
correction ceiling is `min(c0, min(s) + c0)`, combining no-downward-shift with
world Y=0 nonpenetration. A two-sided Lipschitz minorant with slope 0.2 m/s
enforces nonpenetration and limits the TOTAL correction, not two separately
limited stages. Joints and global translation receive the same `c0-c` shift;
pose, shape, XY and incam remain unchanged. Sequence FK is regenerated. This
`foot-surface-ground-v2` stage only raises the person and never removes a positive
height offset. Rate limiting can also lift adjacent frames before a penetration;
nonpenetration does not prove correct ground contact or overall height accuracy.

`foot_surface_y` and `surface_ground` diagnostics are now automatic. When enabled,
the source declares an error if a surface is below -2 mm. Elevated feet (>3 cm) are advisory
`review`, never claimed to be valid flight. `ground_input_transl`,
`ground_input_floor_correction_y`, `ground_input_foot_surface_y` preserve the input;
`ground_stabilization` diagnostics describe the camera-only preceding stage.
Disabled source ground leaves every pre-existing numeric field unchanged. This
is a flat-ground assumption, not stair/object support or a physical motion fit.

MotiForge reads the numeric report into its generic source validation contract.
Inspection stays available, but execution rejects declared source errors before
any engine, except the withdrawn v1 static-support-residual rejection, now advisory.
Old caches are not implicitly modified or certified. Reprocess into
a NEW path using the same implementation:

```bash
python -m hmr4d.backends.motiforge export-ground INPUT.npz \
  --output out/ground-checked.npz --asset-root .
```

The command requires an enabled-ground original cache and rejects double correction.
It first undoes that cache's `floor_correction_y`, recomputes camera-only height,
then applies surface nonpenetration; it does NOT keep the old contact-floor shift.
An applied legacy correction without its numeric array is rejected.
Derived body22/SMPL-X geometry is regenerated, not left stale. Backend identity
includes both surface helpers, so new inference cannot hit the old ground cache.

Two reported C3-P4yh5yos clips were re-exported from their original caches. Both
now have nonnegative foot surfaces, but the lower-foot-above-3-cm fractions are
109/356 and 177/198: disabling anchors is NOT an overall height improvement.
The earlier v1 rejected the second clip at static-support residual p95 8.67 cm;
that rejection was based on incorrect static/contact semantics, not established
invalid human posture. Tests cover stance, swing, real flight, probability
independence, fast negative excursions, disabled parity and stale geometry.

### Explicit grounded reference window (opt-in, 2026-09-16)

```bash
python -m hmr4d.backends.motiforge export-ground INPUT.npz \
  --output out/human-calibrated.npz --asset-root . \
  --reference-start 0 --reference-duration 0.5 --assume-reference-grounded
```

The operator confirms at least one foot is on the same flat floor during this
interval; this is not inferred from the checkpoint's static probabilities. It
must be within the original timeline and contain at least 0.1 s and 3 samples.
The median lower-foot surface provides ONE constant offset `b`. The final total
correction is the 0.2 m/s Lipschitz minorant of `min(c0+b, min(s)+c0)`. Thus the
constant calibration can lower an initially floating body, but subsequent
nonpenetration only raises it relative to this fixed reference. Real jumps are
not explicitly pulled down. Pose, shape, XY and incam remain unchanged; FK and
the sequence are rebuilt as in the uncalibrated export.

The `foot-surface-ground-v3` report records the reference interval, offset,
reference height spread and maximum additional nonpenetration shift. Spread
over 5 cm and a shift over 10 cm are advisory review warnings, not certified
support. A large shift must not be described as a small refinement. The first
interval can be airborne or occluded; choose another confirmed interval instead.
This does not calibrate scale/gravity or guarantee later root-height accuracy.

For explicit calibration, a fresh `foot-surface-ground-v2` inference artifact is
also accepted when its numeric total `floor_correction_y` is present. Export
undoes that total before recomputing camera/reference/geometry; it does not stack
a second correction. v3 recalibration remains rejected. Original and v2 inputs
are covered by an equivalence test.
Default video inference and exports without these flags stay unchanged (v2).
The calibrated NPZ is accepted by the regular MotiForge reader/retarget pipeline.

### Backend availability

`python -m hmr4d.backends.motiforge capabilities` provides lightweight protocol,
mode and implementation identity negotiation without importing Torch or loading
checkpoints. Default source identity covers the adapter and its observation
stability, local-arm and short-gap helpers; a helper change must not reuse an older prediction cache.

Prepare the upstream checkpoints described in `docs/INSTALL.md`, then run:

```bash
python -m hmr4d.backends.motiforge doctor --asset-root /path/to/gvhmr-assets
```

The asset root may differ from this code checkout. It must contain:

```text
inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
inputs/checkpoints/yolo/yolov8x.pt
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

DPVO additionally requires its checkpoint and initialized submodule. The
default SimpleVO path does not require DPVO.

## MotiForge usage

```bash
export MOTIFORGE_GVHMR_ROOT=/absolute/path/to/GVHMR
export MOTIFORGE_GVHMR_PYTHON=/absolute/path/to/gvhmr/bin/python
export MOTIFORGE_GVHMR_ASSET_ROOT=/path/to/gvhmr-assets

motiforge video clip.mp4 -r unitree_g1 \
  --engine motiforge-mink --frames 450 --ground \
  --sink npz -o output/video-test
```

MotiForge calls the backend as:

```bash
python -m hmr4d.backends.motiforge run request.json response.json
```

The portable prediction contains world body joints, global/in-camera SMPL-X
parameters, camera intrinsics, left/right static-foot probability (legacy
`contact_confidence` fields, not physical contact confidence), the applied
world-Y floor correction, source SHA-256, inference options, the GVHMR Git
revision and backend revision. Writes use temporary files followed by atomic
replacement; one bad video does not abort the rest of a batch.

The upstream static-camera postprocessor includes an XYZ camera-root correction
with a 0.25 m discrepancy dead zone, followed by a horizontal-only static-joint
correction. It does not leave Y entirely untouched, but residual height drift
inside the camera dead zone can remain.

The legacy `contact-floor-v1` helper is retained only for historical experiments;
production inference no longer calls it. Neither high nor low ankle/foot static
probability establishes support or flight. Native GVHMR postprocessing is unchanged.

Only when both `static_camera` and `ground_stabilization` are enabled, the backend
uses `camera-height-no-support-v3`: reconstruct the raw in-camera pelvis,
map it with the original frame-zero camera-to-world rotation, and Gaussian-filter
only the world-minus-camera Y discrepancy with sigma=0.5 seconds (nearest edge
padding, truncate=4). The camera translation is not independently smoothed, so
real vertical motion shared by both representations cancels before filtering.
The first camera correction sample is subtracted to retain the original initial
height gauge. This correction is projected through a 0.2 m/s Lipschitz minorant;
no contact-floor stage follows. The surface-only stage above also constrains the
TOTAL correction. There is no separate 0.25 m amplitude cap. The exported total
`floor_correction_y` is
subtracted once from the original world-joint Y and global SMPL translation Y;
XZ, root-relative pose, in-camera parameters and source confidence are unchanged
apart from float32 rounding. This does not hard-snap each frame to the floor.

Dynamic cameras and disabled stabilization make no pre-surface height change.
Missing, malformed or nonfinite in-camera parameters also leave this stage
unchanged with an explicit `camera_stage.reason` and `detail`, never falling back
to contact anchors. `ground_stabilization.support_anchor_enabled` is false.
Final surface diagnostics describe actual nonpenetration and unknown elevations;
there is no static-support-height rejection.

MotiForge enables source ground stabilization by default. Use
`--gvhmr-no-ground-stabilization` on `motiforge video` to reproduce the raw
upstream world-Y behavior. Robot sole clearance and collision grounding remain
downstream retarget concerns, so Mink dataset generation should still use
`--ground`.

Historical `static_support_missing_frames`/`flight_frames` fields from the old
helper are NOT flight classifications and are not used by the current height
path. The legacy `contact-floor-v1` helper and protocol-3 format remain readable. The
backend file's content revision invalidates caches when this source algorithm
changes; no retarget-engine options enter this source artifact identity.

## Observation stability (2026-09-11)

The optional request field `options.observation_stability` defaults to `audit`:

- `off`: skip the new observation/final-human audit and repair.
- `audit`: preserve the exact model input and numerical export, append advisory
  visibility, boundary/truncation and temporal-spike evidence only.
- `conservative`: additionally bridge short, asymmetric opposite-side elbow/wrist
  collapses with reliable endpoints, using the previously validated fixed rule.
  Scores are never increased; only effective visible-coordinate changes replace
  model input. The original native bbox/ViT cache is retained unchanged.

The NumPy-only `backends/observation_stability.py` owns the numerical policy.
The adapter calls `_predict_observation_candidate`: it prepares observation
evidence and always predicts the original input. Only an effective conservative
proposal causes a second prediction using the same model and portable/ground
path. `_select_observation_candidate` compares both complete human motions using
`evaluate_repair_candidate`; accepted whole-model repairs remain unchanged.
For a rejected proposal, `_localize_observation_candidate` tries an arm-local
SMPL rotation hypothesis. `_interpolate_observation_gaps` also reconstructs only
short gaps with reliable adjacent neighborhoods. Both use the same guard; a gap
must additionally pass against an already accepted local correction to replace
it. Otherwise the preceding selection is retained. No independent world-joint
XYZ tracks are spliced.
`_finish_observation_stability` attaches
the selected-output audit after this choice. Audit/off/no-hit paths predict once.
No robot, retarget, contact/height, network-weight or global smoothing change is
part of this stage. `audit` is not a physical quality gate. A large velocity alone
does not establish an invalid action, and an interpolated coordinate is a
hypothesis, not a recovered observation. Changed inputs may affect the whole clip.

`observation_stability.json` next to the human cache and
`motiforge_video.observation_stability` store the same schema-version-1 report,
including inclusive source-frame intervals at 30 Hz, unchanged native score
semantics, accepted repair coordinates and diagnostic thresholds. Original detector
presence/identity are unknown in existing smoothed-bbox caches, not fabricated.
No automatic clipping, boundary filling, motion freezing or silent rejection is
performed. All frames remain available for inspection and downstream evaluation.

The MotiForge CLI option is `--gvhmr-observation-stability`; Web exposes the same
three modes. Mode enters the source cache key. Old protocol-3 portable artifacts
remain readable, but do not acquire audit evidence retroactively. Multi-person
output is not added: the 2D tracker selects the largest accumulated-area track and
only that person enters the SMPL-X model. Independent per-track reconstruction and
shared-world identity/interaction consistency require separate work.

Regression evidence is maintained in sibling MotiForge's
`docs/regression_baseline.md`, with full35 cached-video and fixed Mink comparisons
under its ignored `out/gvhmr-stability-20260911/` directory.

The 2026-09-11 full35 test preserves audit numerics and confirms 4 clips / 10
effective visible keypoint-frame edits. The paired full-sequence fresh off/audit
control is exactly equal. Conservative repair reduces the two known dink08
shoulder spikes, but NJd37 is a genuine local counterexample: near the repaired
wrist, robot elbow peak acceleration in 13.82–14.26 s rises from 98.75 to
136.41 rad/s² despite a slightly lower whole-clip jerk p95. This is not unrelated
tail noise. Consequently only audit is accepted as the default; conservative
remains experimental opt-in, not a generally regression-free correction.
These figures describe the first unguarded implementation, before the following
acceptance guard; no threshold was tuned to exclude a video by identity.

### Candidate temporal guard v1

The source-only guard compares 49 channels separately: world root, 21 root-relative
joints, 21 unit bone directions, four shoulder-relative elbow/wrist trajectories
and two elbow angles. It measures finite-difference speed/acceleration/jerk peaks
in every effective repair interval padded by 0.25 seconds, the full clip and the
outside-window region. Derivative samples belong to a region by their stencil
midpoint on the original timeline. No channel peak may exceed its reference by
more than 10% plus an absolute floor: position [0.03, 1, 30] in m/s^n; unit bone
direction [0.1, 3, 90] in 1/s^n; elbow angle [0.1, 3, 90] in rad/s^n.
Each repaired limb/window also needs an acceleration decrease of at least the
larger of 10% and the corresponding floor in one of its channels. Invalid or
degenerate candidate geometry is rejected; invalid reference/contracts fail
explicitly. This checks changes in temporal peaks, not pose accuracy, all local
oscillations, amplitude preservation or downstream robot feasibility.

The report records `candidate_acceptance`, per-channel measurements and reasons.
Rejected candidates have `keypoint_repair.applied=false`, zero applied frames and
`keypoint_repair_rolled_back`; attempted coordinates remain available as evidence.
Both complete pre-selection portable outputs are saved as
`observation_original.npz` and `observation_candidate.npz` when dual inference ran.
Prediction failures remain explicit per-video failures, not silent acceptance.
No new passes are added to detector/ViTPose/features, and the model stays loaded.

Four paired fresh inferences on the same35 cached videos reproduced the original
numerics exactly. The fixed guard accepts NJd07 and rolls back NJd31, NJd37 and
dink08 (including smaller other-limb regressions despite its shoulder improvement).
The other31 preserve cached source numerics with explicit reuse provenance.
Full downstream results are recorded under sibling MotiForge's ignored
`out/gvhmr-stability-guarded-20260911/` and `docs/regression_baseline.md`.
Default audit is unchanged; rejecting a repair does not remove original spikes.

### Local arm fallback

`backends/local_arm_repair.py::localize_arm_pose` addresses whole-body side effects
from a short 2D arm edit. Only the affected side's SMPL22 shoulder/elbow/wrist
(left 16/18/20, right 17/19/21; body-pose indices minus one) can change, using
`R_raw Exp(g*w*Log(R_raw^-1 R_candidate))`. Gains are fixed at shoulder 1 and
elbow/wrist 0.5. The error interval has unit window weight; a 0.25-second quintic
C2 taper returns it to zero on both sides. Touching/overlapping hit intervals
are merged and other tapers use a smooth union, avoiding max-weight cusps.

The original shape, global/camera roots/translations, all other local poses,
contacts and floor correction remain unchanged. Both coordinate systems share
the same localized body pose and the already-loaded native `fk_v2` recomputes
the complete world skeleton. Position differences outside the arm may include
float32 roundoff from recomputing an already ground-adjusted translation; local
pose/parameter invariants are exact. This is not arbitrary XYZ stitching or a
third model prediction, and does not import any robot implementation.

The existing guard thresholds are unchanged. `model_candidate_acceptance` keeps
the whole-model decision; `local_arm_repair` records the bounded hypothesis;
the final `candidate_acceptance.selected` is `candidate`, `localized_candidate`,
`gap_candidate` or `original`. `observation_local_candidate.npz` is saved separately even if
rejected. The final applied flag/count and warning codes reflect the selected
output, not the initial unsuccessful attempt. Default audit/off and no-hit
numerics stay unchanged. Backend cache identity includes the new helper.

The frozen dink08 test reduces two shoulder acceleration peaks by about 43%
and 52% without reducing Mink's speed limit; NJd07 keeps its previously accepted
whole-model repair, while NJd31/37 still retain their original outputs. This
does not remove all motion uncertainty or the out-of-frame ankle event at the
clip tail. Complete35 downstream evidence and current commit identity are
recorded in sibling MotiForge's regression and implementation documents.

### Short anchored arm gaps

`backends/arm_gap_repair.py::interpolate_arm_gaps` starts from the original local
SMPL pose, not from the network-correction delta. It merges touching/overlapping
effective same-side runs and requires the original maximum gap of 1/6 second.
The three immediately preceding and three immediately following frames must
have all shoulder/elbow/wrist scores >= 0.7, lie inside the inclusive native
crop and, when known, the half-open image bounds. Anchors cannot overlap another
known same-side gap. Insufficient evidence skips the hypothesis; no farther
anchor search, gap expansion, tail extrapolation or confidence promotion occurs.

A SciPy `RotationSpline` through the six original anchor poses reconstructs only
the original gap's shoulder/elbow/wrist local rotations. Every other pose sample
remains exact. The spline's own continuous angular rate/acceleration is not a
guarantee about finite-difference seams with the retained original trajectory;
the full original temporal guard remains mandatory. `_arm_pose_prediction`
shares the preserved-root/shape and full native-FK path with local correction.
The candidate must pass against the original, and against any accepted localized
candidate, using unchanged guard thresholds. Accepted whole-model predictions
are never replaced, and at most two model predictions are made.

`arm_gap_repair` records anchors, skipped reasons and changed frames;
`local_candidate_acceptance`, `gap_candidate_acceptance` and optional
`gap_vs_local_acceptance` preserve the selection evidence. The final selected
artifact alone determines applied flags/warnings. Save
`observation_gap_candidate.npz` separately even when the guard rejects it.
Backend identity includes this numerical helper; audit/off remain unchanged.

The two bounded dink08 experiments favored this original-gap-only method.
Searching for farther high-score blocks made motion smoother but altered visible
wrist motion and increased its projection error, so that expansion is not
implemented. The selected candidate further reduces shoulder acceleration over
the prior local correction, with a small opposite change in one robot-elbow
acceleration peak. This is not all-channel or ground-truth improvement; paired
full35 evidence is recorded in MotiForge's regression document.

## Optional foot-surface evidence from an existing prediction

The default video inference export remains unchanged: it does not reconstruct a
full mesh. To add geometric foot-surface evidence without rerunning video
preprocessing, inference, or ground stabilization, write a separate artifact:

```bash
PYTHONNOUSERSITE=1 python -m hmr4d.backends.motiforge export-foot-surface \
  /path/to/completed-prediction.npz \
  --output /path/to/prediction-with-foot-surface.npz \
  --asset-root /path/to/gvhmr-assets
```

This command needs only the existing GVHMR Python/body-model runtime and
`inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` under the asset root;
CUDA, video checkpoints, and the original video are not needed. Missing model
assets fail explicitly. It never substitutes joints for a missing foot surface,
never overwrites its input or an existing output, and publishes the new NPZ
atomically. The original world joints, SMPL parameters, camera arrays, contacts,
and `floor_correction_y` remain byte-value identical.

CPU `torch.no_grad()` batches of at most 64 frames use the existing full-mesh
SMPL-X body-model API with GVHMR's neutral, 10-beta, mean-hand defaults. Left and
right foot regions contain vertices whose ankle-plus-toe skinning weights
(`7 + 10` and `8 + 11`) sum to at least `0.5`. The per-side minimum mesh Y becomes
`foot_surface_y: float32[T, 2]`, in the same already-corrected Y-up world frame as
`pred_w_j3d` and `smpl_params_global.transl`. The command checks that reconstructed
body22 joints agree with the portable `fk_v2` result within 0.1 mm before publishing.

`motiforge_video.foot_surface` records:

- `schema: "smplx-foot-surface-v1"`, `foot_order: ["left", "right"]`;
- `up_axis: "y"`, `units: "m"`;
- `body_model_sha256`, `vertex_counts`, and the full `exporter_sha256` of
  `hmr4d/backends/foot_surface.py`;
- `helper_sha256`, covering the shared portable validation/publication/model
  loader and existing `BodyModelSMPLX` wrapper implementations.

`motiforge_video.foot_surface_export` records the input artifact path/content
SHA-256, its prior backend revision, the export backend revision, and the maximum
body22 consistency error. The top-level backend revision is updated while the
original video `source_path`, `source_sha256`, inference revision/options, and
floor diagnostics are preserved. `backend_revision()` covers the default adapter
and observation stability helper; the optional exporter has its own recorded
content SHA. This is additive protocol-3 evidence: old
artifacts remain valid and downstream surface-height use must be explicitly
enabled.

The surface trajectory is geometric evidence, **not a contact label or a new
ground correction**. It can represent real flight and retained source height
error. Robot collision clearance, target-height policy, and quality evaluation
remain downstream concerns. Neither per-frame hard grounding nor the diagnostic
gap-floor heuristic is part of this command.

## Optional body22 pose evidence from an existing prediction

To preserve explicit SMPL-X body orientation for downstream pose-aware A/B tests,
augment a completed portable NPZ in the isolated GVHMR environment:

```bash
PYTHONNOUSERSITE=1 python -m hmr4d.backends.motiforge export-body-pose \
  /path/to/completed-prediction.npz \
  --output /path/to/prediction-with-body-pose.npz \
  --asset-root /path/to/gvhmr-assets
```

This is **explicit A/B evidence, not default inference**, and has not demonstrated
an overall quality improvement. Full-clip comparisons on 2026-09-08, with the
same source positions/height and Mink + Ground, exposed the following tradeoffs:

- Feishu / Adam Lite: tracking p95 improved from 67.250 to 62.853 mm and
  orientation p95 from 28.358 to 22.140 degrees, but foot sliding increased from
  40.009 to 57.548 mm/s and the self-collision metric from 35.354 to 50.547 mm.
  Physical quality remained C.
- Tennis / G1: tracking p95 improved from 60.868 to 57.808 mm and fidelity from B
  to A, but foot sliding increased from 152.776 to 193.053 mm/s. Floating frames
  remained 16/313 in both runs, and physical quality remained C.

Consequently, pose export remains opt-in for comparisons; better tracking or
fidelity alone does not establish better overall motion quality. These tests do
not demonstrate that floating feet are solved.

This uses the existing CPU SMPL-X/PyTorch runtime and the same neutral model asset
as the foot-surface exporter. It does not run video preprocessing, inference,
upstream postprocessing, ground correction or retargeting. No new dependency or
default video option is introduced. It preserves every pre-existing array,
including world joints, global/in-camera parameters, floor correction, contacts
and any `foot_surface_y`, and adds only:

- `body22_world_rotations: float32[T, 22, 4]`: world-space **wxyz** unit quaternions;
- `body22_bind_positions: float32[22, 3]`: shaped neutral global positions in Y-up metres;
- `body22_bind_rotations: float32[22, 4]`: global identity **wxyz** quaternions,
  since SMPL zero local body pose uses identity joint frames.

The body22 order is pelvis, left/right hip, spine1, left/right knee, spine2,
left/right ankle, spine3, left/right foot, neck, left/right collar, head,
left/right shoulder, left/right elbow and left/right wrist. Its parents are
`[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]`.
No fingers or inferred end joints are added. The shaped neutral bind includes
the model's pelvis offset; it is not the first frame of the motion.

Before publication, the exporter requires **exactly constant per-frame betas**;
even a small dynamic shape change is rejected rather than silently collapsed
into one bind. It verifies the model parent chain, the neutral model against its
shaped skeleton, and every pose against the portable body22 world joints using
both full-model FK and SMPL-X's rigid parent-chain transforms. Each maximum
Euclidean joint error must be at most **0.1 mm**. The final float32 quaternions
and bind positions must also reconstruct the observed bone directions and joint
positions within that tolerance, using parent world rotations on neutral bone
offsets. Model work uses CPU `torch.no_grad()` chunks of at most 64 frames.

`motiforge_video.body_pose` records `schema: "smplx-body22-pose-v1"`, `up_axis: "y"`,
`units: "m"`, `quaternion_order: "wxyz"`, `joint_names`, `parents`,
`body_model_sha256`, the full `exporter_sha256` of `hmr4d/backends/body_pose.py`,
`helper_sha256`, `body22_max_error_m`, and `bind_reconstruction_max_error_m`.
`motiforge_video.body_pose_export` records the input artifact's exact bytes SHA,
path, parent backend revision and export backend revision. The original video
SHA, source path, inference options and ground diagnostics remain unchanged;
the top-level backend revision is updated to identify the export entry point.

Input/output must be different files. Existing outputs (including publication
races), malformed/nonfinite data, pickle-dependent arrays, missing model assets,
existing body-pose evidence and incompatible geometry are rejected without
replacing any artifact. Both optional exporters share neutral portable I/O and
record those helper hashes; neither imports the other's geometry algorithm.
Optional module changes do not become hidden dependencies of default inference:
the default identity covers the adapter/stability module only, while explicit
evidence records the additional exporter implementation identities.

`simple_vo_workers=1` is the deterministic default. Higher values parallelize
adjacent-frame matching but pycolmap RANSAC does not guarantee bitwise-identical
results, so the worker count is part of MotiForge's cache and provenance.

## 可选平地支撑姿态候选（2026-09-11，默认不执行）

显式命令从原始 portable 人体缓存及其同目录 ViTPose/bbox 观测生成新候选，不重跑视频识别，
不覆盖原始文件，也不改变默认推理的 backend revision/cache。此模块属于源侧，不包含机器人
求解器或 MotiForge core 依赖。

```bash
PYTHONNOUSERSITE=1 python -m hmr4d.backends.support_pose INPUT.npz \
  --output out/human-flat-support.npz --asset-root . --assume-flat-ground
```

必须显式确认平地，原输入标为固定机位，30Hz、6–1800帧，并保留其
`native/motiforge/preprocess/{vitpose,bbx}.pt`。必须在其他脚面/姿态增强之前执行；重复精修、
动态shape、世界/相机局部pose不一致、无可靠支撑或不匹配帧序拒绝。使用weights-only读取缓存。

`_support_mask`结合模型静止概率/3D足速与图像高分低速踝观测，`_support_bias`从实际脚面构造
平地高度偏差；双脚都无支撑证据时平滑采用共同偏差，避免逐脚独立压平真实腾空。`_fit`以
L-BFGS联合拟合pelvis Y与双髋/膝/踝局部旋转，包含脚面、XY、投影、姿态幅度及二阶时序先验。
shape、heading、root XZ、上肢局部pose保持；全模型FK、global/incam与脚面同步更新。
ViTPose热图分数不等于概率，可超过1。投影/姿态/损失保护只排除明显失控，不证明恢复真值。

输出保留源视频身份及原高度诊断，并新增`support_pose_refinement`、`support_pose_bias_y`与
合法`smplx-foot-surface-v1`证据，记录源工件、模型、观测和算法SHA；标记
`candidate_requires_review`。MotiForge端显式选`--foot-target-mode surface --ground`后走
原公共Job/Quality/Sink，不自动将候选当作最终结果。现有CLI/Web视频入口不自动调用本模块。

同十条90秒匹克球缓存验证：09关键浮脚14.61→0.34cm；物理B×1/C×9→A×1/B×1/C×8，
Fidelity仍B×6/C×4。6条脚滑下降、03/08/09/10四条脚滑上升；10真实轻跳仍保留但高度改变。
二维静止也可能是悬停抬脚，楼梯/物体支撑不适用，因此不能全批无条件采用。新增12项无模型
测试与既有后端测试共133项通过；完整源/机器人对照及代价记录于MotiForge对应回归文档。
本节点新增support_pose模块及其测试/本节说明；未改默认推理或源网络权重。
该实验入口只接受尚未附加几何证据的原始缓存；已含 `smplx-sequence-v1` 的新导出会明确拒绝，
防止优化后的姿态与保留下来的旧几何不一致，不自动移除或重写公共 sequence。

## 公共 SMPL-X sequence（2026-09-15）

最终预测现在默认附加 `smplx-sequence-v1`，保留原 protocol-3 字段；独立消费者可直接
使用标准化的参数、body22 旋转和 bind，而不必重新恢复人体。旧缓存可用 `export-smplx`
升级。它属于 Source 交换合同，不依赖 UMR 或机器人；详细字段、未观测手型、坐标枢轴、
验证和兼容性见 [SMPLX_SEQUENCE.md](SMPLX_SEQUENCE.md)。
