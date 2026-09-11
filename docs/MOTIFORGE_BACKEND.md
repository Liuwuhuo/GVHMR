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

`python -m hmr4d.backends.motiforge capabilities` provides lightweight protocol,
mode and implementation identity negotiation without importing Torch or loading
checkpoints. Default source identity covers the adapter and its observation
stability helper; a helper change must not reuse an older prediction cache.

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
parameters, camera intrinsics, left/right foot-contact confidence, the applied
world-Y floor correction, source SHA-256, inference options, the GVHMR Git
revision and backend revision. Writes use temporary files followed by atomic
replacement; one bad video does not abort the rest of a batch.

The upstream static-camera postprocessor includes an XYZ camera-root correction
with a 0.25 m discrepancy dead zone, followed by a horizontal-only static-joint
correction. It does not leave Y entirely untouched, but residual height drift
inside the camera dead zone can remain.

The backend's legacy `contact-floor-v1` estimates the slow floor component from
sustained ankle/foot static probabilities predicted by the same checkpoint.
Frames without confident static support are not floor anchors: low static
probability can also mean sliding contact or fast steps, not flight. Its desired
interpolated floor correction is clipped to +/-0.25 m before a joint-height
ceiling and a 0.2 m/s rate constraint are applied. This is not mesh-sole collision
grounding.

Only when both `static_camera` and `ground_stabilization` are enabled, the backend
uses `static-camera-contact-floor-v2`: reconstruct the raw in-camera pelvis,
map it with the original frame-zero camera-to-world rotation, and Gaussian-filter
only the world-minus-camera Y discrepancy with sigma=0.5 seconds (nearest edge
padding, truncate=4). The camera translation is not independently smoothed, so
real vertical motion shared by both representations cancels before filtering.
The first camera correction sample is subtracted to retain the original initial
height gauge. The existing floor algorithm then runs on this camera-corrected
prediction. Camera and floor corrections are summed and jointly projected through
the same 0.2 m/s Lipschitz minorant, preventing two individually bounded stages
from exceeding the total speed limit. The total correction is not subject to a
separate 0.25 m amplitude cap. Exactly the same exported `floor_correction_y` is
subtracted once from the original world-joint Y and global SMPL translation Y;
XZ, root-relative pose, in-camera parameters and source confidence are unchanged
apart from float32 rounding. This does not hard-snap each frame to the floor.

Dynamic cameras and disabled stabilization retain the exact legacy numerical
path. Missing or insufficient static support also retains the legacy fallback.
Missing, malformed or nonfinite in-camera parameters fall back with an explicit
`camera_stage.reason` and `detail`; they are not silently treated as reliable
camera evidence. Top-level ground diagnostics describe the final total
correction, final support-height p95 error and final speed, while `camera_stage`
and `floor_stage` describe their individual stages. In particular a floor stage
below its 5 mm threshold can still produce an applied camera-only total.

MotiForge enables source ground stabilization by default. Use
`--gvhmr-no-ground-stabilization` on `motiforge video` to reproduce the raw
upstream world-Y behavior. Robot sole clearance and collision grounding remain
downstream retarget concerns, so Mink dataset generation should still use
`--ground`.

`ground_stabilization.static_support_missing_frames` counts frames without a
retained static-foot anchor. The old `flight_frames` key remains as a deprecated
alias with exactly the same value; it is not a flight classification. The legacy
`contact-floor-v1` numerical helper and protocol-3 format remain unchanged. The
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
The adapter calls `_prepare_observation_data` before the single normal model
prediction and `_finish_observation_stability` after portable geometry export.
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
All 67 backend unit tests pass. No threshold was tuned to exclude this one clip.

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
