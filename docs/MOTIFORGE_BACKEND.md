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
  `hmr4d/backends/foot_surface.py`.

`motiforge_video.foot_surface_export` records the input artifact path/content
SHA-256, its prior backend revision, the export backend revision, and the maximum
body22 consistency error. The top-level backend revision is updated while the
original video `source_path`, `source_sha256`, inference revision/options, and
floor diagnostics are preserved. `backend_revision()` retains its original
single-file identity for default video-cache compatibility; the optional exporter
has its own recorded content SHA. This is additive protocol-3 evidence: old
artifacts remain valid and downstream surface-height use must be explicitly
enabled.

The surface trajectory is geometric evidence, **not a contact label or a new
ground correction**. It can represent real flight and retained source height
error. Robot collision clearance, target-height policy, and quality evaluation
remain downstream concerns. Neither per-frame hard grounding nor the diagnostic
gap-floor heuristic is part of this command.

`simple_vo_workers=1` is the deterministic default. Higher values parallelize
adjacent-frame matching but pycolmap RANSAC does not guarantee bitwise-identical
results, so the worker count is part of MotiForge's cache and provenance.
