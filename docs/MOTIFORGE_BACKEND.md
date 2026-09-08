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

The upstream static-camera postprocessor deliberately corrects only horizontal
stationary-joint drift. The MotiForge backend additionally estimates the slow
vertical floor component from sustained ankle/foot contacts predicted by the
same checkpoint. Frames without confident static support are not floor anchors;
low static probability can also mean sliding contact or fast steps, not flight. The
interpolated correction is applied equally to world joints and global SMPL
translation, is capped at 0.25 m, and cannot push the lowest foot through the
estimated floor. The correction is also limited to 0.2 m/s so a contact switch
cannot create a one-frame root jump. MotiForge enables this behavior by default. Use
`--gvhmr-no-ground-stabilization` on `motiforge video` to reproduce the raw
upstream world-Y behavior. Robot sole clearance and collision grounding remain
downstream retarget concerns, so Mink dataset generation should still use
`--ground`.

`ground_stabilization.static_support_missing_frames` counts frames without a
retained static-foot anchor. The old `flight_frames` key remains as a deprecated
alias with exactly the same value; it is not a flight classification. This naming
clarification does not change the `contact-floor-v1` numerical algorithm or the
protocol-3 format.

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
