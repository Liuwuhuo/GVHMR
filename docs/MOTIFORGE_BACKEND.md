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
parameters, camera intrinsics, source SHA-256, inference options, the GVHMR Git
revision and backend revision. Writes use temporary files followed by atomic
replacement; one bad video does not abort the rest of a batch.

`simple_vo_workers=1` is the deterministic default. Higher values parallelize
adjacent-frame matching but pycolmap RANSAC does not guarantee bitwise-identical
results, so the worker count is part of MotiForge's cache and provenance.
