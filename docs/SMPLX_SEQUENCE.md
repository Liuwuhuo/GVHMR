# Public SMPL-X sequence extension

The MotiForge protocol remains version 3. After selecting the final prediction, the backend now
adds an engine-independent `smplx-sequence-v1` block to the portable NPZ. Original Y-up parameters,
body22 world joints, confidence and observation diagnostics remain intact. Video inference and
candidate selection are unchanged. This is a project interchange schema using standard SMPL-X
parameters, not a file format mandated by SMPL-X or UMR.

`motiforge_video.smplx_sequence` declares model type/gender, Z-up/meters, FPS/frame count, full-sequence
status, wxyz global rotations, axis-angle radians, model/exporter identity and unobserved hand/face
semantics. Arrays are prefixed `smplx.`:

- `poses`: T×165 standard root/body/jaw/eyes/hands axis-angle parameters.
- `trans`: T×3, `betas`: T×10; shape must be constant over time.
- `positions`: T×22×3, `rotations`: T×22×4 world body22 evidence.
- `bind_positions`: 22×3, `bind_rotations`: 22×4 shaped neutral bind in the same Z-up frame.

GVHMR predicts body22, not fingers or face. The new parameters explicitly retain the model's
relaxed hand means and declare `flat_hand_mean=true`, `hands=model-mean-unobserved` and
`face=zero-unobserved`. Consumers must not interpret defaults as observed finger motion.
No robot assets, UMR modules or MotiForge Python imports are needed to produce this block.

Coordinate conversion includes the shaped pelvis pivot: for old translation t, pelvis p and
world rotation R, new translation is R(t+p)-p. Rotating translation alone is incorrect. Full model
FK and serialized bind/rotation evidence are checked against the existing prediction before
publishing. Capped recovery is marked `full_sequence=false` rather than silently called a full clip.

New inference emits this extension by default. Existing cached predictions can be upgraded without
rerunning video inference, using the GVHMR environment from the repository root:

```bash
python -m hmr4d.backends.motiforge export-smplx old.npz --output new.npz --asset-root /path/to/GVHMR
```

Output must be new; existing files or already-enriched predictions are not overwritten. The body
model comes from `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` under the asset root.
The backend's lightweight `capabilities` advertises `smplx_sequence=smplx-sequence-v1`;
helper-aware backend revision includes exporter and validation code so source caches can invalidate.

The matching MotiForge reader maps this block into `HumanMotion/body22 + smplx-geometry-v1` without
loading a model. Original files remain readable, and an explicit `geometry=False` reader option
preserves legacy position-only behavior. Saving another geometry cache is optional. Retaining
rotations/bind can change downstream native targets; the export does not claim unchanged IK output.

Validation on the complete tennis example: 313 frames at 30 Hz, no motion crop. Re-evaluating the
new standard flat-hand-mean parameters against the native relaxed-hand model in the converted world
frame gives maximum vertex difference 6.20e-7 m and first-55-joint difference 5.51e-7 m. Synthetic
checks cover a nonzero pelvis pivot, explicit hands, immutable original arrays and partial timeline;
flow tests verify export occurs once after final candidate selection. Full backend suite on
2026-09-15: 137 unittest cases passed, including rejection of stale sequence evidence by the
optional support-pose candidate exporter. This candidate exporter does not yet regenerate an
existing sequence; use an original unenriched cache for that experiment.
MotiForge evidence and integration results are in its `docs/gvhmr_smplx_source_bridge.md` and
`out/gvhmr-smplx-bridge-20260915/`. No checkpoints or trained robot assets are committed here.
