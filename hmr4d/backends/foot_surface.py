"""Optional SMPL-X foot-surface evidence export from a portable prediction.

This module runs only in the isolated GVHMR environment. It does not estimate
contact, alter the prediction, or run video inference / floor stabilization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hmr4d.backends.portable import (
    BATCH_FRAMES,
    BODY22_TOLERANCE_M,
    MODEL_RELATIVE_PATH,
    _helper_sha256,
    _load_body_model,
    _sha256,
    _validated_prediction,
    _write_new_artifact,
)

SURFACE_SCHEMA = "smplx-foot-surface-v1"


def _surface_evidence(arrays: dict[str, np.ndarray], model) -> tuple[np.ndarray, list[int], float]:
    import torch

    model = model.eval() if hasattr(model, "eval") else model
    model = model.cpu() if hasattr(model, "cpu") else model
    weights = model.bm.lbs_weights.detach().cpu()
    if weights.ndim != 2 or weights.shape[1] < 12 or not torch.isfinite(weights).all():
        raise ValueError("SMPL-X model has invalid ankle/foot skinning weights")
    masks = (weights[:, [7, 10]].sum(dim=1) >= 0.5, weights[:, [8, 11]].sum(dim=1) >= 0.5)
    counts = [int(mask.sum().item()) for mask in masks]
    if not all(counts):
        raise ValueError("SMPL-X model has no vertices in one or both foot regions")
    frames = len(arrays["pred_w_j3d"])
    surface = np.empty((frames, 2), dtype=np.float32)
    max_error = 0.0
    with torch.no_grad():
        for start in range(0, frames, BATCH_FRAMES):
            end = min(frames, start + BATCH_FRAMES)
            params = {
                field: torch.as_tensor(arrays[f"smpl_params_global.{field}"][start:end], dtype=torch.float32)
                for field in ("body_pose", "betas", "global_orient", "transl")
            }
            prediction = model(**params)
            vertices = prediction.vertices.detach().cpu()
            joints = prediction.joints.detach().cpu()
            if vertices.shape != (end - start, len(weights), 3) or not torch.isfinite(vertices).all():
                raise ValueError("SMPL-X returned invalid full-mesh vertices")
            if joints.ndim != 3 or joints.shape[0] != end - start or joints.shape[1] < 22 or joints.shape[2] != 3:
                raise ValueError("SMPL-X returned invalid body22 joints")
            actual = joints[:, :22].numpy()
            if not np.isfinite(actual).all():
                raise ValueError("SMPL-X returned nonfinite body22 joints")
            error = float(np.max(np.abs(actual - arrays["pred_w_j3d"][start:end])))
            max_error = max(max_error, error)
            if error > BODY22_TOLERANCE_M:
                raise ValueError(
                    f"SMPL-X body22 differs from portable fk_v2 world joints by {error:.6g} m "
                    f"(limit {BODY22_TOLERANCE_M:g} m); refusing mismatched surface evidence"
                )
            for side, mask in enumerate(masks):
                surface[start:end, side] = vertices[:, mask, 1].amin(dim=1).numpy()
    return surface, counts, max_error


def export_foot_surface(
    input_path: Path,
    output_path: Path,
    asset_root: Path,
    *,
    backend_id: str,
    model_factory=None,
) -> dict[str, Any]:
    """Add optional evidence to a new artifact, preserving the existing motion."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().absolute()
    if input_path == output_path.resolve():
        raise ValueError("Foot-surface output must differ from the input artifact")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Foot-surface output already exists: {output_path}")
    model_path = asset_root.expanduser().resolve() / MODEL_RELATIVE_PATH
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Foot-surface export requires the SMPL-X body model: {model_path}")
    arrays, metadata, source_digest = _validated_prediction(input_path)
    model_digest = _sha256(model_path)
    model = (model_factory or _load_body_model)(model_path)
    surface, counts, max_error = _surface_evidence(arrays, model)
    metadata["foot_surface_export"] = {
        "source_artifact": str(input_path),
        "source_artifact_sha256": source_digest,
        "source_backend_revision": metadata.get("gvhmr_backend_revision"),
        "backend_revision": backend_id,
        "body22_max_error_m": max_error,
    }
    metadata["foot_surface"] = {
        "schema": SURFACE_SCHEMA,
        "foot_order": ["left", "right"],
        "up_axis": "y",
        "units": "m",
        "body_model_sha256": model_digest,
        "vertex_counts": counts,
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "helper_sha256": _helper_sha256(),
    }
    metadata["gvhmr_backend_revision"] = backend_id
    arrays["foot_surface_y"] = surface
    arrays["motiforge_video_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    _write_new_artifact(output_path, arrays)
    return {
        "prediction": str(output_path),
        "frames": len(surface),
        **metadata["foot_surface_export"],
        **metadata["foot_surface"],
    }
