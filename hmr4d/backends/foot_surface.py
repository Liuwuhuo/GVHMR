"""Optional SMPL-X foot-surface evidence export from a portable prediction.

This module runs only in the isolated GVHMR environment. It does not estimate
contact, alter the prediction, or run video inference / floor stabilization.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

MODEL_RELATIVE_PATH = "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
SURFACE_SCHEMA = "smplx-foot-surface-v1"
BATCH_FRAMES = 64
BODY22_TOLERANCE_M = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_body_model(model_path: Path):
    from hmr4d.utils.body_model.body_model_smplx import BodyModelSMPLX

    # Match make_smplx("supermotion") / SmplxLite's default body and hand pose.
    return BodyModelSMPLX(
        model_path=str(model_path.parent.parent),
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        num_pca_comps=12,
        flat_hand_mean=False,
    )


def _validated_prediction(input_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    # Bind provenance to exactly the portable bytes parsed, not a second read.
    payload = input_path.read_bytes()
    source_digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    try:
        metadata = json.loads(arrays["motiforge_video_json"].item())
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Portable input requires scalar motiforge_video_json metadata") from exc
    if not isinstance(metadata, dict) or metadata.get("protocol") != 3:
        raise ValueError("Foot-surface export requires a protocol-3 portable prediction")
    joints = arrays.get("pred_w_j3d")
    if joints is None or joints.ndim != 3 or joints.shape[1:] != (22, 3) or not len(joints):
        raise ValueError("pred_w_j3d must have shape (T, 22, 3) with T > 0")
    frames = len(joints)
    try:
        fps = float(metadata["fps"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Portable prediction requires a finite positive fps") from exc
    if not np.isfinite(fps) or fps <= 0 or metadata.get("normalized_num_frames") != frames:
        raise ValueError("Portable fps/frame-count metadata does not match the prediction")
    if not isinstance(metadata.get("source_sha256"), str) or not metadata["source_sha256"]:
        raise ValueError("Portable prediction is missing the original video source_sha256")
    required = {"pred_w_j3d": (frames, 22, 3)}
    for name, width in (("body_pose", 63), ("betas", 10), ("global_orient", 3), ("transl", 3)):
        required[f"smpl_params_global.{name}"] = (frames, width)
    for key, shape in required.items():
        value = arrays.get(key)
        if value is None or value.shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite numeric values")
    if "floor_correction_y" in arrays:
        value = arrays["floor_correction_y"]
        if value.shape != (frames,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("floor_correction_y must contain T finite numeric values")
    return arrays, metadata, source_digest


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


def _write_new_artifact(output_path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish atomically without replacing a file created by another process."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.link(temporary, output_path)  # Atomic exclusive creation, never os.replace.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
