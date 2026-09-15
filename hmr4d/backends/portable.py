"""Shared validation and publication for explicit portable-evidence exports.

Default video inference does not import this module. Heavy body-model imports
remain inside the isolated export command's loader.
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
BATCH_FRAMES = 64
BODY22_TOLERANCE_M = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _helper_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative: _sha256(root / relative)
        for relative in (
            "hmr4d/backends/portable.py",
            "hmr4d/utils/body_model/body_model_smplx.py",
        )
    }


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
    metadata = validate_prediction_arrays(arrays)
    return arrays, metadata, source_digest


def validate_prediction_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Validate file and in-memory exports through the same prediction contract."""
    try:
        metadata = json.loads(arrays["motiforge_video_json"].item())
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Portable input requires scalar motiforge_video_json metadata") from exc
    if not isinstance(metadata, dict) or metadata.get("protocol") != 3:
        raise ValueError("Evidence export requires a protocol-3 portable prediction")
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
    return metadata


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
        os.link(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
