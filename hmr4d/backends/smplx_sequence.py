"""Engine-independent SMPL-X sequence export, attached to the selected prediction.

Original Y-up arrays remain intact. The additional Z-up axis-angle parameters
use explicit hand means, so a standard flat-hand SMPL-X model reproduces the
native surface without perception or robot-dependent fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hmr4d.backends.body_pose import BODY22_NAMES, BODY22_PARENTS, _pose_evidence
from hmr4d.backends.portable import (
    MODEL_RELATIVE_PATH, _load_body_model, _sha256, _validated_prediction,
    _write_new_artifact, validate_prediction_arrays,
)

SCHEMA = "smplx-sequence-v1"
Y_TO_Z = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
FIELDS = ("poses", "trans", "betas", "positions", "rotations", "bind_positions", "bind_rotations")


def sequence_evidence(arrays, model):
    """Preserve a complete body22 sequence and validate against the native FK."""
    metadata = validate_prediction_arrays(arrays)
    betas = arrays["smpl_params_global.betas"]
    if not np.array_equal(betas, np.broadcast_to(betas[:1], betas.shape)):
        raise ValueError("SMPL-X sequence currently requires constant betas; refusing to average shape")
    pose, error, bind_error = _pose_evidence(arrays, model)
    frames = len(betas)
    change = Rotation.from_matrix(Y_TO_Z)
    root = arrays["smpl_params_global.global_orient"]
    poses = np.zeros((frames, 165), dtype=np.float64)
    poses[:, :3] = (change * Rotation.from_rotvec(root)).as_rotvec()
    poses[:, 3:66] = arrays["smpl_params_global.body_pose"]
    # GVHMR predicts body22, not fingers. Explicitly retain the model's default
    # relaxed hands, rather than silently replacing them with flat zero hands.
    poses[:, 75:120] = model.bm.left_hand_mean.detach().cpu().numpy()
    poses[:, 120:165] = model.bm.right_hand_mean.detach().cpu().numpy()
    pivot = pose["body22_bind_positions"][0].astype(np.float64)
    trans = arrays["smpl_params_global.transl"].astype(np.float64)
    # SMPL-X rotates around its shaped pelvis, not the world origin.
    trans = (trans + pivot) @ Y_TO_Z.T - pivot
    q = pose["body22_world_rotations"]
    rotated = change * Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4))
    rotations = rotated.as_quat()[:, [3, 0, 1, 2]].reshape(frames, 22, 4)
    bind_q = np.tile(change.as_quat()[[3, 0, 1, 2]], (22, 1))
    return {
        "poses": poses, "trans": trans, "betas": betas.copy(),
        "positions": arrays["pred_w_j3d"].astype(np.float64) @ Y_TO_Z.T,
        "rotations": rotations,
        "bind_positions": pose["body22_bind_positions"].astype(np.float64) @ Y_TO_Z.T,
        "bind_rotations": bind_q,
    }, {
        "schema": SCHEMA, "model_type": "smplx", "gender": "neutral",
        "output_up": "z", "units": "m", "rotation_type": "axis-angle-radians",
        "quaternion_order": "wxyz", "flat_hand_mean": True,
        "hands": "model-mean-unobserved", "face": "zero-unobserved",
        "joint_names": list(BODY22_NAMES), "parents": list(BODY22_PARENTS),
        "fps": float(metadata["fps"]), "frame_count": frames,
        "full_sequence": metadata.get("max_frames") is None, "has_objects": False,
        "body22_max_error_m": error, "bind_reconstruction_max_error_m": bind_error,
    }


def add_sequence(arrays, model_path, *, model_factory=None):
    """Add standard parameter evidence to one NPZ payload without changing old arrays."""
    model_path = Path(model_path)
    metadata = validate_prediction_arrays(arrays)
    if "smplx_sequence" in metadata or any(k.startswith("smplx.") for k in arrays):
        raise ValueError("SMPL-X sequence already exists; refusing to replace it")
    evidence, info = sequence_evidence(arrays, (model_factory or _load_body_model)(model_path))
    info.update(body_model_sha256=_sha256(model_path), exporter_sha256=_sha256(Path(__file__)))
    metadata["smplx_sequence"] = info
    result = dict(arrays)
    result.update({f"smplx.{key}": value for key, value in evidence.items()})
    result["motiforge_video_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return result


def enrich_prediction(portable, model_path):
    """Called once after the final complete prediction has been selected."""
    from hmr4d.backends.motiforge import _portable_arrays
    from hmr4d.backends.surface_ground import add_surface_ground

    model = _load_body_model(Path(model_path))
    arrays = add_surface_ground(
        _portable_arrays(portable), model,
        enabled=bool(portable["motiforge_video"].get("ground_stabilization", {}).get("enabled", False)),
        model_digest=_sha256(Path(model_path)),
        assume_grounded=portable["motiforge_video"].get("assume_grounded", False),
    )
    enriched = add_sequence(arrays, model_path, model_factory=lambda _: model)
    portable["smpl_params_global"]["transl"] = enriched["smpl_params_global.transl"]
    portable.update({key: value for key, value in enriched.items()
                     if not key.startswith("smpl_params_") and key != "motiforge_video_json"})
    portable["motiforge_video"] = json.loads(str(enriched["motiforge_video_json"]))


def export_sequence(input_path, output_path, asset_root):
    """Upgrade an existing complete prediction without rerunning video inference."""
    input_path, output_path = Path(input_path).resolve(), Path(output_path).absolute()
    if input_path == output_path.resolve() or output_path.exists():
        raise ValueError("SMPL-X sequence export requires a new output path")
    arrays, metadata, digest = _validated_prediction(input_path)
    result = add_sequence(arrays, Path(asset_root) / MODEL_RELATIVE_PATH)
    info = json.loads(str(result["motiforge_video_json"]))
    info["smplx_sequence"]["source_artifact_sha256"] = digest
    result["motiforge_video_json"] = np.asarray(json.dumps(info, ensure_ascii=False, sort_keys=True))
    _write_new_artifact(output_path, result)
    return {"prediction": str(output_path), **info["smplx_sequence"]}
