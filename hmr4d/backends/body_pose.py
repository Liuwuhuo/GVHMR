"""Explicit, validated SMPL-X body22 rotations and shaped neutral-bind export.

This adds evidence to an existing portable artifact, never rerunning perception,
changing height or manufacturing finger joints. It is isolated from video infer.
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

POSE_SCHEMA = "smplx-body22-pose-v1"
BODY22_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
BODY22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
POSE_ARRAYS = ("body22_world_rotations", "body22_bind_positions", "body22_bind_rotations")


def _body22_error(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"{label} returned invalid body22 joints")
    error = float(np.linalg.norm(actual - expected, axis=-1).max())
    if error > BODY22_TOLERANCE_M:
        raise ValueError(
            f"{label} body22 differs by {error:.6g} m (limit {BODY22_TOLERANCE_M:g} m); "
            "refusing incompatible pose evidence"
        )
    return error


def _pose_evidence(arrays: dict[str, np.ndarray], model) -> tuple[dict[str, np.ndarray], float, float]:
    import torch
    from pytorch3d.transforms import (
        axis_angle_to_matrix,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
    from smplx.lbs import batch_rigid_transform

    model = model.eval().cpu()
    parents = model.bm.parents.detach().cpu()[:22]
    if parents.tolist() != list(BODY22_PARENTS):
        raise ValueError("SMPL-X body22 parent chain does not match the portable pose schema")
    frames = len(arrays["pred_w_j3d"])
    rotations = np.empty((frames, 22, 4), dtype=np.float32)
    max_error = 0.0
    bind_error = 0.0
    with torch.no_grad():
        betas = torch.as_tensor(arrays["smpl_params_global.betas"][:1], dtype=torch.float32)
        bind = model.get_skeleton(betas).detach().cpu()
        if bind.shape != (1, 22, 3) or not torch.isfinite(bind).all():
            raise ValueError("SMPL-X returned invalid shaped neutral body22 bind positions")
        neutral = model(
            betas=betas,
            body_pose=torch.zeros((1, 63)),
            global_orient=torch.zeros((1, 3)),
            transl=torch.zeros((1, 3)),
        ).joints.detach().cpu().numpy()[:, :22]
        max_error = _body22_error(neutral, bind.numpy(), "SMPL-X neutral FK")
        bind_positions = bind[0].numpy().astype(np.float32)

        for start in range(0, frames, BATCH_FRAMES):
            end = min(frames, start + BATCH_FRAMES)
            params = {
                field: torch.as_tensor(arrays[f"smpl_params_global.{field}"][start:end], dtype=torch.float32)
                for field in ("body_pose", "betas", "global_orient", "transl")
            }
            actual = model(**params).joints.detach().cpu().numpy()[:, :22]
            expected = arrays["pred_w_j3d"][start:end]
            max_error = max(max_error, _body22_error(actual, expected, "SMPL-X full-model FK"))
            pose = torch.cat((params["global_orient"], params["body_pose"]), dim=-1)
            local_rotations = axis_angle_to_matrix(pose.double().reshape(-1, 22, 3))
            fk_joints, transforms = batch_rigid_transform(
                local_rotations, bind.double().expand(end - start, -1, -1), parents,
                dtype=torch.float64,
            )
            fk_world = (fk_joints + params["transl"].double()[:, None]).numpy()
            max_error = max(max_error, _body22_error(fk_world, expected, "SMPL-X rigid-chain FK"))
            quaternions = matrix_to_quaternion(transforms[:, :, :3, :3])
            quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
            rotations[start:end] = quaternions.float().numpy()

            # Validate the *serialized* rotations and bind, not only internal FK.
            world_rotations = quaternion_to_matrix(torch.from_numpy(rotations[start:end]).double()).numpy()
            reconstructed = np.empty(expected.shape, dtype=np.float64)
            reconstructed[:, 0] = expected[:, 0]
            for joint, parent in enumerate(BODY22_PARENTS[1:], start=1):
                offset = bind_positions[joint].astype(np.float64) - bind_positions[parent]
                reconstructed[:, joint] = reconstructed[:, parent] + world_rotations[:, parent] @ offset
            bind_error = max(bind_error, _body22_error(reconstructed, expected, "Serialized rotation/bind FK"))

    bind_rotations = np.zeros((22, 4), dtype=np.float32)
    bind_rotations[:, 0] = 1.0  # SMPL zero local pose has identity global joint frames.
    return {
        "body22_world_rotations": rotations,
        "body22_bind_positions": bind_positions,
        "body22_bind_rotations": bind_rotations,
    }, max_error, bind_error


def export_body_pose(
    input_path: Path,
    output_path: Path,
    asset_root: Path,
    *,
    backend_id: str,
    model_factory=None,
) -> dict[str, Any]:
    """Publish new optional pose evidence, preserving every existing motion array."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().absolute()
    if input_path == output_path.resolve():
        raise ValueError("Body-pose output must differ from the input artifact")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Body-pose output already exists: {output_path}")
    model_path = asset_root.expanduser().resolve() / MODEL_RELATIVE_PATH
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Body-pose export requires the SMPL-X body model: {model_path}")
    arrays, metadata, source_digest = _validated_prediction(input_path)
    if any(key in arrays for key in POSE_ARRAYS) or any(key in metadata for key in ("body_pose", "body_pose_export")):
        raise ValueError("Portable input already contains body-pose evidence; refusing to replace it")
    betas = arrays["smpl_params_global.betas"]
    if not np.array_equal(betas, np.broadcast_to(betas[:1], betas.shape)):
        raise ValueError("Body-pose export requires constant betas; dynamic shape cannot use one neutral bind")
    model_digest = _sha256(model_path)
    evidence, max_error, bind_error = _pose_evidence(arrays, (model_factory or _load_body_model)(model_path))
    metadata["body_pose_export"] = {
        "source_artifact": str(input_path),
        "source_artifact_sha256": source_digest,
        "source_backend_revision": metadata.get("gvhmr_backend_revision"),
        "backend_revision": backend_id,
    }
    metadata["body_pose"] = {
        "schema": POSE_SCHEMA,
        "up_axis": "y",
        "units": "m",
        "quaternion_order": "wxyz",
        "joint_names": list(BODY22_NAMES),
        "parents": list(BODY22_PARENTS),
        "body_model_sha256": model_digest,
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "helper_sha256": _helper_sha256(),
        "body22_max_error_m": max_error,
        "bind_reconstruction_max_error_m": bind_error,
    }
    metadata["gvhmr_backend_revision"] = backend_id
    arrays.update(evidence)
    arrays["motiforge_video_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    _write_new_artifact(output_path, arrays)
    return {
        "prediction": str(output_path),
        "frames": len(arrays["pred_w_j3d"]),
        **metadata["body_pose_export"],
        **metadata["body_pose"],
    }
