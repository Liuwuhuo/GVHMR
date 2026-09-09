"""Read-only CPU audit of original incam evidence versus a fixed-gauge world source.

Run in the existing GVHMR environment. No camera, body, ground or BVH fitting is
performed. The frame-zero root-derived extrinsic is an estimated gauge, not a
calibrated camera. Writes only the explicitly requested new JSON audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from hmr4d.backends.portable import _load_body_model, _validated_prediction
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

FIELDS = ("body_pose", "betas", "global_orient", "transl")
BODY12 = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
NAMES12 = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32, device="cpu")


def stats(values) -> dict | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return None
    require(bool(np.isfinite(values).all()), "Nonfinite audit metric")
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def frame_index(seconds: float, fps: float) -> int:
    require(math.isfinite(seconds) and seconds >= 0, "Interval endpoints must be finite and nonnegative")
    index = round(seconds * fps)
    require(abs(index - seconds * fps) < 1e-4, "Interval endpoints must fall on video frames")
    return index


def projection_report(joints, intrinsics, observed, mask) -> dict:
    camera = joints[:, BODY12]
    require(
        bool((camera[..., 2] > 0.1).all()), "Projected joint behind camera or near projection singularity"
    )
    homogeneous = camera @ intrinsics.T
    pixels = homogeneous[..., :2] / homogeneous[..., 2:]
    errors = torch.linalg.vector_norm(pixels - observed[..., :2], dim=-1).numpy()
    return {
        "error_px": stats(errors[mask]),
        "reliable_samples": int(mask.sum()),
        "per_joint_error_px": {
            name: stats(errors[:, index][mask[:, index]]) for index, name in enumerate(NAMES12)
        },
        "per_frame_median_error_px": [
            stats(row[selected]) for row, selected in zip(errors, mask, strict=True)
        ],
        "camera_depth_m": stats(camera[..., 2].numpy()),
    }


def audit(args) -> dict:
    arrays, metadata, source_digest = _validated_prediction(args.input)
    frames = len(arrays["pred_w_j3d"])
    fps = float(metadata["fps"])
    start, stop = frame_index(args.start, fps), frame_index(args.end, fps)
    require(0 <= start < stop <= frames, "Audit interval outside complete source timeline")
    require(
        metadata.get("static_camera") is True and not metadata.get("mirror"),
        "Fixed unmirrored source required",
    )
    for name, width in (("body_pose", 63), ("betas", 10), ("global_orient", 3), ("transl", 3)):
        value = arrays.get(f"smpl_params_incam.{name}")
        require(value is not None and value.shape == (frames, width), f"Invalid incam {name}")
        require(bool(np.isfinite(value).all()), f"Nonfinite incam {name}")
    require(arrays["K_fullimg"].shape == (frames, 3, 3), "Intrinsics timeline mismatch")
    np.testing.assert_array_equal(
        arrays["K_fullimg"], np.broadcast_to(arrays["K_fullimg"][:1], (frames, 3, 3))
    )
    require(
        arrays["foot_surface_y"].shape == (frames, 2), "Enhanced source must contain complete foot surface"
    )
    keypoints = torch.load(args.keypoints, map_location="cpu", weights_only=True)
    require(
        isinstance(keypoints, torch.Tensor) and keypoints.shape == (frames, 17, 3),
        "Keypoint timeline/schema mismatch",
    )
    require(bool(torch.isfinite(keypoints).all()), "Nonfinite original keypoint observations")
    observed = keypoints[start:stop, 5:17].to(dtype=torch.float32)
    mask = (observed[..., 2] >= 0.5).numpy()
    require(bool(mask.any()), "No reliable observations in requested audit interval")
    require(
        args.model.name == "SMPLX_NEUTRAL.npz" and args.model.is_file(),
        "Expected licensed neutral SMPL-X model",
    )
    model = _load_body_model(args.model).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def params(first: int, last: int, prefix="smpl_params_global"):
        return {key: tensor(arrays[f"{prefix}.{key}"][first:last]) for key in FIELDS}

    foot_ids = [torch.where(model.bm.lbs_weights[:, ids].sum(-1) >= 0.5)[0] for ids in ((7, 10), (8, 11))]
    with torch.no_grad():
        first_world = model(**params(0, 1))
        first_incam = model(**params(0, 1, "smpl_params_incam"))
        rw = axis_angle_to_matrix(tensor(arrays["smpl_params_global.global_orient"]))
        rc = axis_angle_to_matrix(tensor(arrays["smpl_params_incam.global_orient"]))
        rotation = rc[0] @ rw[0].T
        translation = first_incam.joints[0, 0] - rotation @ first_world.joints[0, 0]
        frame0_error = float(
            (first_world.joints[:, :22] @ rotation.T + translation - first_incam.joints[:, :22]).abs().max()
        )
        require(frame0_error < 1e-4, f"Frame-zero world/incam rigid transformation mismatch: {frame0_error}")
        max_joint_error, max_foot_error = 0.0, 0.0
        for first in range(0, frames, 32):
            last = min(first + 32, frames)
            full = model(**params(first, last))
            max_joint_error = max(
                max_joint_error,
                float((full.joints[:, :22] - tensor(arrays["pred_w_j3d"][first:last])).abs().max()),
            )
            feet = torch.stack([full.vertices[:, ids, 1].amin(1) for ids in foot_ids], -1)
            max_foot_error = max(
                max_foot_error, float((feet - tensor(arrays["foot_surface_y"][first:last])).abs().max())
            )
        require(
            max_joint_error < 1e-4 and max_foot_error < 1e-4, "Full-timeline world FK/foot geometry mismatch"
        )
        incam = model(**params(start, stop, "smpl_params_incam")).joints[:, :22]
        world = model(**params(start, stop)).joints[:, :22]
        intrinsics = tensor(arrays["K_fullimg"][0])
        implied_rotations = rc @ rw.transpose(-1, -2)
        deviation = torch.rad2deg(
            torch.linalg.vector_norm(matrix_to_axis_angle(implied_rotations @ rotation.T), dim=-1)
        ).numpy()
        direct = projection_report(incam, intrinsics, observed, mask)
        fixed = projection_report(world @ rotation.T + translation, intrinsics, observed, mask)
    pose_equal = np.all(
        arrays["smpl_params_global.body_pose"] == arrays["smpl_params_incam.body_pose"], axis=1
    )
    return {
        "kind": "fixed_camera_world_incam_consistency_readonly_v1",
        "input": str(args.input.resolve()),
        "input_sha256": source_digest,
        "keypoints": str(args.keypoints.resolve()),
        "keypoints_sha256": sha256(args.keypoints),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "script_sha256": sha256(Path(__file__)),
        "source_video": metadata["source_path"],
        "source_video_sha256": metadata["source_sha256"],
        "frames": frames,
        "fps": fps,
        "window_half_open_frames": [start, stop],
        "window_half_open_seconds": [args.start, args.end],
        "original_incam_role": (
            "Original model evidence, not refitted to the current corrected/refined world trajectory."
        ),
        "world_incam_body_pose_equal_frames": int(pose_equal.sum()),
        "world_incam_body_pose_equal_window_frames": int(pose_equal[start:stop].sum()),
        "source_experiment": metadata.get("source_experiment"),
        "experimental_back_support_fit": {
            key: metadata.get("experimental_back_support_fit", {}).get(key)
            for key in (
                "kind",
                "script_sha256",
                "window_half_open_frames",
                "manual_back_support_half_open_frames",
            )
        },
        "camera": {
            "kind": "frame0_root_derived_fixed_extrinsic_not_calibrated_ground_truth",
            "R_world_to_camera": rotation.tolist(),
            "t_world_to_camera": translation.tolist(),
            "K": intrinsics.tolist(),
            "frame0_all_body22_transform_max_error_m": frame0_error,
        },
        "full_timeline_fk": {
            "body22_max_error_m": max_joint_error,
            "foot_surface_max_error_m": max_foot_error,
        },
        "mask": {
            "rule": "same original ViTPose heatmap_peak >= 0.5 for both projection paths",
            "joint_names": NAMES12,
            "body22_indices": BODY12,
            "coco_indices": list(range(5, 17)),
            "reliable_samples": int(mask.sum()),
            "possible_samples": int(mask.size),
            "frames_with_reliable_samples": int(mask.any(axis=1).sum()),
            "per_frame_reliable_samples": mask.sum(axis=1).tolist(),
            "shared_mask": mask.tolist(),
        },
        "original_incam_direct_projection": direct,
        "current_world_frame0_extrinsic_projection": fixed,
        "implied_camera_rotation": {
            "formula": "R_incam_root(t) @ R_world_root(t).T relative to its frame-zero value",
            "window_deviation_deg": stats(deviation[start:stop]),
            "full_timeline_deviation_deg": stats(deviation),
            "per_frame_window_deviation_deg": deviation[start:stop].tolist(),
            "note": (
                "Root-derived inconsistency only, not measured camera motion; "
                "changed local poses cannot be reconciled by this rigid transform."
            ),
        },
        "limitations": [
            "Frame-zero transform and source gravity are estimated, not camera/ground calibration.",
            "The two paths share exactly the same detector samples; "
            "detector keypoints are not mocap ground truth.",
            "Body22 joints and COCO landmarks are approximate anatomical correspondences, "
            "not mesh surface points.",
            "Original incam stays historical evidence after source pose optimization; "
            "do not interpret it as the new fitted camera pose.",
            "No optimization, BVH fitting, data correction, "
            "or robot quality evaluation occurs in this script.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite an audit report: {args.output}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "incam": report["original_incam_direct_projection"]["error_px"],
                "world_fixed": report["current_world_frame0_extrinsic_projection"]["error_px"],
                "rotation_deviation_deg": report["implied_camera_rotation"]["window_deviation_deg"],
                "samples": report["mask"]["reliable_samples"],
                "full_timeline_fk": report["full_timeline_fk"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
