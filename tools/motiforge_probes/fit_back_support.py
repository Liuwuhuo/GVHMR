"""Manual back-support/source-pose diagnostic; isolated GVHMR runtime only.

This is not a contact detector or production default. Freeze these weights before
looking at holdout results. The source first frame must visibly have grounded
feet; the support interval must visibly show supine pelvis AND upper-back support.
No BVH, robot, fitted camera, or presumed foot contact enters the objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from hmr4d.backends.portable import _load_body_model, _validated_prediction, _write_new_artifact
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from smplx.lbs import batch_rigid_transform

BODY12 = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
COCO12 = list(range(5, 17))
# Hips/knees and spine, not toes, ankles, shoulders, arms, or hands.
LOCAL_JOINTS = [1, 2, 3, 4, 5, 6, 9]
FIELDS = ("body_pose", "betas", "global_orient", "transl")
CHANGED = {
    "pred_w_j3d",
    "smpl_params_global.transl",
    "smpl_params_global.global_orient",
    "smpl_params_global.body_pose",
    "foot_surface_y",
}
LIMITS = {"root_norm_m": 0.65, "root_axis_deg": 45.0, "body_axis_deg": 20.0}
WEIGHTS = {
    "reprojection": 1.0,
    "back_contact": 4.0,
    "surface_nonpenetration": 1.0,
    "root_prior": 0.1,
    "root_orientation_prior": 0.1,
    "body_pose_prior": 0.15,
    "root_delta_acceleration": 0.1,
    "angle_delta_acceleration": 0.1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summary(value: torch.Tensor) -> dict:
    values = value.detach().cpu().numpy().reshape(-1)
    require(bool(len(values)), "Cannot report an empty metric")
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def index_at(seconds: float, fps: float) -> int:
    require(math.isfinite(seconds) and seconds >= 0, "Times must be finite nonnegative seconds")
    index = round(seconds * fps)
    require(abs(index - seconds * fps) < 1e-4, "Interval endpoints must fall on video frames")
    return index


def pick_patches(bm, shaped: torch.Tensor, rest: torch.Tensor):
    """Static material sets in subject-shaped neutral space, with fail-closed axes."""
    toe_directions = torch.stack((rest[10] - rest[7], rest[11] - rest[8]))
    require(bool((toe_directions[:, 2] > 0.015).all()), "Neutral toe-forward +Z assumption failed")
    require(bool((rest[9, 1] - rest[0, 1]) > 0.15), "Neutral torso-up +Y assumption failed")
    patches = []
    definitions = [
        ("pelvis_back", [0, 3], 0, 0.14, 0.14),
        ("chest_back", [6, 9], 9, 0.14, 0.18),
    ]
    descriptions = []
    for name, joint_ids, center_id, y_radius, x_radius in definitions:
        mask = bm.lbs_weights[:, joint_ids].sum(-1) >= 0.35
        mask &= (shaped[:, 1] - rest[center_id, 1]).abs() <= y_radius
        mask &= (shaped[:, 0] - rest[center_id, 0]).abs() <= x_radius
        candidate = torch.where(mask)[0]
        require(len(candidate) >= 20, f"Insufficient central torso vertices for {name}")
        rear_z = shaped[candidate, 2].quantile(0.15)
        patch = candidate[shaped[candidate, 2] <= rear_z]
        require(len(patch) >= 5, f"Insufficient posterior vertices for {name}")
        center = shaped[patch].mean(0)
        require(bool(center[2] < rest[center_id, 2] - 0.005), f"{name} is not posterior to joint")
        patches.append(patch)
        descriptions.append(
            {
                "name": name,
                "skin_joints": joint_ids,
                "skin_weight_min": 0.35,
                "neutral_center_xyz_m": center.tolist(),
                "neutral_joint_xyz_m": rest[center_id].tolist(),
                "rear_z_quantile": 0.15,
                "vertex_ids": patch.tolist(),
            }
        )
    require(not bool(torch.isin(patches[0], patches[1]).any()), "Back patches overlap")
    feet = [torch.where(bm.lbs_weights[:, joints].sum(-1) >= 0.5)[0] for joints in ((7, 10), (8, 11))]
    hands = [
        torch.where(bm.lbs_weights[:, joints].sum(-1) >= 0.5)[0]
        for joints in ([20, *range(25, 40)], [21, *range(40, 55)])
    ]
    require(all(len(part) >= 5 for part in feet + hands), "Incomplete foot or hand surface masks")
    # Deterministic broad mesh coverage; full relevant contact surfaces are added.
    sample = torch.arange(0, len(shaped), 8, device=shaped.device)
    subset = torch.unique(torch.cat([sample, *feet, *hands, *patches]), sorted=True)
    slots = [torch.where(torch.isin(subset, patch))[0] for patch in patches]
    return subset, slots, feet, descriptions, toe_directions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Enhanced protocol-3 portable NPZ")
    parser.add_argument("--keypoints", type=Path, required=True, help="Matching native/preprocess/vitpose.pt")
    parser.add_argument("--model", type=Path, required=True, help="Licensed SMPLX_NEUTRAL.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-start", type=float, default=12.0)
    parser.add_argument("--window-end", type=float, default=17.0)
    parser.add_argument("--support-start", type=float, default=14.0)
    parser.add_argument("--support-end", type=float, default=15.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--surface-penalty", choices=("mean", "worst-frame"), default="mean")
    args = parser.parse_args()
    require(1 <= args.iterations <= 150, "Use one fixed run of at most 150 iterations")
    require(args.model.name == "SMPLX_NEUTRAL.npz" and args.model.is_file(), "Expected SMPLX_NEUTRAL.npz")
    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to replace {args.output} or {report_path}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    device = torch.device(args.device)
    arrays, metadata, source_digest = _validated_prediction(args.input)
    require(
        not any(key.startswith("body22_") for key in arrays),
        "Use position-only input; this probe cannot preserve precomputed body22 rotations/bind evidence",
    )
    require(
        isinstance(metadata.get("foot_surface_export"), dict), "Requires enhanced foot-surface provenance"
    )
    require(
        metadata.get("static_camera") is True and not metadata.get("mirror"),
        "Requires fixed unmirrored camera",
    )
    total = len(arrays["pred_w_j3d"])
    fps = float(metadata["fps"])
    start, stop, support_first, support_last = [
        index_at(value, fps)
        for value in (args.window_start, args.window_end, args.support_start, args.support_end)
    ]
    require(
        0 <= start < support_first < support_last < stop <= total,
        "Require support strictly within fitting window",
    )
    count = stop - start
    support = slice(support_first - start, support_last - start)
    for key in ("foot_surface_y", "K_fullimg", *(f"smpl_params_incam.{name}" for name in FIELDS)):
        require(key in arrays, f"Missing {key}")
        require(bool(np.isfinite(arrays[key]).all()), f"Nonfinite {key}")
    require(arrays["foot_surface_y"].shape == (total, 2), "Invalid foot_surface_y")
    require(arrays["K_fullimg"].shape == (total, 3, 3), "Invalid K_fullimg")
    np.testing.assert_array_equal(
        arrays["K_fullimg"], np.broadcast_to(arrays["K_fullimg"][:1], (total, 3, 3))
    )
    for field in ("body_pose", "betas"):
        np.testing.assert_array_equal(
            arrays[f"smpl_params_global.{field}"], arrays[f"smpl_params_incam.{field}"]
        )
    beta_np = arrays["smpl_params_global.betas"][:1]
    np.testing.assert_array_equal(arrays["smpl_params_global.betas"], np.broadcast_to(beta_np, (total, 10)))
    observed_all = torch.load(args.keypoints, map_location="cpu", weights_only=True)
    require(isinstance(observed_all, torch.Tensor), "VitPose must be a plain keypoint tensor")
    require(observed_all.shape == (total, 17, 3), "VitPose must match the complete portable timeline")
    require(bool(torch.isfinite(observed_all).all()), "Nonfinite keypoint evidence")
    observed = observed_all[start:stop, COCO12].to(device=device, dtype=torch.float32)
    points_2d, raw_confidence = observed[..., :2], observed[..., 2]
    # ViTPose exports heatmap maxima, not probabilities: valid peaks can exceed 1.
    # Bound only the objective weights; leave the original observations untouched.
    confidence = raw_confidence.clamp(0, 1).square() * (raw_confidence >= 0.5)
    require(
        float(confidence.sum()) > 0 and float(confidence[support].sum()) > 0,
        "No reliable 2D support observations",
    )

    def tensor(value):
        return torch.as_tensor(value, device=device, dtype=torch.float32)

    model = _load_body_model(args.model).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bm = model.bm
    beta = tensor(beta_np)
    shaped = (bm.v_template + torch.einsum("bl,vcl->bvc", beta, bm.shapedirs)[0]).detach()
    rest_single = (bm.J_regressor @ shaped).detach()
    rest = rest_single.expand(count, -1, -1)
    indices, patch_slots, foot_ids, patch_description, toe_directions = pick_patches(bm, shaped, rest_single)
    shape_small = shaped[indices]
    pose_dirs = (
        bm.posedirs.reshape(bm.posedirs.shape[0], -1, 3)[:, indices]
        .reshape(bm.posedirs.shape[0], -1)
        .detach()
    )
    weights = bm.lbs_weights[indices].detach()
    identity = torch.eye(3, device=device)
    mean_pose = bm.pose_mean
    full_pose = mean_pose.expand(count, -1).clone()
    full_pose[:, :3] += tensor(arrays["smpl_params_global.global_orient"][start:stop])
    full_pose[:, 3:66] += tensor(arrays["smpl_params_global.body_pose"][start:stop])
    baseline_rotations = axis_angle_to_matrix(full_pose.reshape(count, -1, 3)).detach()
    baseline_trans = tensor(arrays["smpl_params_global.transl"][start:stop])

    # Source-global -> camera gauge estimated ONCE from frame 0 root transforms.
    # SMPL-X rotates about the shaped root joint, not the origin.
    rw = axis_angle_to_matrix(tensor(arrays["smpl_params_global.global_orient"][0]))
    rc = axis_angle_to_matrix(tensor(arrays["smpl_params_incam.global_orient"][0]))
    camera_r = (rc @ rw.T).detach()
    world_root0 = tensor(arrays["pred_w_j3d"][0, 0])
    camera_root0 = rest_single[0] + tensor(arrays["smpl_params_incam.transl"][0])
    camera_t = (camera_root0 - camera_r @ world_root0).detach()
    intrinsics = tensor(arrays["K_fullimg"][0])
    plane_y = tensor(float(arrays["foot_surface_y"][0].min())).detach()

    def params_from(payload, first, last, prefix="smpl_params_global"):
        return {field: tensor(payload[f"{prefix}.{field}"][first:last]) for field in FIELDS}

    def forward(rotations, translation):
        joints, transforms = batch_rigid_transform(rotations, rest, bm.parents)
        pose_feature = (rotations[:, 1:] - identity).flatten(1)
        posed = shape_small + (pose_feature @ pose_dirs).reshape(count, -1, 3)
        skin = (weights[None] @ transforms.reshape(count, -1, 16)).reshape(count, -1, 4, 4)
        homogeneous = torch.cat((posed, torch.ones((count, len(indices), 1), device=device)), -1)
        vertices = (skin @ homogeneous[..., None])[..., :3, 0] + translation[:, None]
        patches = torch.stack([vertices[:, slots].mean(1) for slots in patch_slots], 1)
        return joints[:, :22] + translation[:, None], patches, vertices

    with torch.no_grad():
        base_joints, _base_patches, base_vertices = forward(baseline_rotations, baseline_trans)
        initial_fk_error = float((base_joints - tensor(arrays["pred_w_j3d"][start:stop])).abs().max())
        require(initial_fk_error < 1e-4, f"Input world FK mismatch: {initial_fk_error}")
        subset_error = 0.0
        for offset in sorted({0, count // 2, count - 1}):
            full = model(**params_from(arrays, start + offset, start + offset + 1))
            subset_error = max(
                subset_error, float((full.vertices[0, indices] - base_vertices[offset]).abs().max())
            )
        require(subset_error < 1e-4, f"Subset/full skinning mismatch: {subset_error}")
        initial_world = model(**params_from(arrays, 0, 1))
        initial_camera = model(**params_from(arrays, 0, 1, "smpl_params_incam"))
        camera_fk_error = float(
            ((initial_world.joints[:, :22] @ camera_r.T + camera_t) - initial_camera.joints[:, :22])
            .abs()
            .max()
        )
        require(camera_fk_error < 1e-4, f"Frame-0 camera transform mismatch: {camera_fk_error}")
        foot0 = torch.stack([initial_world.vertices[:, ids, 1].amin(1) for ids in foot_ids], -1)
        require(
            float((foot0 - tensor(arrays["foot_surface_y"][:1])).abs().max()) < 1e-4,
            "Input floor surface gauge mismatch",
        )

    # Half-second sin^2 ramps preserve zero parameter delta at both endpoints.
    ramp_count = min(max(2, round(fps * 0.5)), count // 2)
    ramp = torch.sin(torch.linspace(0, math.pi / 2, ramp_count, device=device)).square()
    envelope = torch.ones(count, device=device)
    envelope[:ramp_count], envelope[-ramp_count:] = ramp, ramp.flip(0)
    envelope = envelope[:, None]
    root_variable = torch.nn.Parameter(torch.zeros((count, 3), device=device))
    root_angle_variable = torch.nn.Parameter(torch.zeros((count, 3), device=device))
    local_variable = torch.nn.Parameter(torch.zeros((count, len(LOCAL_JOINTS), 3), device=device))
    variables = [root_variable, root_angle_variable, local_variable]
    insertion = torch.zeros((len(LOCAL_JOINTS) + 1, 55), device=device)
    insertion[torch.arange(len(LOCAL_JOINTS) + 1, device=device), [0, *LOCAL_JOINTS]] = 1

    def prediction():
        root_delta = (
            LIMITS["root_norm_m"]
            * root_variable
            / torch.sqrt(1 + root_variable.square().sum(-1, keepdim=True))
            * envelope
        )
        root_angle = math.radians(LIMITS["root_axis_deg"]) * torch.tanh(root_angle_variable) * envelope
        local_angle = math.radians(LIMITS["body_axis_deg"]) * torch.tanh(local_variable) * envelope[:, None]
        angles = torch.cat((root_angle[:, None], local_angle), dim=1)
        tangent = torch.einsum("bkc,kj->bjc", angles, insertion)
        rotations = axis_angle_to_matrix(tangent) @ baseline_rotations
        joints, patches, vertices = forward(rotations, baseline_trans + root_delta)
        return joints, patches, vertices, rotations, root_delta, angles

    def project(joints):
        camera = joints[:, BODY12] @ camera_r.T + camera_t
        require(
            bool((camera[..., 2] > 0.1).all()), "Optimization moved a projected joint behind the fixed camera"
        )
        pixels = camera @ intrinsics.T
        return pixels[..., :2] / pixels[..., 2:]

    def losses():
        joints, patches, vertices, _rot, delta, angles = prediction()
        residual = (project(joints) - points_2d) / 5.0
        robust = functional.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=1.0)
        pad_delta = functional.pad(delta.T, (2, 2)).T
        pad_angle = functional.pad(angles.permute(1, 2, 0), (2, 2)).permute(2, 0, 1)
        depth = torch.relu(plane_y - vertices[..., 1]) / 0.005
        surface_penalty = (
            depth.amax(dim=1).square().mean()
            if args.surface_penalty == "worst-frame"
            else depth.square().mean()
        )
        parts = {
            "reprojection": (robust.sum(-1) * confidence).sum() / confidence.sum(),
            "back_contact": ((patches[support, :, 1] - plane_y) / 0.015).square().mean(),
            "surface_nonpenetration": surface_penalty,
            "root_prior": (delta / 0.25).square().mean(),
            "root_orientation_prior": (angles[:, 0] / math.radians(20)).square().mean(),
            "body_pose_prior": (angles[:, 1:] / math.radians(10)).square().mean(),
            "root_delta_acceleration": (torch.diff(pad_delta, n=2, dim=0) / 0.003).square().mean(),
            "angle_delta_acceleration": (torch.diff(pad_angle, n=2, dim=0) / math.radians(2)).square().mean(),
        }
        weighted = {name: value * WEIGHTS[name] for name, value in parts.items()}
        return sum(weighted.values()), weighted

    def metrics():
        with torch.no_grad():
            joints, patches, vertices, _rot, delta, angles = prediction()
            residual = torch.linalg.vector_norm(project(joints) - points_2d, dim=-1)
            return {
                "reprojection_px": summary(residual[confidence > 0]),
                "support_reprojection_px": summary(residual[support][confidence[support] > 0]),
                "reprojection_valid_samples": int((confidence > 0).sum()),
                "root_delta_norm_m": summary(delta.norm(dim=-1)),
                "root_delta_step_norm_m": summary(torch.diff(delta, dim=0).norm(dim=-1)),
                "root_rotation_delta_norm_deg": summary(torch.rad2deg(angles[:, 0].norm(dim=-1))),
                "root_rotation_delta_axis_max_deg": float(torch.rad2deg(angles[:, 0].abs()).max()),
                "body_rotation_delta_norm_deg": summary(torch.rad2deg(angles[:, 1:].norm(dim=-1))),
                "body_rotation_delta_axis_max_deg": float(torch.rad2deg(angles[:, 1:].abs()).max()),
                "back_patch_clearance_support_m": {
                    name: summary(patches[support, i, 1] - plane_y)
                    for i, name in enumerate(("pelvis_back", "chest_back"))
                },
                "sampled_surface_penetration_m": summary(torch.relu(plane_y - vertices[..., 1])),
                "losses": {name: float(value) for name, value in losses()[1].items()},
            }

    def full_mesh_metrics(payload):
        penetration, per_frame, sampled_error = [], [], 0.0
        with torch.no_grad():
            _j, _p, subset_vertices, _rot, _d, _a = prediction()
            for first in range(start, stop, 16):
                last = min(first + 16, stop)
                full = model(**params_from(payload, first, last))
                depth = torch.relu(plane_y - full.vertices[..., 1])
                penetration.append(depth.cpu())
                per_frame.append(depth.amax(1).cpu())
                sampled_error = max(
                    sampled_error,
                    float(
                        (full.vertices[:, indices] - subset_vertices[first - start : last - start])
                        .abs()
                        .max()
                    ),
                )
        return {
            "full_mesh_penetration_m": summary(torch.cat(penetration)),
            "full_mesh_penetration_per_frame_max_m": summary(torch.cat(per_frame)),
            "subset_full_max_error_m": sampled_error,
        }

    before = metrics()
    before.update(full_mesh_metrics(arrays))
    require(before["subset_full_max_error_m"] < 1e-4, "Initial full-window subset mismatch")
    print("before", json.dumps(before), flush=True)
    optimizer = torch.optim.LBFGS(
        variables,
        lr=1.0,
        max_iter=args.iterations,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-6,
        tolerance_change=1e-8,
    )
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        loss, _parts = losses()
        require(bool(torch.isfinite(loss)), "Nonfinite fitting objective")
        loss.backward()
        require(
            all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in variables),
            "Nonfinite gradient",
        )
        calls += 1
        if calls % 25 == 0:
            print("objective_evaluations", calls, "loss", float(loss.detach()), flush=True)
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - started
    after = metrics()
    result = {key: value.copy() for key, value in arrays.items()}
    with torch.no_grad():
        joints, _patches, _vertices, rotations, delta, _angles = prediction()
        result["smpl_params_global.transl"][start:stop] = (baseline_trans + delta).cpu().numpy()
        new_pose = matrix_to_axis_angle(rotations).reshape(count, -1) - mean_pose
        result["smpl_params_global.global_orient"][start:stop] = new_pose[:, :3].cpu().numpy()
        for joint_id in LOCAL_JOINTS:
            columns = slice((joint_id - 1) * 3, joint_id * 3)
            result["smpl_params_global.body_pose"][start:stop, columns] = (
                new_pose[:, joint_id * 3 : (joint_id + 1) * 3].cpu().numpy()
            )
        result["pred_w_j3d"][start:stop] = joints.cpu().numpy()
        serial_error = 0.0
        for first in range(start, stop, 16):
            last = min(first + 16, stop)
            full = model(**params_from(result, first, last))
            serial_error = max(
                serial_error,
                float((full.joints[:, :22] - tensor(result["pred_w_j3d"][first:last])).abs().max()),
            )
            result["pred_w_j3d"][first:last] = full.joints[:, :22].cpu().numpy()
            result["foot_surface_y"][first:last] = (
                torch.stack([full.vertices[:, ids, 1].amin(1) for ids in foot_ids], -1).cpu().numpy()
            )
        require(serial_error < 1e-4, f"Serialized FK mismatch: {serial_error}")
        frozen = np.flatnonzero(envelope[:, 0].cpu().numpy() == 0) + start
        for key in CHANGED:
            result[key][frozen] = arrays[key][frozen]
            np.testing.assert_array_equal(result[key][:start], arrays[key][:start])
            np.testing.assert_array_equal(result[key][stop:], arrays[key][stop:])
        for key in arrays.keys() - CHANGED - {"motiforge_video_json"}:
            np.testing.assert_array_equal(result[key], arrays[key])
        # The complete timeline, including untouched regions, must still agree
        # with serialized SMPL-X FK and the identical full foot vertex masks.
        full_fk_error, full_foot_error = 0.0, 0.0
        for first in range(0, total, 16):
            last = min(first + 16, total)
            full = model(**params_from(result, first, last))
            full_fk_error = max(
                full_fk_error,
                float((full.joints[:, :22] - tensor(result["pred_w_j3d"][first:last])).abs().max()),
            )
            foot = torch.stack([full.vertices[:, ids, 1].amin(1) for ids in foot_ids], -1)
            full_foot_error = max(
                full_foot_error, float((foot - tensor(result["foot_surface_y"][first:last])).abs().max())
            )
        require(
            full_fk_error < 1e-4 and full_foot_error < 1e-4,
            f"Full timeline FK/foot mismatch: {full_fk_error}, {full_foot_error}",
        )
    after.update(full_mesh_metrics(result))
    require(after["subset_full_max_error_m"] < 1e-4, "Serialized subset/full mesh mismatch")
    for key, value in result.items():
        if value.dtype.kind in "fciu":
            require(bool(np.isfinite(value).all()), f"Nonfinite serialized {key}")
    require(after["root_delta_norm_m"]["max"] <= LIMITS["root_norm_m"] + 1e-6, "Root norm bound exceeded")
    report = {
        "kind": "manual_pelvis_back_and_chest_back_video_fit_not_production",
        "source": str(args.input.resolve()),
        "source_sha256": source_digest,
        "script_sha256": sha256(Path(__file__)),
        "keypoints": str(args.keypoints.resolve()),
        "keypoints_sha256": sha256(args.keypoints),
        "keypoint_weight_rule": "clip(heatmap_peak,0,1)^2 * (heatmap_peak>=0.5)",
        "keypoint_heatmap_peak_min": float(observed_all[..., 2].min()),
        "keypoint_heatmap_peak_max": float(observed_all[..., 2].max()),
        "model_sha256": sha256(args.model),
        "original_video_sha256": metadata["source_sha256"],
        "device": str(device),
        "window_half_open_frames": [start, stop],
        "manual_back_support_half_open_frames": [support_first, support_last],
        "fps": fps,
        "optimized_body22_joints": LOCAL_JOINTS,
        "camera_kind": "frame0_root_extrinsic_estimate_fixed_not_calibrated_ground_truth",
        "camera_R": camera_r.tolist(),
        "camera_t": camera_t.tolist(),
        "camera_K": intrinsics.tolist(),
        "contact_plane_y_m": float(plane_y),
        "contact_plane_kind": "original_first_frame_minimum_foot_surface_source_gauge",
        "neutral_toe_directions_xyz": toe_directions.tolist(),
        "patches": patch_description,
        "nonpenetration_vertex_count": len(indices),
        "nonpenetration_vertex_ids": indices.tolist(),
        "weights": WEIGHTS,
        "surface_penalty": args.surface_penalty,
        "limits": LIMITS,
        "envelope_ramp_frames": ramp_count,
        "frozen_window_frames": frozen.tolist(),
        "input_fk_max_error_m": initial_fk_error,
        "input_subset_full_max_error_m": subset_error,
        "frame0_camera_fk_max_error_m": camera_fk_error,
        "serialized_window_fk_max_error_m": serial_error,
        "serialized_full_timeline_fk_max_error_m": full_fk_error,
        "serialized_full_timeline_foot_max_error_m": full_foot_error,
        "outside_window_exact": True,
        "untouched_arrays_exact": True,
        "optimizer_iterations_requested": args.iterations,
        "objective_evaluations": calls,
        "optimization_seconds": elapsed,
        "before": before,
        "after": after,
        "limitations": [
            "Manual back-support labels, not automatic inference or measured 3D contact.",
            "First-frame camera and foot floor are estimated gauges, not camera/ground calibration.",
            "COCO12 observes shoulders/elbows/wrists/hips/knees/ankles, not back surface or toe contact.",
            "Back patches are shaped-mesh surrogates; clothing and contact deformation are not modeled.",
            "Optimization samples the mesh; independent before/after audit evaluates the full mesh.",
            "Incam parameters, confidence, and floor_correction_y remain ORIGINAL inference evidence; "
            "they do not describe the refined world pose or total transform.",
            "No robot quality claim: requires unchanged retarget, independent reference and holdout tests.",
        ],
    }
    metadata["experimental_back_support_fit"] = report
    metadata["foot_surface_export"]["refined_geometry"] = (
        "recomputed_from_serialized_world_parameters_in_experimental_back_support_fit"
    )
    result["motiforge_video_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    _write_new_artifact(args.output, result)
    # Reopen the actual NPZ bytes before publishing the companion report.
    with np.load(args.output, allow_pickle=False) as saved:
        require(set(saved.files) == set(result), "Serialized keys changed")
        for key in result:
            np.testing.assert_array_equal(saved[key], result[key])
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("after", json.dumps(after), flush=True)
    print("output", args.output, "seconds", elapsed, flush=True)


if __name__ == "__main__":
    main()
