"""Bounded video-conditioned diagnostic in the isolated GVHMR environment.

Not an automatic contact detector or production source default. The two support
intervals and four protected airborne frames were checked in the original video.
This fits actual COCO observations, preserves shape and segment lengths, and uses
SMPL-X skinned forefoot patches, not an ankle forced to an arbitrary floor zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from smplx.lbs import batch_rigid_transform

from hmr4d.backends.portable import _load_body_model, _write_new_artifact


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="frame0_root_extrinsic")
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--nonpenetration-weight", type=float, default=0.0)
    parser.add_argument("--freeze-toe", action="store_true")
    parser.add_argument("--plane", choices=("window", "first-frame"), default="window")
    parser.add_argument("--local-support-only", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    arrays = dict(np.load(args.input, allow_pickle=False))
    geometry = dict(np.load(args.geometry, allow_pickle=False))
    meta = json.loads(arrays["motiforge_video_json"].item())
    assert meta["fps"] == 30.0 and meta["static_camera"] and not meta["mirror"]
    assert (
        meta["source_sha256"]
        == "80fa6378001d92df61459105ef59cb2043d5c4c141ce71a383d7bd0cf0827f1f"
    )
    assert arrays["pred_w_j3d"].shape == (2834, 22, 3)
    start, stop = 450, 660
    count = stop - start
    selected = np.flatnonzero(
        (geometry["frames"] >= start) & (geometry["frames"] < stop)
    )
    assert np.array_equal(geometry["frames"][selected], np.arange(start, stop))
    np.testing.assert_array_equal(
        geometry["world_body22_yup"][selected], arrays["pred_w_j3d"][start:stop]
    )
    observed = torch.tensor(geometry["keypoints_coco17"][selected], dtype=torch.float32)
    bi, ci = (
        geometry["body22_projection_indices"],
        geometry["coco17_projection_indices"],
    )
    kp, confidence = observed[:, ci, :2], observed[:, ci, 2]
    confidence = confidence.square() * (confidence >= 0.5)
    camera_r = torch.tensor(geometry[args.camera + ".R"], dtype=torch.float32)
    camera_t = torch.tensor(geometry[args.camera + ".t"], dtype=torch.float32)
    intrinsics = torch.tensor(geometry["K"], dtype=torch.float32)
    model = _load_body_model(args.model).eval().cpu()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bm = model.bm
    beta = torch.tensor(arrays["smpl_params_global.betas"][:1])
    assert np.array_equal(
        arrays["smpl_params_global.betas"], np.broadcast_to(beta.numpy(), (2834, 10))
    )
    shaped = bm.v_template + torch.einsum("bl,vcl->bvc", beta, bm.shapedirs)[0]
    rest = (bm.J_regressor @ shaped).expand(count, -1, -1).detach()
    full_pose = bm.pose_mean.expand(count, -1).clone()
    full_pose[:, :3] += torch.tensor(
        arrays["smpl_params_global.global_orient"][start:stop]
    )
    full_pose[:, 3:66] += torch.tensor(
        arrays["smpl_params_global.body_pose"][start:stop]
    )
    baseline_rot = axis_angle_to_matrix(full_pose.reshape(count, -1, 3)).detach()
    baseline_trans = torch.tensor(arrays["smpl_params_global.transl"][start:stop])

    # One small forefoot material patch per side. Index selection is performed
    # once in the subject's shaped neutral mesh, never per-frame nearest points.
    patch_ids, foot_ids = [], []
    for side in ((7, 10), (8, 11)):
        ids = torch.where(bm.lbs_weights[:, list(side)].sum(-1) >= 0.5)[0]
        foot_ids.append(ids)
        vertices = shaped[ids]
        low = vertices[:, 1] <= vertices[:, 1].quantile(0.25)
        front = vertices[:, 2] >= vertices[:, 2].quantile(0.6)
        ids = ids[low & front]
        assert len(ids) >= 3
        patch_ids.append(ids)
    ids = torch.cat(foot_ids)
    patch_slots = [torch.where(torch.isin(ids, patch))[0] for patch in patch_ids]
    shaped_small = shaped[ids].detach()
    posedirs = (
        bm.posedirs.reshape(bm.posedirs.shape[0], -1, 3)[:, ids]
        .reshape(bm.posedirs.shape[0], -1)
        .detach()
    )
    weights = bm.lbs_weights[ids].detach()
    eye = torch.eye(3)

    def forward(rot, trans):
        joints, transforms = batch_rigid_transform(rot, rest, bm.parents)
        features = (rot[:, 1:] - eye).flatten(1)
        posed = shaped_small + (features @ posedirs).reshape(count, -1, 3)
        skin = (weights[None] @ transforms.reshape(count, -1, 16)).reshape(
            count, -1, 4, 4
        )
        homo = torch.cat((posed, torch.ones((count, len(ids), 1))), -1)
        vertices = (skin @ homo[..., None])[..., :3, 0] + trans[:, None]
        points = torch.stack([vertices[:, slots].mean(1) for slots in patch_slots], 1)
        return joints[:, :22] + trans[:, None], points, vertices

    with torch.no_grad():
        base_joints, base_points, base_vertices = forward(baseline_rot, baseline_trans)
        expected = torch.tensor(arrays["pred_w_j3d"][start:stop])
        fk_error = float((base_joints - expected).abs().max())
        assert fk_error < 1e-4, fk_error
        # Independent full-model validation of the actual subset skinning.
        params = {
            field: torch.tensor(
                arrays[f"smpl_params_global.{field}"][start : start + 3]
            )
            for field in ("body_pose", "betas", "global_orient", "transl")
        }
        mesh_error = float(
            (model(**params).vertices[:, ids] - base_vertices[:3]).abs().max()
        )
        assert mesh_error < 1e-4, mesh_error

    windows = [(561 - start, 579 - start, 1), (594 - start, 601 - start, 0)]
    plane_y = (
        torch.cat([base_points[a:b, side, 1] for a, b, side in windows])
        .median()
        .detach()
    )
    if args.plane == "first-frame":
        # Match the source gauge used by existing first-frame target alignment.
        # Valid here only because the first video frame has grounded feet.
        plane_y = torch.tensor(float(arrays["foot_surface_y"][0].min()))
    envelope = np.ones(count, dtype=np.float32)
    ramp = np.sin(np.linspace(0, np.pi / 2, 16)) ** 2
    envelope[:16], envelope[-16:] = ramp, ramp[::-1]
    # Protect independently observed real hops; soften the transition to them.
    for a, b in ((498, 500), (553, 555)):
        for f in range(a - 4, b + 4):
            distance = max(a - f, f - (b - 1), 0)
            envelope[f - start] = min(
                envelope[f - start], np.sin(min(distance / 4, 1) * np.pi / 2) ** 2
            )
    if args.local_support_only:
        local = np.zeros(count, dtype=np.float32)
        for a, b, _ in windows:
            local[a:b] = 1.0
            for margin in range(1, 7):
                value = np.cos(margin * np.pi / 12) ** 2
                for frame in (a - margin, b - 1 + margin):
                    local[frame] = max(local[frame], value if margin < 6 else 0.0)
        envelope *= local
    envelope = torch.tensor(envelope)[:, None]
    root_variable = torch.nn.Parameter(torch.zeros((count, 3)))
    leg_ids = [1, 2, 4, 5, 7, 8, 10, 11]
    if args.freeze_toe:
        # Toe local rotations cannot be reconstructed from body22 positions.
        # Do not spend invisible rotational DOFs to satisfy the mesh residual.
        leg_ids = leg_ids[:-2]
    pose_variable = torch.nn.Parameter(
        torch.zeros((count, len(leg_ids), 3)), requires_grad=args.pose
    )
    variables = [root_variable] + ([pose_variable] if args.pose else [])
    insertion = torch.zeros((len(leg_ids), 55))
    insertion[torch.arange(len(leg_ids)), leg_ids] = 1

    def prediction():
        delta = 0.15 * torch.tanh(root_variable) * envelope
        angle = np.deg2rad(15) * torch.tanh(pose_variable) * envelope[:, None]
        tangent = torch.einsum("bkc,kj->bjc", angle, insertion)
        rotations = axis_angle_to_matrix(tangent) @ baseline_rot
        joints, points, vertices = forward(rotations, baseline_trans + delta)
        return joints, points, rotations, delta, angle, vertices

    def project(joints):
        camera = joints[:, bi] @ camera_r.T + camera_t
        assert torch.all(camera[..., 2] > 0.1)
        pixels = camera @ intrinsics.T
        return pixels[..., :2] / pixels[..., 2:]

    def losses():
        joints, points, _rotations, delta, angle, vertices = prediction()
        residual = (project(joints) - kp) / 5.0
        huber = torch.nn.functional.smooth_l1_loss(
            residual, torch.zeros_like(residual), reduction="none", beta=1.0
        )
        reprojection = (huber.sum(-1) * confidence).sum() / confidence.sum()
        contact = []
        for a, b, side in windows:
            p = points[a:b, side]
            anchor = torch.stack((p[:, 0].mean(), plane_y, p[:, 2].mean()))
            contact.append(((p - anchor) / 0.015).square().mean())
        pad_delta = torch.nn.functional.pad(delta.T, (2, 2)).T
        pad_angle = torch.nn.functional.pad(angle.permute(1, 2, 0), (2, 2)).permute(
            2, 0, 1
        )
        parts = {
            "reprojection": reprojection,
            "contact": args.contact_weight * torch.stack(contact).mean(),
            "foot_nonpenetration": args.nonpenetration_weight
            * (torch.relu(plane_y - vertices[..., 1]) / 0.005).square().mean(),
            "root_prior": 0.15 * (delta / 0.08).square().mean(),
            "pose_prior": 0.15 * (angle / np.deg2rad(10)).square().mean(),
            "root_accel_delta": 0.1
            * (torch.diff(pad_delta, n=2, dim=0) / 0.003).square().mean(),
            "pose_accel_delta": 0.1
            * (torch.diff(pad_angle, n=2, dim=0) / np.deg2rad(2)).square().mean(),
        }
        return sum(parts.values()), parts

    def metrics():
        with torch.no_grad():
            joints, points, _rotations, delta, angle, vertices = prediction()
            errors = torch.linalg.vector_norm(project(joints) - kp, dim=-1)[
                confidence > 0
            ].numpy()
            return {
                "reprojection_mean_px": float(errors.mean()),
                "reprojection_p95_px": float(np.percentile(errors, 95)),
                "root_delta_max_m": float(delta.norm(dim=-1).max()),
                "root_delta_step_max_m": float(
                    torch.diff(delta, dim=0).norm(dim=-1).max()
                ),
                "pose_delta_max_deg": float(torch.rad2deg(angle.norm(dim=-1)).max()),
                "foot_mesh_penetration_max_m": float(
                    torch.relu(plane_y - vertices[..., 1]).max()
                ),
                "support": [
                    {
                        "frames": [a + start, b + start],
                        "side": side,
                        "patch_xy_diameter_mm": float(
                            torch.cdist(
                                points[a:b, side][:, [0, 2]],
                                points[a:b, side][:, [0, 2]],
                            ).max()
                            * 1000
                        ),
                        "patch_y_span_mm": float(
                            (points[a:b, side, 1].max() - points[a:b, side, 1].min())
                            * 1000
                        ),
                    }
                    for a, b, side in windows
                ],
                "losses": {k: float(v) for k, v in losses()[1].items()},
            }

    before = metrics()
    print("before", json.dumps(before), flush=True)
    optimizer = torch.optim.LBFGS(
        variables,
        max_iter=args.iterations,
        lr=1.0,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-6,
        tolerance_change=1e-8,
    )
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        loss, _parts = losses()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite fitting objective")
        loss.backward()
        calls += 1
        if calls % 25 == 0:
            print("evaluation", calls, "loss", float(loss), flush=True)
        return loss

    t0 = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - t0
    after = metrics()
    print("after", json.dumps(after), flush=True)
    with torch.no_grad():
        joints, _points, rotations, delta, _angle, _vertices = prediction()
        result = {k: v.copy() for k, v in arrays.items()}
        result["pred_w_j3d"][start:stop] = joints.numpy()
        result["smpl_params_global.transl"][start:stop] += delta.numpy()
        if args.pose:
            new_pose = (
                matrix_to_axis_angle(rotations[:, 1:22]).reshape(count, 63).numpy()
            )
            for jid in leg_ids:
                columns = slice(3 * (jid - 1), 3 * jid)
                result["smpl_params_global.body_pose"][start:stop, columns] = new_pose[
                    :, columns
                ]
        # Full-model recomputation validates the serialized pose (no stale FK or
        # surface evidence), rather than only trusting the differentiable copy.
        masks = [
            bm.lbs_weights[:, list(side)].sum(-1) >= 0.5 for side in ((7, 10), (8, 11))
        ]
        final_error = 0.0
        for first in range(start, stop, 32):
            last = min(first + 32, stop)
            params = {
                field: torch.tensor(result[f"smpl_params_global.{field}"][first:last])
                for field in ("body_pose", "betas", "global_orient", "transl")
            }
            full = model(**params)
            final_error = max(
                final_error,
                float(
                    np.abs(
                        full.joints[:, :22].numpy() - result["pred_w_j3d"][first:last]
                    ).max()
                ),
            )
            result["pred_w_j3d"][first:last] = full.joints[:, :22].numpy()
            result["foot_surface_y"][first:last] = torch.stack(
                [full.vertices[:, mask, 1].amin(1) for mask in masks], -1
            ).numpy()
        assert final_error < 1e-4, final_error
        # Protected frames and untouched boundaries are restored bit-for-bit.
        frozen = np.flatnonzero(envelope[:, 0].numpy() == 0) + start
        changed = {
            "pred_w_j3d",
            "smpl_params_global.transl",
            "smpl_params_global.body_pose",
            "foot_surface_y",
        }
        for key in changed:
            result[key][frozen] = arrays[key][frozen]
            assert np.array_equal(result[key][:start], arrays[key][:start])
            assert np.array_equal(result[key][stop:], arrays[key][stop:])
        for key in arrays.keys() - changed - {"motiforge_video_json"}:
            np.testing.assert_array_equal(result[key], arrays[key])
    report = {
        "kind": "manual_support_video_reprojection_window_experiment_not_production",
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_video_sha256": meta["source_sha256"],
        "window_half_open": [start, stop],
        "camera": args.camera,
        "camera_R": camera_r.tolist(),
        "camera_t": camera_t.tolist(),
        "contact_weight": args.contact_weight,
        "optimize_leg_pose": args.pose,
        "nonpenetration_weight": args.nonpenetration_weight,
        "optimized_body22_joints": leg_ids if args.pose else [],
        "forefoot_vertex_indices_left_right": [p.tolist() for p in patch_ids],
        "contact_plane_y_source_gauge_m": float(plane_y),
        "contact_plane_initialization": args.plane,
        "local_support_only": args.local_support_only,
        "fk_max_error_m": fk_error,
        "mesh_subset_max_error_m": mesh_error,
        "serialized_fk_max_error_m": final_error,
        "protected_airborne_frames": [498, 499, 553, 554],
        "seconds": elapsed,
        "objective_evaluations": calls,
        "before": before,
        "after": after,
        "limitations": [
            "Manual support intervals: not an automatic estimator.",
            "COCO17 does not measure toes; forefoot patch is a model surrogate, not observed 3D contact.",
            "Fixed camera extrinsics are estimated, not calibrated ground truth; source floor gauge is not physical zero.",
            "Incam predictions, confidence and floor_correction_y retained as original inference evidence; only world pose is refined.",
            "No guarantee of robot quality: must rerun full original engine/pipeline and fixed-reference evaluation.",
        ],
    }
    meta["experimental_video_window_fit"] = report
    meta["foot_surface_export"]["refined_geometry"] = (
        "recomputed_from_serialized_world_parameters_in_experimental_video_window_fit"
    )
    result["motiforge_video_json"] = np.asarray(
        json.dumps(meta, ensure_ascii=False, sort_keys=True)
    )
    _write_new_artifact(args.output, result)
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print("output", args.output, "seconds", elapsed, flush=True)


if __name__ == "__main__":
    main()
