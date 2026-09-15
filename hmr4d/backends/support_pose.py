"""Opt-in fixed-camera, flat-ground SMPL-X support-pose candidate.

Runs only in the GVHMR environment. Does not change default inference. Evidence
is a support hypothesis, not a contact sensor; keep the original and compare.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform

from hmr4d.backends.foot_surface import _surface_evidence
from hmr4d.backends.portable import (
    MODEL_RELATIVE_PATH,
    _helper_sha256,
    _load_body_model,
    _sha256,
    _validated_prediction,
    _write_new_artifact,
)


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _support_mask(points, confidence, keypoints, boxes, fps):
    """A hypothesis for static support, not proof of physical contact."""
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))
    image_speed = (
        np.linalg.norm(
            np.gradient(gaussian_filter1d(keypoints[:, [15, 16], :2], 1.5, axis=0), axis=0) * fps,
            axis=-1,
        )
        / height
    )
    pixel_still = (image_speed < 0.12) & (keypoints[:, [15, 16], 2] > 0.7)
    pixel_still &= keypoints[:, [15, 16], 1] > keypoints[:, [11, 12], 1] + height * 0.15
    speed = np.linalg.norm(np.gradient(points[:, :, [0, 2]], axis=0) * fps, axis=-1)
    mask = ((confidence >= 0.8) & (speed < 0.2)) | pixel_still
    for foot in range(2):
        for start, stop in _runs(mask[:, foot]):
            if stop - start < 4:
                mask[start:stop, foot] = False
    return mask, pixel_still, height


def _support_bias(surface, mask, fps):
    """Interpolate height bias; do not impose independent feet during flight."""
    floor = float(np.nanquantile(np.where(mask, surface, np.nan), 0.1))
    bias = np.zeros_like(surface)
    for foot in range(2):
        times = np.flatnonzero(mask[:, foot])
        if not len(times):
            continue
        curve = np.interp(np.arange(len(surface)), times, surface[times, foot] - floor)
        bias[:, foot] = np.clip(gaussian_filter1d(curve, 0.08 * fps, mode="nearest"), -0.20, 0.20)
    unsupported = gaussian_filter1d((~mask.any(1)).astype(float), 0.08 * fps)[:, None]
    return floor, bias * (1 - unsupported) + bias.min(1, keepdims=True) * unsupported


def _fit(a, model, iterations, observed, fps):
    t0 = time.perf_counter()
    count = len(a["pred_w_j3d"])
    bm = model.bm
    confidence = np.stack([a["left_contact_confidence"], a["right_contact_confidence"]], 1)
    beta = torch.tensor(a["smpl_params_global.betas"][:1])
    shaped = (bm.v_template + torch.einsum("bl,vcl->bvc", beta, bm.shapedirs)[0]).detach()
    rest = (bm.J_regressor @ shaped).expand(count, -1, -1).detach()
    full = bm.pose_mean.expand(count, -1).clone()
    full[:, :3] += torch.tensor(a["smpl_params_global.global_orient"])
    full[:, 3:66] += torch.tensor(a["smpl_params_global.body_pose"])
    base_rotation = axis_angle_to_matrix(full.reshape(count, -1, 3)).detach()
    base_trans = torch.tensor(a["smpl_params_global.transl"])
    # Use all vertices selected by the same skinning mask as the surface exporter.
    ids = []
    groups = []
    for side in ([7, 10], [8, 11]):
        foot = torch.where(bm.lbs_weights[:, side].sum(-1) >= 0.5)[0]
        groups.append(torch.arange(sum(len(x) for x in ids), sum(len(x) for x in ids) + len(foot)))
        ids.append(foot)
    ids = torch.cat(ids)
    verts = shaped[ids]
    posedirs = (
        bm.posedirs.reshape(bm.posedirs.shape[0], -1, 3)[:, ids].reshape(bm.posedirs.shape[0], -1).detach()
    )
    weights = bm.lbs_weights[ids].detach()
    eye = torch.eye(3)

    def fk(rot, trans):
        joints, transforms = batch_rigid_transform(rot, rest, bm.parents)
        posed = verts + ((rot[:, 1:] - eye).flatten(1) @ posedirs).reshape(count, -1, 3)
        skin = (weights[None] @ transforms.reshape(count, -1, 16)).reshape(count, -1, 4, 4)
        homo = torch.cat((posed, torch.ones((count, len(ids), 1))), -1)
        v = (skin @ homo[..., None])[..., :3, 0] + trans[:, None]
        soles = torch.stack([v[:, g, 1].amin(1) for g in groups], 1)
        return joints[:, :22] + trans[:, None], soles

    with torch.no_grad():
        base_joints, base_surface = fk(base_rotation, base_trans)
        error = float((base_joints - torch.tensor(a["pred_w_j3d"])).abs().max())
        if error > 1e-4:
            raise ValueError(f"SMPL-X input FK mismatch: {error:g} m")
    surface = base_surface.numpy()
    points = a["pred_w_j3d"][:, [10, 11]]
    mask, pixel_still, height = _support_mask(points, confidence, *observed, fps)
    if not mask.any():
        return a, {"applied": False, "reason": "no_reliable_static_support"}
    floor, bias = _support_bias(surface, mask, fps)
    xy_target = base_joints[:, [10, 11]][:, :, [0, 2]].clone()
    for foot in range(2):
        for start, stop in _runs(mask[:, foot]):
            target = xy_target[start:stop, foot].median(0).values
            xy_target[start:stop, foot] = target
    desired = torch.tensor(surface - bias)
    rootvar = torch.nn.Parameter(torch.zeros((count, 1)))
    posevar = torch.nn.Parameter(torch.zeros((count, 6, 3)))
    leg_ids = [1, 2, 4, 5, 7, 8]
    insertion = torch.zeros((6, 55))
    insertion[torch.arange(6), leg_ids] = 1
    mask_t = torch.tensor(mask).float()
    rg = Rotation.from_rotvec(a["smpl_params_global.global_orient"]).as_matrix()
    rc = Rotation.from_rotvec(a["smpl_params_incam.global_orient"]).as_matrix()
    r_cw = torch.tensor(rc @ rg.transpose(0, 2, 1)).float()
    cam_pelvis = base_joints[:, 0] - base_trans + torch.tensor(a["smpl_params_incam.transl"])
    K = torch.tensor(a["K_fullimg"]).float()

    def project(points):
        cam = torch.einsum("tij,tkj->tki", r_cw, points - base_joints[:, :1]) + cam_pelvis[:, None]
        px = torch.einsum("tij,tkj->tki", K, cam)
        return px[..., :2] / px[..., 2:]

    base_pixels = project(base_joints).detach()

    def prediction():
        dy = 0.15 * torch.tanh(rootvar)
        angles = np.deg2rad(20) * torch.tanh(posevar)
        rot = axis_angle_to_matrix(torch.einsum("bkc,kj->bjc", angles, insertion)) @ base_rotation
        delta = torch.cat((torch.zeros_like(dy), dy, torch.zeros_like(dy)), 1)
        joints, sole = fk(rot, base_trans + delta)
        return joints, sole, rot, delta, angles

    def losses():
        joints, sole, _rot, delta, angles = prediction()
        foot_xy = joints[:, [10, 11]][:, :, [0, 2]]
        xy_reference = base_joints[:, [10, 11]][:, :, [0, 2]]
        loss = ((sole - desired) / 0.01).square().mean()
        loss += ((foot_xy - xy_reference) / 0.01).square().mean()
        loss += (((foot_xy - xy_target) / 0.01).square().sum(-1) * mask_t).mean()
        loss += 0.08 * (delta / 0.05).square().mean() + 0.08 * (angles / np.deg2rad(10)).square().mean()
        loss += 0.1 * (torch.diff(delta, n=2, dim=0) / 0.001).square().mean()
        loss += 0.1 * (torch.diff(angles, n=2, dim=0) / np.deg2rad(0.5)).square().mean()
        loss += (torch.relu(floor - sole) / 0.005).square().mean()
        loss += ((project(joints) - base_pixels) / (height * 0.02)).square().mean()
        return loss

    optimizer = torch.optim.LBFGS(
        [rootvar, posevar],
        max_iter=iterations,
        line_search_fn="strong_wolfe",
        tolerance_change=1e-8,
        tolerance_grad=1e-5,
    )

    def closure():
        optimizer.zero_grad()
        loss = losses()
        loss.backward()
        return loss

    initial = float(losses().detach())
    optimizer.step(closure)
    with torch.no_grad():
        joints, sole, rot, delta, angles = prediction()
        new = {k: v.copy() for k, v in a.items()}
        pose = matrix_to_axis_angle(rot[:, 1:22]).reshape(count, 63).numpy()
        # Copy untouched joints exactly, do not rewrite SO(3) representations.
        for jid in leg_ids:
            sl = slice((jid - 1) * 3, jid * 3)
            for prefix in ("global", "incam"):
                new[f"smpl_params_{prefix}.body_pose"][:, sl] = pose[:, sl]
        new["smpl_params_global.transl"] += delta.numpy()
        rg = Rotation.from_rotvec(a["smpl_params_global.global_orient"]).as_matrix()
        rc = Rotation.from_rotvec(a["smpl_params_incam.global_orient"]).as_matrix()
        r_cw_numpy = rc @ rg.transpose(0, 2, 1)
        new["smpl_params_incam.transl"] += np.einsum("tij,tj->ti", r_cw_numpy, delta.numpy())
        # Serialized full model is the authoritative output, not the optimizer's subset.
        all_joints = []
        for start in range(0, count, 64):
            out = model(
                **{
                    k: torch.tensor(new[f"smpl_params_global.{k}"][start : start + 64])
                    for k in ("body_pose", "betas", "global_orient", "transl")
                }
            )
            all_joints.append(out.joints[:, :22].numpy())
        new["pred_w_j3d"] = np.concatenate(all_joints)
    fullsurface, counts, error = _surface_evidence(new, model)
    new["foot_surface_y"] = fullsurface
    report = {
        "applied": True,
        "seconds": time.perf_counter() - t0,
        "loss_before": initial,
        "loss_after": float(losses().detach()),
        "max_root_shift": float(delta.abs().max()),
        "max_leg_rotation_deg": float(torch.rad2deg(angles.norm(dim=-1)).max()),
        "static_frames": mask.sum(0).tolist(),
        "max_bias": bias.max(0).tolist(),
        "floor": floor,
        "residual_height_p95": float(np.percentile(abs(sole.detach().numpy() - desired.numpy()), 95)),
        "fk_error": error,
    }
    report["observed_static_frames"] = pixel_still.sum(0).tolist()
    report["reprojection_change_p95_px"] = float(
        torch.linalg.vector_norm(project(joints) - base_pixels, dim=-1).quantile(0.95)
    )
    meta = json.loads(str(new["motiforge_video_json"]))
    meta["support_pose_refinement"] = report
    meta["foot_surface"] = {
        "schema": "smplx-foot-surface-v1",
        "foot_order": ["left", "right"],
        "up_axis": "y",
        "units": "m",
        "vertex_counts": counts,
    }
    new["motiforge_video_json"] = np.asarray(json.dumps(meta))
    new["support_pose_bias_y"] = bias
    return new, report


def _validate_inputs(arrays, metadata, keypoints, boxes):
    if metadata.get("static_camera") is not True:
        raise ValueError("Support-pose refinement requires explicit static_camera input")
    if "support_pose_refinement" in metadata:
        raise ValueError("Input was already refined; use the original portable prediction")
    if "body_pose" in metadata or "subject_scale" in metadata or "foot_surface" in metadata:
        raise ValueError("Refine the original prediction before adding other portable evidence")
    if "smplx_sequence" in metadata or any(key.startswith("smplx.") for key in arrays):
        raise ValueError("Refine before SMPL-X sequence export; existing geometry would become stale")
    count = len(arrays["pred_w_j3d"])
    if not 6 <= count <= 1800 or float(metadata["fps"]) != 30.0:
        raise ValueError("Support-pose pilot supports 6–1800 frames at 30 Hz")
    required = {"K_fullimg": (count, 3, 3)}
    for name, width in (("body_pose", 63), ("betas", 10), ("global_orient", 3), ("transl", 3)):
        required[f"smpl_params_incam.{name}"] = (count, width)
    for name in ("left_contact_confidence", "right_contact_confidence"):
        required[name] = (count,)
    for key, shape in required.items():
        value = arrays.get(key)
        if (
            value is None
            or value.shape != shape
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{key} must contain finite numeric values with shape {shape}")
    if keypoints.shape != (count, 17, 3) or boxes.shape != (count, 4):
        raise ValueError("Observation frame counts must match the normalized portable timeline")
    if not np.isfinite(keypoints).all() or not np.isfinite(boxes).all():
        raise ValueError("Observations must be finite")
    if np.any(boxes[:, 3] <= boxes[:, 1]) or np.any(boxes[:, 2] <= boxes[:, 0]):
        raise ValueError("Observation boxes must have positive width and height")
    # ViTPose stores heatmap peak scores, which can exceed one; these are not
    # calibrated probabilities. Static support channels are probabilities.
    if np.any(keypoints[..., 2] < 0):
        raise ValueError("Keypoint heatmap scores must be nonnegative")
    for values in (arrays["left_contact_confidence"], arrays["right_contact_confidence"]):
        if np.any((values < 0) | (values > 1)):
            raise ValueError("Observation and static probabilities must be in [0, 1]")
    k = arrays["K_fullimg"]
    if np.any(k[:, (0, 1), (0, 1)] <= 0) or not np.allclose(k[:, 2], [0, 0, 1]):
        raise ValueError("K_fullimg must contain valid pinhole intrinsics")
    betas = arrays["smpl_params_global.betas"]
    if not np.allclose(betas, betas[:1], atol=1e-6, rtol=0):
        raise ValueError("Support-pose refinement requires a fixed body shape")
    for name in ("body_pose", "betas"):
        if not np.allclose(
            arrays[f"smpl_params_global.{name}"], arrays[f"smpl_params_incam.{name}"], atol=1e-6, rtol=0
        ):
            raise ValueError("World and camera body shape/pose must agree before refinement")


def export_support_pose(input_path, output_path, asset_root, *, assume_flat_ground=False, iterations=80):
    """Write a new candidate from an original cache, never replace the source.

    The neighboring preprocess cache is required so no observation interpolation,
    person reassociation or frame-rate conversion occurs in this operation.
    """
    if not assume_flat_ground:
        raise ValueError("Explicit --assume-flat-ground is required; stairs/objects are unsupported")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 200:
        raise ValueError("iterations must be an integer in [1, 200]")
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().absolute()
    if input_path == output_path.resolve() or output_path.exists() or output_path.is_symlink():
        raise FileExistsError("Output must be a new path, distinct from the source")
    arrays, metadata, source_digest = _validated_prediction(input_path)
    preprocessing = input_path.parent / "native/motiforge/preprocess"
    kp_path, box_path = preprocessing / "vitpose.pt", preprocessing / "bbx.pt"
    # Trusted local GVHMR cache only, no general pickle loader.
    keypoints = torch.load(kp_path, map_location="cpu", weights_only=True).numpy()
    boxes = torch.load(box_path, map_location="cpu", weights_only=True)["bbx_xyxy"].numpy()
    _validate_inputs(arrays, metadata, keypoints, boxes)
    model_path = Path(asset_root).expanduser().resolve() / MODEL_RELATIVE_PATH
    model = _load_body_model(model_path).eval().cpu()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidate, report = _fit(arrays, model, iterations, (keypoints, boxes), float(metadata["fps"]))
    if not report.get("applied"):
        raise ValueError("No reliable support evidence; original retained, no candidate exported")
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))
    if (
        not np.isfinite(report["loss_after"])
        or report["loss_after"] > report["loss_before"] + 1e-6
        or report["reprojection_change_p95_px"] > height * 0.08
        or report["max_leg_rotation_deg"] > 25
    ):
        raise ValueError("Candidate exceeds fit/projection/pose guard; original retained")
    for value in candidate.values():
        if value.dtype.kind in "fiu" and not np.isfinite(value).all():
            raise ValueError("Nonfinite candidate; original retained")
    result_meta = json.loads(str(candidate["motiforge_video_json"]))
    report.update(
        {
            "version": "static-flat-support-pose-v1",
            "status": "candidate_requires_review",
            "assume_flat_ground": True,
            "source_artifact": str(input_path),
            "source_artifact_sha256": source_digest,
            "source_backend_revision": metadata.get("gvhmr_backend_revision"),
            "algorithm_sha256": _sha256(Path(__file__).resolve()),
            "helper_sha256": _helper_sha256(),
            "body_model_sha256": _sha256(model_path),
            "observations_sha256": {"keypoints": _sha256(kp_path), "boxes": _sha256(box_path)},
            "iterations": iterations,
            "warnings": [
                "2D stationarity does not prove contact; lifted stationary feet can be misclassified.",
                "Fixed camera and flat floor are operator assumptions, not estimated calibration.",
                "Compare slip, pose and genuine flight before adopting; no automatic replacement.",
            ],
        }
    )
    result_meta["support_pose_refinement"] = report
    result_meta["foot_surface"].update(
        {
            "body_model_sha256": report["body_model_sha256"],
            "exporter_sha256": report["algorithm_sha256"],
            "helper_sha256": report["helper_sha256"],
        }
    )
    candidate["motiforge_video_json"] = np.asarray(
        json.dumps(result_meta, ensure_ascii=False, sort_keys=True)
    )
    _write_new_artifact(output_path, candidate)
    return {"prediction": str(output_path), "frames": len(candidate["pred_w_j3d"]), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Original portable NPZ beside its native preprocessing cache"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--assume-flat-ground", action="store_true")
    parser.add_argument("--iterations", type=int, default=80)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        report = export_support_pose(
            args.input,
            args.output,
            args.asset_root,
            assume_flat_ground=args.assume_flat_ground,
            iterations=args.iterations,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"support-pose refinement failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
