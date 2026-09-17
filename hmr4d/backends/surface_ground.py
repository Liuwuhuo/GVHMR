"""Flat-ground validation and vertical correction of the selected SMPL-X source.

Runs in the source runtime, before portable geometry is generated. No robot,
IK, pose fitting or camera-space modification belongs here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hmr4d.backends.foot_surface import SURFACE_SCHEMA, _surface_evidence
from hmr4d.backends.portable import validate_prediction_arrays

SCHEMA = "foot-surface-ground-v2"
CALIBRATED_SCHEMA = "foot-surface-ground-v3"
GROUNDED_SCHEMA = "foot-surface-ground-v4"
MAX_SPEED_MPS = 0.20


def surface_correction(surface, previous, fps, *, reference_offset=0.0):
    """Return the TOTAL downward correction relative to the raw prediction.

    After an explicit constant reference offset, only prevent penetration.
    Never consume static probabilities as contact/support/flight evidence.
    The ceiling limits the TOTAL correction, including the prior camera stage.
    """
    from hmr4d.backends.motiforge import _lipschitz_minorant

    surface = np.asarray(surface, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if (surface.ndim != 2 or surface.shape[1] != 2 or not len(surface)
            or previous.shape != (len(surface),) or not np.isfinite(surface).all()
            or not np.isfinite(previous).all() or not np.isfinite(fps) or fps <= 0
            or not np.isscalar(reference_offset) or not np.isfinite(reference_offset)):
        raise ValueError("Invalid surface-ground geometry/time contract")
    raw_low = surface.min(axis=1) + previous
    return _lipschitz_minorant(np.minimum(previous + reference_offset, raw_low), MAX_SPEED_MPS / fps)


def reference_calibration(surface, fps, window):
    """Calibrate one known-grounded interval, not automatically detected support.

    The caller explicitly asserts that at least one foot is on the same flat
    floor in this interval. A fixed median removes outliers, not height drift.
    """
    if (not isinstance(window, (tuple, list)) or len(window) != 2
            or not np.isfinite(window).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Ground reference requires finite start/end seconds and positive FPS")
    start, stop = map(float, window)
    if start < 0 or stop <= start or stop > len(surface) / fps + 1e-9:
        raise ValueError("Ground reference must be inside the complete source timeline")
    first, last = int(np.ceil(start * fps - 1e-9)), int(np.ceil(stop * fps - 1e-9))
    if last - first < max(3, int(np.ceil(.1 * fps))):
        raise ValueError("Ground reference needs at least 0.1 seconds and 3 samples")
    lower = np.asarray(surface[first:last], dtype=np.float64).min(axis=1)
    if not np.isfinite(lower).all():
        raise ValueError("Ground reference contains nonfinite surfaces")
    offset = float(np.median(lower))
    return offset, {
        "assumption": "operator-confirmed-at-least-one-foot-on-flat-ground",
        "window_seconds": [start, stop], "frame_window_half_open": [first, last],
        "estimator": "median-of-lower-foot-surface", "offset_m": offset,
        "surface_p95_minus_p05_m": float(np.percentile(lower, 95) - np.percentile(lower, 5)),
    }


def add_surface_ground(arrays, model, *, enabled, model_digest, reference_window=None,
                       assume_grounded=False):
    """Add measured feet and, if enabled, replace the effective ground correction.

    All pose/shape/XY and incam parameters stay untouched. Old ground diagnostics
    and world translations are retained; the caller rebuilds geometry afterwards.
    """
    if type(assume_grounded) is not bool:
        raise ValueError("assume_grounded must be boolean")
    if assume_grounded and (not enabled or reference_window is not None):
        raise ValueError("assume_grounded requires ground enabled and no fixed reference window")
    metadata = validate_prediction_arrays(arrays)
    if "surface_ground" in metadata:
        raise ValueError("Surface ground already evaluated; refusing to correct twice")
    if any(k.startswith("smplx.") or k.startswith("body22_") for k in arrays):
        raise ValueError("Rebuild derived geometry before applying source ground correction")
    surface, counts, error = _surface_evidence(arrays, model)
    previous = np.asarray(arrays.get("floor_correction_y", np.zeros(len(surface))), dtype=np.float64)
    if reference_window is not None and not enabled:
        raise ValueError("A ground reference cannot be used while ground correction is disabled")
    offset, calibration = 0.0, None
    if reference_window is not None:
        offset, calibration = reference_calibration(surface, metadata["fps"], reference_window)
    total = surface_correction(surface, previous, metadata["fps"], reference_offset=offset)
    if assume_grounded:
        # Strict contact and an arbitrary speed bound cannot both be guaranteed.
        # This is a declared no-flight assumption, not detected support. Project
        # exactly and report speed instead of smoothing hovering back into it.
        total = previous + surface.astype(np.float64).min(axis=1)
    result = dict(arrays)
    corrected = surface.copy()
    if enabled:
        delta = previous - total
        result["ground_input_transl"] = arrays["smpl_params_global.transl"].copy()
        result["ground_input_floor_correction_y"] = previous.copy()
        result["ground_input_foot_surface_y"] = surface.copy()
        for key in ("pred_w_j3d", "smpl_params_global.transl"):
            values = arrays[key].copy()
            values[..., 1] += delta.reshape((-1,) + (1,) * (values.ndim - 2))
            result[key] = values
        corrected = (surface.astype(np.float64) + delta[:, None]).astype(np.float32)
        result["floor_correction_y"] = total.astype(np.float32)
    else:
        total = previous
    low = corrected.min(axis=1)
    errors, warnings = [], []
    if low.min() < -.002:
        errors.append("foot_surface_below_ground")
    unknown_high = low > .03
    if unknown_high.any():
        warnings.append("elevated_foot_surfaces_unclassified")
    correction_speed = float(np.max(np.abs(np.diff(total)), initial=0) * metadata["fps"])
    if assume_grounded and correction_speed > MAX_SPEED_MPS:
        warnings.append("grounded_projection_fast_height_change")
    if calibration is not None:
        calibration["max_nonpenetration_adjustment_m"] = float(np.max(np.abs(total - previous - offset)))
        if calibration["surface_p95_minus_p05_m"] > .05:
            warnings.append("ground_reference_height_spread")
        if calibration["max_nonpenetration_adjustment_m"] > .1:
            warnings.append("large_post_calibration_adjustment")
    report = {
        "schema": GROUNDED_SCHEMA if assume_grounded else (CALIBRATED_SCHEMA if calibration is not None else SCHEMA),
        "enabled": bool(enabled), "plane_height_m": 0.0,
        "policy": "assume-grounded" if assume_grounded else (
            "fixed-reference-plus-nonpenetration" if calibration is not None else "nonpenetration-only"),
        "support_anchor_enabled": False,
        "correction_speed_limit_mps": None if assume_grounded else MAX_SPEED_MPS,
        "correction_speed_max_mps": correction_speed,
        "max_added_shift_m": float(np.max(np.abs(previous - total))),
        "min_surface_before_m": float(surface.min()), "min_surface_after_m": float(low.min()),
        "unresolved_elevated_frames": int(unknown_high.sum()),
        "body22_max_error_m": error, "errors": errors, "warnings": warnings,
        "status": "reject" if errors else ("review" if warnings else "pass"),
    }
    if calibration is not None:
        report["reference_calibration"] = calibration
    if assume_grounded:
        report["assumption"] = "operator-declared-at-least-one-foot-on-flat-ground-every-frame"
        report["max_contact_residual_m"] = float(np.max(np.abs(low)))
        report["removes_real_flight"] = True
    metadata["surface_ground"] = report
    metadata["foot_surface"] = {
        "schema": SURFACE_SCHEMA, "foot_order": ["left", "right"], "up_axis": "y",
        "units": "m", "body_model_sha256": model_digest, "vertex_counts": counts,
    }
    result["foot_surface_y"] = corrected
    result["motiforge_video_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return result


def export_ground(input_path, output_path, asset_root, *, backend_id, reference_window=None,
                  assume_grounded=False):
    """Reprocess the selected cached source, without inference or overwriting it."""
    from hmr4d.backends.portable import (
        MODEL_RELATIVE_PATH, _load_body_model, _sha256, _validated_prediction, _write_new_artifact,
    )
    from hmr4d.backends.smplx_sequence import add_sequence

    input_path, output_path = Path(input_path).resolve(), Path(output_path).absolute()
    if input_path == output_path.resolve() or output_path.exists() or output_path.is_symlink():
        raise ValueError("Ground export requires a new output path")
    arrays, metadata, digest = _validated_prediction(input_path)
    if assume_grounded and reference_window is not None:
        raise ValueError("assume_grounded and fixed reference window are mutually exclusive")
    if reference_window is None and not assume_grounded and not metadata.get("ground_stabilization", {}).get("enabled", False):
        raise ValueError("Ground export requires an existing enabled flat-ground source policy")
    prior_surface = metadata.get("surface_ground")
    if prior_surface is not None:
        # Fresh inference already carries v2 nonpenetration. Explicit calibration
        # must replace that TOTAL correction, not stack another offset on top.
        allowed = {SCHEMA, CALIBRATED_SCHEMA} if assume_grounded else {SCHEMA}
        if ((reference_window is None and not assume_grounded) or prior_surface.get("schema") not in allowed
                or "floor_correction_y" not in arrays):
            raise ValueError("Surface ground already evaluated; use the original artifact")
        metadata.pop("surface_ground")
    # Derived geometry must be regenerated, never adjusted independently of FK.
    arrays = {k: v for k, v in arrays.items() if not k.startswith(("smplx.", "body22_"))}
    for key in ("smplx_sequence", "body_pose", "body_pose_export", "foot_surface", "foot_surface_export"):
        metadata.pop(key, None)
    arrays.pop("foot_surface_y", None)
    metadata["surface_ground_input"] = {"path": str(input_path), "sha256": digest,
                                        "backend_revision": metadata.get("gvhmr_backend_revision")}
    if prior_surface is not None:
        metadata["surface_ground_input"]["surface_ground"] = prior_surface
    metadata["gvhmr_backend_revision"] = backend_id
    # Cached inputs may already contain the DISABLED legacy contact-floor
    # correction. Recover its pre-backend world first, then use today's camera
    # path. Otherwise an export would retain the very support shift it disables.
    from hmr4d.backends.motiforge import _stabilize_prediction_ground

    if "floor_correction_y" not in arrays and metadata.get("ground_stabilization", {}).get("applied"):
        raise ValueError("Cannot undo legacy source correction without floor_correction_y")
    previous = arrays.get("floor_correction_y", np.zeros(len(arrays["pred_w_j3d"])))
    world = arrays["pred_w_j3d"].copy()
    root = arrays["smpl_params_global.transl"].copy()
    world[..., 1] += previous[:, None]
    root[:, 1] += previous
    incam = {k.removeprefix("smpl_params_incam."): v for k, v in arrays.items()
             if k.startswith("smpl_params_incam.")}
    world, root, correction, diagnostics = _stabilize_prediction_ground(
        world, root, None, global_orient=arrays["smpl_params_global.global_orient"],
        incam=incam, fps=metadata["fps"], enabled=True,
        static_camera=bool(metadata.get("static_camera", False)),
    )
    metadata["surface_ground_input"]["ground_stabilization"] = metadata.get("ground_stabilization", {})
    metadata["ground_stabilization"] = diagnostics
    metadata["assume_grounded"] = assume_grounded
    arrays.update(pred_w_j3d=world, floor_correction_y=correction)
    arrays["smpl_params_global.transl"] = root
    arrays["motiforge_video_json"] = np.asarray(json.dumps(metadata))
    model_path = Path(asset_root) / MODEL_RELATIVE_PATH
    model = _load_body_model(model_path)
    result = add_surface_ground(arrays, model, enabled=True, model_digest=_sha256(model_path),
                                reference_window=reference_window, assume_grounded=assume_grounded)
    result = add_sequence(result, model_path, model_factory=lambda _: model)
    _write_new_artifact(output_path, result)
    return {"prediction": str(output_path), **json.loads(str(result["motiforge_video_json"]))["surface_ground"]}
