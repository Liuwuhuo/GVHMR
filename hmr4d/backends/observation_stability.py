"""NumPy-only observation evidence and conservative, bounded keypoint repair.

Confidence is the native pose-detector score, not a calibrated probability.
Audits are diagnostic evidence about one selected track, not ground truth or a
quality gate. They never cut, smooth, or otherwise change their input arrays.
Only ``detect_and_repair`` proposes changed coordinates, and it preserves all
confidence values and the exact research short-cross-side numerical rule.
"""

from __future__ import annotations

import numpy as np

POLICY = {
    "schema": "research-short-cross-side-keypoint-candidate-v1",
    "maximum_gap_seconds": 1 / 6,
    "low_confidence": 0.7,
    "opposite_reliable_confidence": 0.8,
    "minimum_opposite_confidence_margin": 0.2,
    "maximum_cross_side_distance_crop_fraction": 0.1,
    "minimum_interpolation_residual_crop_fraction": 0.08,
    "anchor_confidence": 0.7,
    "confidence_policy": "preserve_original_exactly; never promote interpolation",
    "coordinates_policy": "linear bridge between immediate reliable endpoints",
    "scope": "COCO17 left/right elbow and wrist; all frames scanned",
}
JOINTS = {7: (8, "left_elbow"), 8: (7, "right_elbow"), 9: (10, "left_wrist"), 10: (9, "right_wrist")}


def runs(mask):
    """Yield contiguous true intervals as [start, stop), without per-frame output."""
    boundaries = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True)


def _validated_fps(fps):
    value = np.asarray(fps)
    if value.ndim != 0 or value.dtype.kind not in "fiu" or not np.isfinite(value) or value <= 0:
        raise ValueError("fps must be a finite positive scalar")
    return float(value)


def _validated_observations(keypoints, crop_scale, fps):
    original = np.asarray(keypoints)
    scale = np.asarray(crop_scale)
    if original.ndim != 3 or original.shape[1:] != (17, 3) or not len(original):
        raise ValueError("keypoints must have nonempty shape (T, 17, 3)")
    if not np.issubdtype(original.dtype, np.floating) or not np.isfinite(original).all():
        raise ValueError("keypoints must be finite floating point values")
    if (
        scale.shape != (len(original),) or scale.dtype.kind not in "fiu"
        or not np.isfinite(scale).all() or np.any(scale <= 0)
    ):
        raise ValueError("crop_scale must contain one finite positive pixel size per frame")
    return original, scale, _validated_fps(fps)


def detect_and_repair(keypoints, crop_scale, fps):
    """Return (copy, accepted, rejected), preserving the established repair rule.

    Only low-confidence COCO17 wrists/elbows collapsed toward a reliable opposite
    joint can be bridged, over at most 1/6 second between immediate reliable
    anchors. The bridge is a hypothesis, not a newly observed position. Scores and
    all other coordinates remain byte-for-byte equivalent numerical values.
    """
    original, crop_scale, fps = _validated_observations(keypoints, crop_scale, fps)
    repaired = original.copy()
    accepted, rejected = [], []
    for joint, (opposite, name) in JOINTS.items():
        for start, stop in runs(original[:, joint, 2] < POLICY["low_confidence"]):
            gap = int(stop - start)
            entry = {
                "joint": name,
                "joint_index": joint,
                "start_frame": int(start),
                "stop_frame_exclusive": int(stop),
                "duration_seconds": gap / fps,
                "confidence": original[start:stop, joint, 2].tolist(),
            }
            reasons = []
            if gap / fps > POLICY["maximum_gap_seconds"]:
                reasons.append("gap_too_long")
            if start == 0 or stop == len(original):
                reasons.append("missing_two_sided_anchors")
            else:
                anchor_conf = original[[start - 1, stop], joint, 2]
                entry["anchor_confidence"] = anchor_conf.tolist()
                if np.any(anchor_conf < POLICY["anchor_confidence"]):
                    reasons.append("unreliable_anchors")
                alpha = np.arange(1, gap + 1)[:, None] / (gap + 1)
                bridge = (1 - alpha) * original[start - 1, joint, :2] + alpha * original[stop, joint, :2]
                observed = original[start:stop, joint]
                other = original[start:stop, opposite]
                cross = np.linalg.norm(observed[:, :2] - other[:, :2], axis=1) / crop_scale[start:stop]
                residual = np.linalg.norm(observed[:, :2] - bridge, axis=1) / crop_scale[start:stop]
                entry["cross_side_distance_crop_fraction"] = cross.tolist()
                entry["bridge_residual_crop_fraction"] = residual.tolist()
                if np.any(other[:, 2] < POLICY["opposite_reliable_confidence"]):
                    reasons.append("opposite_not_reliable")
                if np.any(other[:, 2] - observed[:, 2] < POLICY["minimum_opposite_confidence_margin"]):
                    reasons.append("no_confidence_asymmetry")
                if np.any(cross > POLICY["maximum_cross_side_distance_crop_fraction"]):
                    reasons.append("not_a_cross_side_collapse")
                if np.any(residual < POLICY["minimum_interpolation_residual_crop_fraction"]):
                    reasons.append("insufficient_temporal_discontinuity")
            entry["rejection_reasons"] = reasons
            if reasons:
                rejected.append(entry)
            else:
                entry["original_xy"] = original[start:stop, joint, :2].tolist()
                repaired[start:stop, joint, :2] = bridge
                entry["candidate_xy"] = repaired[start:stop, joint, :2].tolist()
                accepted.append(entry)
    assert np.array_equal(repaired[..., 2], original[..., 2])
    return repaired, accepted, rejected


def _report(kind, frames, fps, thresholds):
    return {
        "schema_version": 1,
        "kind": kind,
        "scope": "single_selected_track",
        "frame_count": int(frames),
        "fps": fps,
        "warnings": [],
        "events": [],
        "thresholds": thresholds,
        "interpretation": "Diagnostic evidence, not ground truth; no automatic rejection, cuts or smoothing.",
    }


def _add_events(report, mask, code, reason, **details):
    found = False
    for start, stop in runs(mask):
        found = True
        report["events"].append({
            "code": code,
            "start_frame": int(start),
            "end_frame": int(stop - 1),
            "start_seconds": float(start / report["fps"]),
            "end_seconds": float((stop - 1) / report["fps"]),
            "duration_seconds": float((stop - start) / report["fps"]),
            "reasons": [reason],
            **details,
        })
    if found and code not in report["warnings"]:
        report["warnings"].append(code)


def audit_observations(keypoints, crop_scale, fps, *, image_size=None, boxes=None):
    """Audit raw COCO17 [T,17,(x,y,score)] evidence without modifying it.

    ``crop_scale`` is one positive pixel size per frame. Optional ``image_size``
    is (width, height); ``boxes`` is [T,4] selected-track xyxy in pixels. Boxes may
    extend beyond the image but must have positive width/height. Native scores may
    exceed one; they are neither clipped nor interpreted as probabilities.
    """
    original, _, fps = _validated_observations(keypoints, crop_scale, fps)
    frames = len(original)
    if image_size is not None:
        size = np.asarray(image_size)
        if size.shape != (2,) or size.dtype.kind not in "fiu" or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("image_size must be finite positive (width, height)")
    if boxes is not None:
        boxes = np.asarray(boxes)
        if (
            boxes.shape != (frames, 4) or boxes.dtype.kind not in "fiu" or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2])
        ):
            raise ValueError("boxes must be finite shape (T, 4) xyxy with positive width and height")

    report = _report("observation_audit", frames, fps, {
        "body_joint_indices": list(range(5, 17)),
        "visible_score_strictly_greater_than": 0.5,
        "low_body_visible_count_less_than": 8,
        "upper_limb_low_score_less_than": POLICY["low_confidence"],
        "image_edge_margin_fraction": 0.02,
        "score_semantics": "native detector score, not calibrated probability; values above 1 are preserved",
    })
    body = original[:, 5:17]
    visible = body[..., 2] > 0.5
    visible_count = np.count_nonzero(visible, axis=1)
    low = visible_count < 8
    _add_events(report, visible_count == 0, "missing_body_observations", "No body joint has native score > 0.5.")
    _add_events(report, low, "low_body_visibility", "Fewer than 8 of 12 body joints have native score > 0.5.")
    boundary = np.zeros(frames, dtype=bool)
    for start, stop in runs(low):
        if start == 0 or stop == frames:
            boundary[start:stop] = True
    _add_events(
        report, boundary, "low_body_visibility_at_clip_boundary",
        "A low-visibility run reaches the first or last frame; a full-body observation is not established there.",
    )
    for joint, (_, name) in JOINTS.items():
        _add_events(
            report, original[:, joint, 2] < POLICY["low_confidence"], "low_upper_limb_confidence",
            "Native wrist/elbow score is below 0.7; this alone does not establish an erroneous position.",
            joint=name, joint_index=joint,
        )
    if image_size is not None:
        outside = np.any((body[..., :2] < 0) | (body[..., :2] >= size), axis=-1)
        _add_events(
            report, np.any(outside & visible, axis=1), "visible_body_keypoint_outside_image",
            "At least one body keypoint with score > 0.5 lies outside the image.",
        )
        if boxes is not None:
            margin = size * report["thresholds"]["image_edge_margin_fraction"]
            at_edge = np.any(boxes[:, :2] <= margin, axis=1) | np.any(boxes[:, 2:] >= size - margin, axis=1)
            _add_events(
                report, at_edge, "selected_box_at_image_boundary",
                "Selected-track box is within 2% of an image boundary or beyond it; possible truncation, not proof.",
            )
    report["summary"] = {
        "minimum_visible_body_joint_count": int(visible_count.min()),
        "missing_body_frame_count": int(np.count_nonzero(visible_count == 0)),
        "low_body_visibility_frame_count": int(np.count_nonzero(low)),
        "boundary_low_visibility_frame_count": int(np.count_nonzero(boundary)),
        "native_score_min": float(original[..., 2].min()),
        "native_score_max": float(original[..., 2].max()),
        "image_bounds_checked": image_size is not None,
        "box_boundary_checked": image_size is not None and boxes is not None,
    }
    report["events"].sort(key=lambda event: (event["start_frame"], event["end_frame"], event["code"]))
    return report


def audit_world_motion(joints, fps):
    """Audit finite [T,22,3] metre, Y-up motion; never alter fast or other motion.

    A candidate needs fast incoming AND outgoing steps, a sharp direction reversal,
    large midpoint residual, AND prominence over nearby midpoint residuals. High
    velocity alone is never a warning. An event can still be genuine fast motion;
    these indicators are for inspection, not rejection or automatic correction.
    """
    original = np.asarray(joints)
    if (
        original.ndim != 3 or original.shape[1:] != (22, 3) or not len(original)
        or original.dtype.kind not in "fiu" or not np.isfinite(original).all()
    ):
        raise ValueError("joints must have nonempty finite shape (T, 22, 3) in metres, Y-up")
    fps = _validated_fps(fps)
    thresholds = {
        "minimum_incoming_and_outgoing_speed_mps": 3.0,
        "maximum_incoming_outgoing_direction_cosine": -0.5,
        "minimum_midpoint_residual_m": 0.08,
        "minimum_local_median_residual_ratio": 4.0,
        "local_median_half_window_seconds": 0.25,
        "minimum_local_median_half_window_frames": 2,
        "rule": "All four conditions must hold for the same interior joint/frame; speed alone is never a warning.",
        "midpoint_residual": "Distance from the current position to the midpoint of immediate neighbours.",
        "local_baseline": "Per-joint median midpoint residual in the inclusive +/- half-window, including the candidate.",
    }
    frames = len(original)
    report = _report("human_temporal_audit", frames, fps, thresholds)
    report.update({"units": "m", "up_axis": "y"})
    # Float64 arithmetic prevents finite float32 inputs overflowing during norm
    # calculations; the original dtype, arrays and coordinate values stay intact.
    positions = original.astype(np.float64, copy=False)
    step = np.diff(positions, axis=0)
    speed = np.linalg.norm(step, axis=-1) * fps
    suspicious = np.zeros((frames, 22), dtype=bool)
    if frames >= 3:
        incoming, outgoing = step[:-1], step[1:]
        denominator = np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1)
        cosine = np.divide(
            np.sum(incoming * outgoing, axis=-1), denominator,
            out=np.ones_like(denominator), where=denominator > 0,
        )
        residual = np.linalg.norm((incoming - outgoing) / 2, axis=-1)
        candidate = (
            (np.minimum(speed[:-1], speed[1:]) >= thresholds["minimum_incoming_and_outgoing_speed_mps"])
            & (cosine <= thresholds["maximum_incoming_outgoing_direction_cosine"])
            & (residual >= thresholds["minimum_midpoint_residual_m"])
        )
        half_window = max(2, min(frames, round(fps * thresholds["local_median_half_window_seconds"])))
        for frame in np.flatnonzero(np.any(candidate, axis=1)):
            baseline = np.median(residual[max(0, frame - half_window):frame + half_window + 1], axis=0)
            suspicious[frame + 1] = candidate[frame] & (
                residual[frame] >= thresholds["minimum_local_median_residual_ratio"] * baseline
            )
    for joint in range(22):
        _add_events(
            report, suspicious[:, joint],
            "temporal_root_reversal_spike" if joint == 0 else "temporal_joint_reversal_spike",
            "Sharp fast reversal with a large locally prominent midpoint residual; genuine motion remains possible.",
            joint_index=joint,
        )
    report["summary"] = {
        "temporal_window_available": frames >= 3,
        "suspicious_joint_frame_count": int(np.count_nonzero(suspicious)),
        "suspicious_frame_count": int(np.count_nonzero(np.any(suspicious, axis=1))),
        "maximum_joint_speed_mps": float(speed.max()) if len(speed) else 0.0,
        "maximum_root_speed_mps": float(speed[:, 0].max()) if len(speed) else 0.0,
        "motion_modified": False,
    }
    report["events"].sort(key=lambda event: (event["start_frame"], event["end_frame"], event["joint_index"]))
    return report
