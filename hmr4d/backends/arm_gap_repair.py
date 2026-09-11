"""Reconstruct only short, observation-identified gaps in local arm rotations.

This is a source hypothesis, not recovered ground truth. The backend must
recompute native FK and pass the unchanged temporal acceptance guard before
publishing it. No world-coordinate tracks, roots, shapes or scores are edited.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

from hmr4d.backends.observation_stability import JOINTS, _validated_observations, runs
from hmr4d.backends.observation_stability import POLICY as OBSERVATION_POLICY

POLICY = {
    "schema": "source-short-arm-so3-gap-v1",
    "maximum_gap_seconds": OBSERVATION_POLICY["maximum_gap_seconds"],
    "anchor_confidence": OBSERVATION_POLICY["anchor_confidence"],
    "anchor_frames_per_side": 3,
    "body22_joints": {"left": [16, 18, 20], "right": [17, 19, 21]},
    "coco_joints": {"left": [5, 7, 9], "right": [6, 8, 10]},
    "rotation": "SO3 RotationSpline through six original neighboring anchor poses",
    "visibility": "all three arm joints >= anchor score; inside closed crop and optional half-open image",
    "preserved": "all original poses outside identified gaps; other joints and observation scores remain exact",
    "interpretation": "bounded interpolation hypothesis, not observed or recovered ground truth",
}


def interpolate_arm_gaps(original, repairs, keypoints, crops, fps, *, image_size=None):
    """Return a new (T,63) SMPL pose and explicit accepted/skipped evidence.

    ``repairs`` contains the detector's effective COCO elbow/wrist runs using
    half-open frame bounds. Only each merged run itself can change. Immediately
    adjacent three-frame anchors must all be observed reliably; this helper
    never searches farther away or absorbs weak neighboring motion into a gap.
    """
    original = np.asarray(original)
    if (original.ndim != 2 or original.shape[1:] != (63,) or not len(original)
            or original.dtype.kind != "f" or not np.isfinite(original).all()):
        raise ValueError("original body_pose must be finite floating point with nonempty shape (T,63)")
    frames = len(original)
    crops = np.asarray(crops)
    if (crops.shape != (frames, 3) or crops.dtype.kind not in "fiu"
            or not np.isfinite(crops).all()):
        raise ValueError("crops must be finite shape (T,3) center-x, center-y, size")
    keypoints, scale, fps = _validated_observations(keypoints, crops[:, 2], fps)
    if len(keypoints) != frames:
        raise ValueError("body_pose and observations must share the original frame timeline")
    size = None
    if image_size is not None:
        size = np.asarray(image_size)
        if (size.shape != (2,) or size.dtype.kind not in "fiu" or not np.isfinite(size).all()
                or np.any(size <= 0)):
            raise ValueError("image_size must be finite positive (width,height)")
    if not isinstance(repairs, (list, tuple)):
        raise ValueError("repairs must contain COCO elbow/wrist run dictionaries")  # noqa: TRY004
    known = {"left": np.zeros(frames, dtype=bool), "right": np.zeros(frames, dtype=bool)}
    names = {name for _, name in JOINTS.values()}
    for repair in repairs:
        if not isinstance(repair, dict) or repair.get("joint") not in names:
            raise ValueError("repair must identify a COCO elbow or wrist")
        start, stop = repair.get("start_frame"), repair.get("stop_frame_exclusive")
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                   for value in (start, stop)):
            raise ValueError("repair frame bounds must be integers")
        if not 0 <= start < stop <= frames:
            raise ValueError("repair frame bounds must lie on the original timeline")
        known[repair["joint"].split("_")[0]][start:stop] = True

    lower = crops[:, None, :2] - scale[:, None, None] / 2
    upper = crops[:, None, :2] + scale[:, None, None] / 2
    inside_crop = np.all((keypoints[..., :2] >= lower) & (keypoints[..., :2] <= upper), axis=-1)
    inside_image = (np.ones(keypoints.shape[:2], dtype=bool) if size is None else
                    np.all((keypoints[..., :2] >= 0) & (keypoints[..., :2] < size), axis=-1))
    score_ok = keypoints[..., 2] >= POLICY["anchor_confidence"]
    output = original.copy()
    before, after = original.reshape(frames, 21, 3), output.reshape(frames, 21, 3)
    windows = []
    for side, body_joints in POLICY["body22_joints"].items():
        coco = POLICY["coco_joints"][side]
        for start, stop in runs(known[side]):
            start, stop = int(start), int(stop)
            reasons = []
            if (stop - start) / fps > POLICY["maximum_gap_seconds"]:
                reasons.append("gap_too_long")
            enough = start >= 3 and stop + 3 <= frames
            anchors = (np.r_[np.arange(start - 3, start), np.arange(stop, stop + 3)]
                       if enough else np.array([], dtype=int))
            if not enough:
                reasons.append("insufficient_three_frame_anchors")
            else:
                if known[side][anchors].any():
                    reasons.append("anchors_overlap_known_gap")
                if not score_ok[anchors][:, coco].all():
                    reasons.append("unreliable_arm_anchor_scores")
                if not inside_crop[anchors][:, coco].all():
                    reasons.append("arm_anchors_outside_crop")
                if not inside_image[anchors][:, coco].all():
                    reasons.append("arm_anchors_outside_image")
            entry = {
                "side": side, "start_frame": start, "stop_frame_exclusive": stop,
                "duration_seconds": (stop - start) / fps,
                "accepted": not reasons, "skipped": bool(reasons), "reasons": reasons,
                "anchor_frames": anchors.tolist(), "body22_joints": body_joints.copy(),
                "coco_joints": coco.copy(), "changed_frames": [], "changed_pose_frames": 0,
            }
            if enough:
                entry["anchor_confidence"] = keypoints[anchors][:, coco, 2].tolist()
                entry["anchor_xy"] = keypoints[anchors][:, coco, :2].tolist()
            if reasons:
                windows.append(entry)
                continue
            selected = np.arange(start, stop)
            times = (anchors - (start - 1)) / fps
            for body_joint in body_joints:
                joint = body_joint - 1
                poses = before[anchors, joint]
                if np.array_equal(poses, np.broadcast_to(poses[0], poses.shape)):
                    # Exact constant anchors also preserve noncanonical rotvecs.
                    after[selected, joint] = poses[0]
                else:
                    spline = RotationSpline(times, Rotation.from_rotvec(poses))
                    after[selected, joint] = spline((selected - (start - 1)) / fps).as_rotvec()
            local_indices = np.asarray(body_joints) - 1
            changed = selected[np.any(after[selected][:, local_indices] != before[selected][:, local_indices], axis=(1, 2))]
            entry["changed_frames"] = changed.tolist()
            entry["changed_pose_frames"] = len(changed)
            windows.append(entry)
    return output, {
        "policy": POLICY.copy(), "windows": windows,
        "changed_pose_frames": int(np.count_nonzero(np.any(output != original, axis=1))),
    }
