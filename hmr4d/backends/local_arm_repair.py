"""Localize an observation-repair hypothesis in SMPL's rotation space.

The temporal model may change the whole body after a local 2D edit. Only the
affected arm's local rotations are eligible here; root, shape, legs and the
other arm remain the original prediction. The caller must recompute native FK
and apply the existing temporal acceptance guard before publishing a candidate.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

POLICY = {
    "schema": "source-local-arm-so3-v1",
    "correction_gains": {"shoulder": 1.0, "elbow": 0.5, "wrist": 0.5},
    "taper_seconds": 0.25,
    "body22_joints": {"left": [16, 18, 20], "right": [17, 19, 21]},
    "rotation": "R_original * Exp(weight * Log(R_original^-1 * R_candidate))",
    "window": "quintic C2 taper; merge touching hits; smooth union of remaining tapers",
    "preserved": "all unselected local rotations, root, translation, shape and original contacts",
}


def localize_arm_pose(original, candidate, repairs, fps):
    """Return a new finite (T,63) local pose and its bounded-update evidence.

    Interpolation is on SO(3), never on world-space joint coordinates. Scores
    and effective COCO runs were validated by the observation detector. This
    helper additionally requires two-sided anchors and a common pose timeline.
    """
    original, candidate = np.asarray(original), np.asarray(candidate)
    if original.ndim != 2 or original.shape[1:] != (63,) or not len(original):
        raise ValueError("original body_pose must have nonempty shape (T,63)")
    for name, value in (("original", original), ("candidate", candidate)):
        if value.shape != original.shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{name} body_pose must be finite floating point on the original (T,63) timeline")
    rate = np.asarray(fps)
    if rate.ndim != 0 or rate.dtype.kind not in "fiu" or not np.isfinite(rate) or rate <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(repairs, (list, tuple)):
        raise ValueError("repairs must be effective COCO elbow/wrist run dictionaries")  # noqa: TRY004
    frames = len(original)
    intervals = {"left": [], "right": []}
    for repair in repairs:
        if not isinstance(repair, dict) or repair.get("joint") not in {
            "left_elbow", "left_wrist", "right_elbow", "right_wrist",
        }:
            raise ValueError("repair must identify a COCO elbow or wrist")
        start, stop = repair.get("start_frame"), repair.get("stop_frame_exclusive")
        if not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in (start, stop)):
            raise ValueError("repair bounds must be integers")
        if not 0 < start < stop < frames:
            raise ValueError("repair requires two-sided anchors on the original timeline")
        intervals[repair["joint"].split("_")[0]].append((int(start), int(stop)))

    output = original.copy()
    a, b, out = (value.reshape(frames, 21, 3) for value in (original, candidate, output))
    time = np.arange(frames) / float(rate)
    windows, changed = [], {}
    for side, joints in POLICY["body22_joints"].items():
        merged = []
        for start, stop in sorted(set(intervals[side])):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        weight = np.zeros(frames)
        for start, stop in merged:
            distance = np.maximum(start / float(rate) - time, time - (stop - 1) / float(rate))
            phase = np.clip(1 - distance / POLICY["taper_seconds"], 0, 1)
            ramp = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
            weight = 1 - (1 - weight) * (1 - ramp)
            windows.append({"side": side, "start_frame": start, "stop_frame_exclusive": stop})
        rows = np.flatnonzero(weight > 0)
        for role, body_joint in zip(("shoulder", "elbow", "wrist"), joints, strict=True):
            joint = body_joint - 1
            # Keep zero-delta values exact, including noncanonical axis-angle
            # representations, instead of needlessly round-tripping rotations.
            selected = rows[np.any(a[rows, joint] != b[rows, joint], axis=1)]
            if not len(selected):
                continue
            first = Rotation.from_rotvec(a[selected, joint])
            delta = (first.inv() * Rotation.from_rotvec(b[selected, joint])).as_rotvec()
            amount = weight[selected, None] * POLICY["correction_gains"][role]
            out[selected, joint] = (first * Rotation.from_rotvec(delta * amount)).as_rotvec()
        indices = np.asarray(joints) - 1
        changed[side] = int(np.count_nonzero(np.any(out[:, indices] != a[:, indices], axis=(1, 2))))
    return output, {"policy": POLICY.copy(), "windows": windows, "changed_pose_frames": changed}
