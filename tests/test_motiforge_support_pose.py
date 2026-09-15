"""CPU contract/evidence tests; no video, checkpoints, or body model required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hmr4d.backends.support_pose import (
    _support_bias,
    _support_mask,
    _validate_inputs,
    export_support_pose,
)


def fixture(frames=60):
    arrays = {"pred_w_j3d": np.zeros((frames, 22, 3), dtype=np.float32)}
    for space in ("global", "incam"):
        for field, width in (("body_pose", 63), ("betas", 10), ("global_orient", 3), ("transl", 3)):
            arrays[f"smpl_params_{space}.{field}"] = np.zeros((frames, width), dtype=np.float32)
    arrays["K_fullimg"] = np.tile(np.diag([500.0, 500.0, 1.0]), (frames, 1, 1))
    for side in ("left", "right"):
        arrays[f"{side}_contact_confidence"] = np.zeros(frames)
    keypoints = np.zeros((frames, 17, 3))
    keypoints[..., 2] = 0.95
    keypoints[:, [11, 12], 1] = 70
    keypoints[:, [15, 16], 1] = 160
    boxes = np.tile([0.0, 0.0, 100.0, 200.0], (frames, 1))
    metadata = {"static_camera": True, "fps": 30.0}
    return arrays, metadata, keypoints, boxes


class SupportPoseTests(unittest.TestCase):
    def test_validates_without_mutating_inputs(self):
        arrays, metadata, keypoints, boxes = fixture()
        before = {k: v.copy() for k, v in arrays.items()}
        _validate_inputs(arrays, metadata, keypoints, boxes)
        for key, value in before.items():
            np.testing.assert_array_equal(arrays[key], value)

    def test_heatmap_scores_are_not_probabilities(self):
        arrays, metadata, keypoints, boxes = fixture()
        keypoints[..., 2] = 1.08
        _validate_inputs(arrays, metadata, keypoints, boxes)
        arrays["left_contact_confidence"][:] = 1.08
        with self.assertRaises(ValueError):
            _validate_inputs(arrays, metadata, keypoints, boxes)

    def test_requires_static_camera(self):
        arrays, metadata, kp, box = fixture()
        metadata["static_camera"] = False
        with self.assertRaisesRegex(ValueError, "static_camera"):
            _validate_inputs(arrays, metadata, kp, box)

    def test_rejects_dynamic_shape_and_mismatched_pose(self):
        for key in ("smpl_params_global.betas", "smpl_params_incam.body_pose"):
            arrays, metadata, kp, box = fixture()
            arrays[key][1, 0] = 0.1
            with self.assertRaises(ValueError):
                _validate_inputs(arrays, metadata, kp, box)

    def test_rejects_stale_evidence_and_repeated_refinement(self):
        for key in ("body_pose", "subject_scale", "foot_surface", "support_pose_refinement", "smplx_sequence"):
            arrays, metadata, kp, box = fixture()
            metadata[key] = {}
            with self.assertRaises(ValueError):
                _validate_inputs(arrays, metadata, kp, box)

    def test_rejects_sequence_arrays_even_without_metadata(self):
        arrays, metadata, kp, box = fixture()
        arrays["smplx.positions"] = arrays["pred_w_j3d"].copy()
        with self.assertRaisesRegex(ValueError, "SMPL-X sequence"):
            _validate_inputs(arrays, metadata, kp, box)

    def test_rejects_bad_observation_timeline_or_geometry(self):
        arrays, metadata, kp, box = fixture()
        for k, b in ((kp[:-1], box), (kp, box[:-1]), (kp, box * 0), (kp * np.nan, box)):
            with self.assertRaises(ValueError):
                _validate_inputs(arrays, metadata, k, b)

    def test_image_support_recovers_low_model_probability(self):
        _, _, kp, box = fixture()
        mask, image_mask, _ = _support_mask(np.zeros((60, 2, 3)), np.zeros((60, 2)), kp, box, 30.0)
        self.assertTrue(mask.all())
        self.assertTrue(image_mask.all())

    def test_foreshortened_lower_leg_is_not_missed(self):
        _, _, kp, box = fixture()
        kp[:, [15, 16], 1] = 110  # 20% bbox below hip, less than the rejected 25% rule.
        mask, _, _ = _support_mask(np.zeros((60, 2, 3)), np.zeros((60, 2)), kp, box, 30.0)
        self.assertTrue(mask.all())

    def test_fast_image_motion_or_low_visibility_adds_no_support(self):
        for moving in (True, False):
            _, _, kp, box = fixture()
            if moving:
                kp[:, [15, 16], 0] = np.arange(60)[:, None] * 5
            else:
                kp[:, [15, 16], 2] = 0.2
            mask, _, _ = _support_mask(np.zeros((60, 2, 3)), np.zeros((60, 2)), kp, box, 30.0)
            self.assertFalse(mask.any())

    def test_three_frame_support_is_removed(self):
        _, _, kp, box = fixture()
        kp[:, [15, 16], 2] = 0.1
        kp[20:23, [15, 16], 2] = 0.9
        mask, _, _ = _support_mask(np.zeros((60, 2, 3)), np.zeros((60, 2)), kp, box, 30.0)
        self.assertFalse(mask.any())

    def test_unknown_flight_keeps_relative_foot_height(self):
        surface = np.tile([0.02, 0.10], (180, 1))
        surface[40:140] += np.sin(np.linspace(0, np.pi, 100))[:, None] * 0.3
        mask = np.ones((180, 2), dtype=bool)
        mask[30:150] = False
        original = surface.copy()
        _, bias = _support_bias(surface, mask, 30.0)
        np.testing.assert_allclose(bias[60:120, 0], bias[60:120, 1])
        np.testing.assert_allclose(np.diff((surface - bias)[60:120], axis=1), 0.08)
        self.assertGreater((surface - bias)[90].min(), 0.25)
        np.testing.assert_array_equal(surface, original)

    def test_explicit_assumption_and_output_protection_precede_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "original.npz"
            p.write_bytes(b"keep-original")
            with self.assertRaisesRegex(ValueError, "assume-flat-ground"):
                export_support_pose(p, Path(tmp) / "new.npz", tmp)
            with self.assertRaises(FileExistsError):
                export_support_pose(p, p, tmp, assume_flat_ground=True)
            self.assertEqual(p.read_bytes(), b"keep-original")


if __name__ == "__main__":
    unittest.main()
