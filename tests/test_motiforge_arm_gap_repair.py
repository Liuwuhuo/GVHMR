"""No-model regressions for short, observed arm-gap reconstruction."""

import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

from hmr4d.backends.arm_gap_repair import interpolate_arm_gaps


def repair(start=30, stop=33, joint="right_wrist"):
    return {"joint": joint, "start_frame": start, "stop_frame_exclusive": stop}


def data(frames=70, dtype=np.float64):
    pose = np.zeros((frames, 63), dtype=dtype)
    kp = np.full((frames, 17, 3), 50, dtype=dtype)
    kp[..., 2] = 0.95
    crops = np.broadcast_to(np.array([50, 50, 100], dtype=dtype), (frames, 3)).copy()
    return pose, kp, crops


def columns(side):
    joints = (16, 18, 20) if side == "left" else (17, 19, 21)
    return np.array([3 * (joint - 1) + axis for joint in joints for axis in range(3)])


class ArmGapRepairTests(unittest.TestCase):
    def test_exact_gap_only_for_each_supported_joint(self):
        for joint in ("left_elbow", "left_wrist", "right_elbow", "right_wrist"):
            with self.subTest(joint=joint):
                pose, kp, crops = data()
                pose[:] = np.linspace(-0.2, 0.2, len(pose))[:, None]
                selected = columns(joint.split("_")[0])
                pose[30:33, selected] = 1.5
                result, evidence = interpolate_arm_gaps(pose, [repair(joint=joint)], kp, crops, 30)
                untouched = np.ones(pose.shape, dtype=bool)
                untouched[np.ix_(np.arange(30, 33), selected)] = False
                np.testing.assert_array_equal(result[untouched], pose[untouched])
                self.assertGreater(np.max(abs(result[30:33, selected] - pose[30:33, selected])), 1)
                self.assertEqual(evidence["changed_pose_frames"], 3)
                self.assertEqual(evidence["windows"][0]["anchor_frames"], [27, 28, 29, 33, 34, 35])
                self.assertEqual(evidence["windows"][0]["changed_frames"], [30, 31, 32])
                json.dumps(evidence, allow_nan=False)

    def test_no_input_mutation_and_output_copy_and_dtype(self):
        for dtype in (np.float32, np.float64):
            pose, kp, crops = data(dtype=dtype)
            pose[30:33, 48:51] = 0.7
            copies = pose.copy(), kp.copy(), crops.copy()
            result, _ = interpolate_arm_gaps(pose, [repair()], kp, crops, 30)
            for original, before in zip((pose, kp, crops), copies, strict=True):
                np.testing.assert_array_equal(original, before)
            self.assertEqual(kp[..., 2].tobytes(), copies[1][..., 2].tobytes())
            self.assertEqual(result.dtype, dtype)
            self.assertFalse(np.shares_memory(result, pose))

    def test_no_repairs_and_constant_pose_are_exact(self):
        pose, kp, crops = data()
        pose[:, 48:51] = [0, 0, 2 * np.pi + 0.2]
        for repairs in ([], [repair()]):
            result, evidence = interpolate_arm_gaps(pose, repairs, kp, crops, 30)
            np.testing.assert_array_equal(result, pose)
            self.assertFalse(np.shares_memory(result, pose))
            self.assertEqual(evidence["changed_pose_frames"], 0)

    def test_shortest_path_across_pi(self):
        pose, kp, crops = data()
        angles = np.deg2rad(149 + np.arange(len(pose)))
        truth = Rotation.from_rotvec(np.column_stack([angles * 0, angles * 0, angles]))
        pose[:, 54:57] = truth.as_rotvec()
        pose[30:33, 54:57] = 0
        result, _ = interpolate_arm_gaps(pose, [repair()], kp, crops, 30)
        reconstructed = Rotation.from_rotvec(result[30:33, 54:57])
        np.testing.assert_allclose((truth[30:33].inv() * reconstructed).magnitude(), 0, atol=1e-12)
        self.assertGreater(np.min(reconstructed.magnitude()), np.deg2rad(178))

    def test_noncommuting_rotations_follow_six_anchor_so3_spline(self):
        pose, kp, crops = data()
        anchors = np.array([27, 28, 29, 33, 34, 35])
        values = np.array([[.2, -.4, .1], [.5, -.1, .4], [.8, .2, .6],
                           [.3, .8, -.4], [.1, 1.0, -.5], [-.1, .7, -.6]])
        pose[anchors, 48:51] = values
        expected = RotationSpline((anchors - 29) / 30, Rotation.from_rotvec(values))(
            (np.arange(30, 33) - 29) / 30)
        result, _ = interpolate_arm_gaps(pose, [repair()], kp, crops, 30)
        actual = Rotation.from_rotvec(result[30:33, 48:51])
        np.testing.assert_allclose((expected.inv() * actual).magnitude(), 0, atol=1e-12)
        np.testing.assert_array_equal(result[anchors], pose[anchors])

    def test_duplicates_touching_and_overlapping_runs_are_merged(self):
        pose, kp, crops = data()
        pose[30:34, columns("right")] = 0.7
        expected, _ = interpolate_arm_gaps(pose, [repair(30, 34)], kp, crops, 30)
        for repairs in ([repair(30, 32), repair(32, 34, "right_elbow")],
                        [repair(30, 33), repair(31, 34), repair(30, 33)]):
            result, evidence = interpolate_arm_gaps(pose, repairs, kp, crops, 30)
            reversed_result, _ = interpolate_arm_gaps(pose, repairs[::-1], kp, crops, 30)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(reversed_result, expected)
            self.assertEqual(len(evidence["windows"]), 1)

    def test_nearby_known_gap_is_never_used_as_anchor(self):
        pose, kp, crops = data()
        pose[30:35, columns("right")] = 0.7
        result, evidence = interpolate_arm_gaps(pose, [repair(30, 32), repair(33, 35)], kp, crops, 30)
        np.testing.assert_array_equal(result, pose)
        self.assertEqual(len(evidence["windows"]), 2)
        self.assertTrue(all("anchors_overlap_known_gap" in item["reasons"] for item in evidence["windows"]))

    def test_bilateral_hits_are_independent_and_actual_changes_counted(self):
        pose, kp, crops = data()
        pose[30:33, columns("left")] = 0.7
        result, evidence = interpolate_arm_gaps(pose,
            [repair(joint="left_wrist"), repair(joint="right_elbow")], kp, crops, 30)
        self.assertEqual(evidence["changed_pose_frames"], 3)
        windows = {item["side"]: item for item in evidence["windows"]}
        self.assertEqual(windows["left"]["changed_pose_frames"], 3)
        self.assertEqual(windows["right"]["changed_pose_frames"], 0)
        np.testing.assert_array_equal(result[:, columns("right")], pose[:, columns("right")])

    def test_anchor_visibility_requires_all_three_arm_joints(self):
        for joint in (6, 8, 10):
            pose, kp, crops = data()
            pose[30:33, 48:51] = 0.7
            kp[28, joint, 2] = 0.699
            result, evidence = interpolate_arm_gaps(pose, [repair()], kp, crops, 30)
            np.testing.assert_array_equal(result, pose)
            self.assertIn("unreliable_arm_anchor_scores", evidence["windows"][0]["reasons"])
        # Native scores > 1 remain valid, and the threshold itself is inclusive.
        pose, kp, crops = data()
        kp[..., 2] = 1.02
        kp[28, 8, 2] = 0.7
        _, evidence = interpolate_arm_gaps(pose, [repair()], kp, crops, 30)
        self.assertTrue(evidence["windows"][0]["accepted"])

    def test_crop_edges_closed_image_edges_half_open(self):
        for xy, image, accepted, reason in (
            ([0, 50], (100, 100), True, None),
            ([100, 50], None, True, None),
            ([100, 50], (100, 100), False, "arm_anchors_outside_image"),
            ([-0.01, 50], None, False, "arm_anchors_outside_crop"),
        ):
            with self.subTest(xy=xy, image=image):
                pose, kp, crops = data()
                kp[28, 8, :2] = xy
                _, evidence = interpolate_arm_gaps(pose, [repair()], kp, crops, 30, image_size=image)
                self.assertEqual(evidence["windows"][0]["accepted"], accepted)
                if reason:
                    self.assertIn(reason, evidence["windows"][0]["reasons"])

    def test_insufficient_anchors_and_long_gaps_skip_without_expansion(self):
        pose, kp, crops = data()
        for run, reason in ((repair(0, 3), "insufficient_three_frame_anchors"),
                            (repair(2, 4), "insufficient_three_frame_anchors"),
                            (repair(66, 69), "insufficient_three_frame_anchors"),
                            (repair(30, 36), "gap_too_long")):
            result, evidence = interpolate_arm_gaps(pose, [run], kp, crops, 30)
            np.testing.assert_array_equal(result, pose)
            self.assertIn(reason, evidence["windows"][0]["reasons"])
            self.assertTrue(evidence["windows"][0]["skipped"])
            json.dumps(evidence, allow_nan=False)

    def test_float32_crop_edge_uses_native_lower_upper_comparison(self):
        pose, kp, crops = data(dtype=np.float32)
        crops[:] = [1766.5648193359375, 500, 676.5870361328125]
        kp[..., :2] = crops[:, None, :2]
        native_lower = crops[28, 0] - crops[28, 2] / np.float32(2)
        kp[28, 8, 0] = native_lower
        # At this real pixel scale abs(x-center) rounds differently from the
        # backend's inclusive native lower/upper bounds; the point is valid.
        self.assertGreater(abs(kp[28, 8, 0] - crops[28, 0]), crops[28, 2] / np.float32(2))
        _, evidence = interpolate_arm_gaps(pose, [repair()], kp, crops, 30, image_size=(1920, 1080))
        self.assertTrue(evidence["windows"][0]["accepted"])

    def test_maximum_gap_is_time_based(self):
        for fps, gap, accepted in ((30, 5, True), (30, 6, False), (60, 10, True), (60, 11, False)):
            pose, kp, crops = data()
            _, evidence = interpolate_arm_gaps(pose, [repair(30, 30 + gap)], kp, crops, fps)
            self.assertEqual(evidence["windows"][0]["accepted"], accepted)

    def test_invalid_shapes_values_timeline_and_fps_raise(self):
        pose, kp, crops = data()
        for invalid in (pose[:, :-1], pose[:0], pose[:-1], pose.reshape(70, 21, 3),
                        pose.astype(int), np.full_like(pose, np.nan), pose.astype(complex)):
            with self.assertRaises(ValueError):
                interpolate_arm_gaps(invalid, [repair()], kp, crops, 30)
        for invalid in (kp[:-1], kp[:, :-1], kp.astype(int), np.full_like(kp, np.inf)):
            with self.assertRaises(ValueError):
                interpolate_arm_gaps(pose, [repair()], invalid, crops, 30)
        for invalid in (crops[:-1], crops[:, :2], np.zeros_like(crops), np.full_like(crops, np.nan)):
            with self.assertRaises(ValueError):
                interpolate_arm_gaps(pose, [repair()], kp, invalid, 30)
        for fps in (0, -1, True, "30", [30], np.nan, np.inf):
            with self.assertRaises(ValueError):
                interpolate_arm_gaps(pose, [repair()], kp, crops, fps)

    def test_invalid_runs_and_image_contract_raise(self):
        pose, kp, crops = data()
        for invalid in (None, {}, [repair(-1, 2)], [repair(30, 30)], [repair(30, 71)],
                        [repair(30.0, 33)], [repair(True, 33)], [repair(joint="neck")], [{}]):
            with self.assertRaises(ValueError):
                interpolate_arm_gaps(pose, invalid, kp, crops, 30)
        for size in ([100], [100, 0], [100, np.inf], ["100", "100"], [True, True]):
            with self.assertRaises(ValueError):
                interpolate_arm_gaps(pose, [repair()], kp, crops, 30, image_size=size)


if __name__ == "__main__":
    unittest.main()
